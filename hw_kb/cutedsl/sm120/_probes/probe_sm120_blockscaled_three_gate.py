"""Numeric gate for all three sm120 block-scaled paths: NVFP4, MXFP4, MXFP8.

flashinfer's DenseGemmKernel (alias Sm120B12xBlockScaledDenseGemmKernel) dispatches to
MmaMXF4NVF4Op / MmaMXF4Op / MmaMXF8Op off sf_vec_size. This runs each and compares to a
dequantized fp32 reference, so the tree can state MXFP8 and MXFP4 as verified-or-not rather
than "the op constructs".

Run in the sglang container (cutlass 4.8.0):
    python3 blockscaled3_gate.py
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda

from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import DenseGemmKernel

F4 = cutlass.Float4E2M1FN
E4M3 = cutlass.Float8E4M3FN
E8M0 = cutlass.Float8E8M0FNU
F16 = cutlass.Float16
F32 = cutlass.Float32

_FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def to_fp4_grid(x):
    g = torch.tensor(_FP4, device=x.device)
    s = x.abs().clamp(min=1e-8)
    return g[torch.bucketize(s, g).clamp(1, 7) - 1] * x.sign()


def pack_fp4(q):
    m, k = q.shape
    a, sg = q.abs(), q.sign()
    nib = torch.zeros_like(a, dtype=torch.uint8)
    for c, v in enumerate(_FP4):
        nib = nib | ((a == v).to(torch.uint8) * c).to(torch.uint8)
    nib = nib | ((sg < 0).to(torch.uint8) << 3).to(torch.uint8)
    n2 = nib.view(m, k // 2, 2)
    return (n2[:, :, 0] | (n2[:, :, 1] << 4)).to(torch.uint8)


def make_sf_atom_layout(sf_bytes, mn, sf_k, torch_sf_dtype):
    """(mn, sf_k) row-major -> MMA atom order (l, mn/128, sf_k/4, 32, 4, 4) perm (3,4,1,5,2,0)."""
    atom_m = (32, 4)
    atom_k = 4
    pad_m = ((mn + 127) // 128) * 128
    pad_k = ((sf_k + atom_k - 1) // atom_k) * atom_k
    full = torch.zeros(1, pad_m, pad_k, dtype=torch_sf_dtype, device=sf_bytes.device)
    full[0, :mn, :sf_k] = sf_bytes
    shape = (1, pad_m // 128, pad_k // atom_k, atom_m[0], atom_m[1], atom_k)
    return full.reshape(shape).permute(3, 4, 1, 5, 2, 0).contiguous()


def _sf_torch_dtype(kind):
    if kind == "nvfp4":
        return torch.float8_e4m3fn
    # UE8M0: torch spells it float8_e8m0fnu on recent builds; fall back to raw bytes.
    if hasattr(torch, "float8_e8m0fnu"):
        return torch.float8_e8m0fnu
    return None


def run(kind, m, n, k, tile_mn=(128, 128)):
    dev = "cuda"
    torch.manual_seed(0)
    sf_vec = 16 if kind == "nvfp4" else 32
    sf_k = k // sf_vec

    if kind in ("nvfp4", "mxfp4"):
        aq = to_fp4_grid(torch.randn(m, k, device=dev) * 2)
        bq = to_fp4_grid(torch.randn(n, k, device=dev) * 2)
        a_pk = pack_fp4(aq).view(torch.float4_e2m1fn_x2)
        b_pk = pack_fp4(bq).view(torch.float4_e2m1fn_x2)
        a_dt = F4
        sf_dt = E4M3 if kind == "nvfp4" else E8M0
        ref = aq @ bq.T
    else:
        a8 = (torch.randn(m, k, device=dev) * 0.3).to(torch.float8_e4m3fn)
        b8 = (torch.randn(n, k, device=dev) * 0.3).to(torch.float8_e4m3fn)
        a_pk, b_pk = a8, b8
        a_dt = E4M3
        sf_dt = E8M0
        ref = a8.float() @ b8.float().T

    tdt = _sf_torch_dtype(kind)
    if tdt is None:
        # UE8M0 with no torch dtype: unit scale is the byte 0x7F in E8M0 (2^0)
        sfa_raw = torch.full((m, sf_k), 0x7F, dtype=torch.uint8, device=dev)
        sfb_raw = torch.full((n, sf_k), 0x7F, dtype=torch.uint8, device=dev)
        sfa_a = make_sf_atom_layout(sfa_raw, m, sf_k, torch.uint8)
        sfb_a = make_sf_atom_layout(sfb_raw, n, sf_k, torch.uint8)
    else:
        sfa = torch.ones(m, sf_k, device=dev).to(tdt)
        sfb = torch.ones(n, sf_k, device=dev).to(tdt)
        sfa_a = make_sf_atom_layout(sfa, m, sf_k, tdt)
        sfb_a = make_sf_atom_layout(sfb, n, sf_k, tdt)

    C = torch.zeros(m, n, device=dev, dtype=torch.float16)
    alpha = torch.ones(1, device=dev, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel = DenseGemmKernel(sf_vec_size=sf_vec, mma_tiler_mn=tile_mn,
                             cluster_shape_mn=(1, 1))

    @cute.jit
    def launch(a, b, sfa_p, sfb_p, c_p, al_p, s):
        mA = cute.make_tensor(a, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
        mB = cute.make_tensor(b, cute.make_layout((n, k, 1), stride=(k, 1, 0)))
        mSFA = cute.make_tensor(sfa_p, cute.make_layout((m, k, 1), stride=(sf_k, 1, 0)))
        mSFB = cute.make_tensor(sfb_p, cute.make_layout((n, k, 1), stride=(sf_k, 1, 0)))
        mC = cute.make_tensor(c_p, cute.make_layout((m, n, 1), stride=(n, 1, 0)))
        mAlpha = cute.make_tensor(al_p, cute.make_layout((1,)))
        kernel(mA, mB, mSFA, mSFB, mC, mAlpha, 1, s)

    launch(make_ptr(a_dt, a_pk.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
           make_ptr(a_dt, b_pk.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
           make_ptr(sf_dt, sfa_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
           make_ptr(sf_dt, sfb_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
           make_ptr(F16, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
           make_ptr(F32, alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
           stream)
    torch.cuda.synchronize()

    md = (C.float() - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        C.float().flatten(), ref.flatten(), dim=0).item()
    print(f"{kind:7s} m={m} n={n} k={k} tile={tile_mn}: md={md:.4g} cos={cos:.6f} "
          f"max|C|={C.abs().max():.4g} ref={ref.abs().max():.4g}", flush=True)


if __name__ == "__main__":
    import traceback
    for kind in ("nvfp4", "mxfp4", "mxfp8"):
        try:
            run(kind, 128, 128, 256)
        except Exception:
            print(f"--- {kind} FAILED ---", flush=True)
            traceback.print_exc()
