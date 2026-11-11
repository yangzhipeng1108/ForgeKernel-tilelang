"""Ground-truth gate on the shipped sm120 NVFP4 kernel (flashinfer's
Sm120BlockScaledGemmKernel, which is the same warp-level block-scaled MMA path).

If the closed-form case returns 4*K, the instruction, the SF convention and the sm120 smem
plumbing are all proven correct on this RTX 6000D -- and the earlier NaN from our hand-built
kernel is our own layout bug, not the hardware or the DSL.
"""
import math

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda

from flashinfer.gemm.kernels.sm12x_cute.raw import Sm120BlockScaledGemmKernel

F4 = cutlass.Float4E2M1FN
SF = cutlass.Float8E4M3FN
F16 = cutlass.Float16
F32 = cutlass.Float32
SF_VEC = 16

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


def sf_to_atom_layout(sf, mn, sf_k):
    """Repack a (mn, sf_k) row-major SF tensor into the MMA atom order
    (l, ceil(mn/128), ceil(sf_k/4), 32, 4, 4) permuted by (3,4,1,5,2,0)."""
    atom_m = (32, 4)
    atom_k = 4
    pad_m = ((mn + 127) // 128) * 128
    pad_k = ((sf_k + atom_k - 1) // atom_k) * atom_k
    full = torch.zeros(1, pad_m, pad_k, dtype=sf.dtype, device=sf.device)
    full[0, :mn, :sf_k] = sf
    shape = (1, pad_m // 128, pad_k // atom_k, atom_m[0], atom_m[1], atom_k)
    return full.reshape(shape).permute(3, 4, 1, 5, 2, 0).contiguous()


def run(m, n, k, closed_form=True):
    dev = "cuda"
    torch.manual_seed(0)
    if closed_form:
        Aq, Bq, sv = torch.ones(m, k, device=dev), torch.ones(n, k, device=dev), 2.0
    else:
        Aq = to_fp4_grid(torch.randn(m, k, device=dev) * 2)
        Bq = to_fp4_grid(torch.randn(n, k, device=dev) * 2)
        sv = 1.0

    Ap = pack_fp4(Aq).view(torch.float4_e2m1fn_x2)
    Bp = pack_fp4(Bq).view(torch.float4_e2m1fn_x2)
    sf_k = (k + SF_VEC - 1) // SF_VEC
    sfa = torch.full((m, sf_k), sv, device=dev).to(torch.float8_e4m3fn)
    sfb = torch.full((n, sf_k), sv, device=dev).to(torch.float8_e4m3fn)
    sfa_a = sf_to_atom_layout(sfa, m, sf_k)
    sfb_a = sf_to_atom_layout(sfb, n, sf_k)
    C = torch.zeros(m, n, device=dev, dtype=torch.float16)
    alpha = torch.ones(1, device=dev, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    gemm = Sm120BlockScaledGemmKernel(
        acc_dtype=F32,
        sf_vec_size=SF_VEC,
        tile_shape_mnk=(128, 128, 128),
        epi_tile=(128, 32),
    )

    @cute.jit
    def launch(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, al_ptr, s):
        mA = cute.make_tensor(a_ptr, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
        mB = cute.make_tensor(b_ptr, cute.make_layout((n, k, 1), stride=(k, 1, 0)))
        mC = cute.make_tensor(c_ptr, cute.make_layout((m, n, 1), stride=(n, 1, 0)))
        # The kernel re-interprets SFA/SFB with tile_atom_to_shape_SF(a.shape, sf_vec),
        # so the passed layout must have the *tensor* shape (m, k, 1), not a flat count.
        mSFA = cute.make_tensor(sfa_ptr, cute.make_layout((m, k, 1), stride=(sf_k, 1, 0)))
        mSFB = cute.make_tensor(sfb_ptr, cute.make_layout((n, k, 1), stride=(sf_k, 1, 0)))
        mA_all = cute.make_tensor(al_ptr, cute.make_layout((1,)))
        gemm(mA, mB, mSFA, mSFB, mC, mA_all, 1, s)

    launch(make_ptr(F4, Ap.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
           make_ptr(F4, Bp.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
           make_ptr(SF, sfa_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
           make_ptr(SF, sfb_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
           make_ptr(F16, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
           make_ptr(F32, alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
           stream)
    torch.cuda.synchronize()

    ref = (Aq * sv) @ (Bq * sv).T
    md = (C.float() - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        C.float().flatten(), ref.flatten(), dim=0).item()
    print(f"m={m} n={n} k={k} closed={closed_form}: max|C|={C.abs().max():.4g} "
          f"ref_max={ref.abs().max():.4g} md={md:.4g} cos={cos:.6f}")
    if closed_form:
        print(f"  expected C == 4*{k} == {4*k}; C[0,0] = {C[0,0].item():.6g}")


if __name__ == "__main__":
    import traceback
    cases = [(128, 128, 128, True), (128, 128, 256, True), (256, 128, 512, True),
             (128, 256, 1024, True), (256, 256, 1024, False)]
    for case in cases:
        try:
            run(*case)
        except Exception:
            print("FAILED", case[:3])
            traceback.print_exc()
