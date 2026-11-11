"""Tune the sm120 NVFP4 GEMM: tile shape, mac, and the options that do nothing.

PROBE: PASS. Run inside the CuTe DSL container:

    docker exec cutedsl_probe bash -lc "cd /w && python3 probe_sm120_nvfp4_tuning.py"

Sweeps mma_tiler_mn, max_active_clusters and the DenseGemmKernel flags against the
shipped kernel, gating every timing run on md so the single_work_tile_per_cta
fabrication (5669 TF/s at md=1e3) cannot be reported as a win.

Numbers reproduced here are tabulated in
`../tuning_the_nvfp4_gemm.md`.
"""
import sys
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cuda.bindings.driver as cuda

from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import DenseGemmKernel

F4 = cutlass.Float4E2M1FN
E4M3 = cutlass.Float8E4M3FN
F32 = cutlass.Float32
BF16 = cutlass.BFloat16
FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _to_fp4_grid(x):
    g = torch.tensor(FP4, device=x.device)
    s = x.abs().clamp(min=1e-8)
    return g[torch.bucketize(s, g).clamp(1, 7) - 1] * x.sign()


def _pack_fp4(q):
    m, k = q.shape
    a, sg = q.abs(), q.sign()
    nib = torch.zeros_like(a, dtype=torch.uint8)
    for c, v in enumerate(FP4):
        nib = nib | ((a == v).to(torch.uint8) * c).to(torch.uint8)
    nib = nib | ((sg < 0).to(torch.uint8) << 3).to(torch.uint8)
    n2 = nib.view(m, k // 2, 2)
    return (n2[:, :, 0] | (n2[:, :, 1] << 4)).to(torch.uint8)


def _sf_atom_layout(sf, mn, sf_k):
    atom_k = 4
    pad_m = ((mn + 127) // 128) * 128
    pad_k = ((sf_k + atom_k - 1) // atom_k) * atom_k
    full = torch.zeros(1, pad_m, pad_k, dtype=sf.dtype, device=sf.device)
    full[0, :mn, :sf_k] = sf
    shape = (1, pad_m // 128, pad_k // atom_k, 32, 4, atom_k)
    return full.reshape(shape).permute(3, 4, 1, 5, 2, 0).contiguous()


def run(m, n, k, tile=(128, 128), mac=156, **kw):
    dev = "cuda"
    sf_vec, sf_k = 16, k // 16
    aq = _to_fp4_grid(torch.randn(m, k, device=dev) * 2)
    bq = _to_fp4_grid(torch.randn(n, k, device=dev) * 2)
    a_u8 = _pack_fp4(aq).view(torch.float4_e2m1fn_x2)
    b_u8 = _pack_fp4(bq).view(torch.float4_e2m1fn_x2)
    sfa_p = _sf_atom_layout(torch.ones(m, sf_k, device=dev).to(torch.float8_e4m3fn), m, sf_k)
    sfb_p = _sf_atom_layout(torch.ones(n, sf_k, device=dev).to(torch.float8_e4m3fn), n, sf_k)
    C = torch.zeros(m, n, device=dev, dtype=torch.bfloat16)
    alpha = torch.ones(1, device=dev, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kobj = DenseGemmKernel(sf_vec_size=sf_vec, mma_tiler_mn=tile,
                           cluster_shape_mn=(1, 1), **kw)

    ap = make_ptr(F4, a_u8.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    bp = make_ptr(F4, b_u8.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    spa = make_ptr(E4M3, sfa_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    spb = make_ptr(E4M3, sfb_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    cp = make_ptr(BF16, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    alp = make_ptr(F32, alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)

    @cute.jit
    def launch(ap_, bp_, spa_, spb_, cp_, alp_, s):
        mA = cute.make_tensor(ap_, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
        mB = cute.make_tensor(bp_, cute.make_layout((n, k, 1), stride=(k, 1, 0)))
        mSFA = cute.make_tensor(spa_, cute.make_layout((m, sf_k, 1), stride=(sf_k, 1, 0)))
        mSFB = cute.make_tensor(spb_, cute.make_layout((n, sf_k, 1), stride=(sf_k, 1, 0)))
        mC = cute.make_tensor(cp_, cute.make_layout((m, n, 1), stride=(n, 1, 0)))
        mAlpha = cute.make_tensor(alp_, cute.make_layout((1,)))
        kobj(mA, mB, mSFA, mSFB, mC, mAlpha, mac, s)

    try:
        compiled = cute.compile(launch, ap, bp, spa, spb, cp, alp, stream)
    except Exception as e:
        return f"tile={tile} mac={mac} COMPILE {type(e).__name__}"
    call = lambda: compiled(ap, bp, spa, spb, cp, alp, stream)
    try:
        call()
        torch.cuda.synchronize()
    except Exception as e:
        return f"tile={tile} mac={mac} RUN {type(e).__name__}"
    md = (C.float() - (aq @ bq.T)).abs().max().item()
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
    s0.record()
    for _ in range(30):
        call()
    e0.record()
    torch.cuda.synchronize()
    ms = s0.elapsed_time(e0) / 30
    tf = 2 * m * n * k / (ms * 1e-3) / 1e12
    flag = "  <-- WRONG ANSWER, ignore the TF/s" if md > 100 else ""
    return f"tile={tile} mac={mac} {ms:.4f}ms {tf:7.1f}TF md={md:.3g}{flag}"


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "tile"
    shapes = [(4096, 4096, 4096), (2048, 4096, 4096)]
    if which == "tile":
        for shape in shapes:
            print(f"=== tile sweep {shape} ===")
            for t in [(128, 128), (128, 256), (256, 128), (128, 64), (64, 128)]:
                print("  " + run(*shape, tile=t))
    elif which == "mac":
        for shape in shapes + [(4096, 4096, 2048), (8192, 4096, 4096)]:
            print(f"=== mac sweep {shape} ===")
            for mac in [64, 128, 156, 312]:
                print("  " + run(*shape, mac=mac))
    else:
        print(f"=== flags {shapes[1]} ===")
        for kw in [dict(), dict(use_prefetch=True), dict(swap_ab=True),
                   dict(tile_k=256), dict(load_path="cpasync"),
                   dict(single_work_tile_per_cta=True)]:
            print("  " + run(*shapes[1], **kw))
