"""split-K on sm120: correctness and whether it actually fills the grid.

Split-K is the standard fix for a starved grid -- when K is long but M*N produces too few
CTAs to fill 156 SMs, slicing K raises the CTA count at the cost of a cross-slice reduction.
Here the reduction is an **atomic bf16 scatter into gmem**, so two things bite:

  * the accumulator buffer must be bf16 (the kernel calls scatter_add_bf16 regardless of the
    C element type you pass -- an fp16 C silently reads as bf16 bits and produces garbage);
  * C must start at zero, because the atomics accumulate.

This probe runs the same GEMM at split_k = 1..16 and reports both correctness and time.

Run in the sglang container (cutlass 4.8.0):
    python3 probe_sm120_splitk.py
"""
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


def make_sf_atom_layout(sf, mn, sf_k):
    atom_m, atom_k = (32, 4), 4
    pad_m = ((mn + 127) // 128) * 128
    pad_k = ((sf_k + atom_k - 1) // atom_k) * atom_k
    full = torch.zeros(1, pad_m, pad_k, dtype=sf.dtype, device=sf.device)
    full[0, :mn, :sf_k] = sf
    shape = (1, pad_m // 128, pad_k // atom_k, atom_m[0], atom_m[1], atom_k)
    return full.reshape(shape).permute(3, 4, 1, 5, 2, 0).contiguous()


def build(m, n, k, split_k, tile_mn=(128, 128), mac=1):
    """Compile once; return (call, ref, C). call() only launches -- what timing measures."""
    dev = "cuda"
    torch.manual_seed(0)
    sf_vec, sf_k = 16, k // 16
    aq = to_fp4_grid(torch.randn(m, k, device=dev) * 2)
    bq = to_fp4_grid(torch.randn(n, k, device=dev) * 2)
    a_pk = pack_fp4(aq).view(torch.float4_e2m1fn_x2)
    b_pk = pack_fp4(bq).view(torch.float4_e2m1fn_x2)
    sfa_a = make_sf_atom_layout(torch.ones(m, sf_k, device=dev).to(torch.float8_e4m3fn), m, sf_k)
    sfb_a = make_sf_atom_layout(torch.ones(n, sf_k, device=dev).to(torch.float8_e4m3fn), n, sf_k)

    C = torch.zeros(m, n, device=dev, dtype=torch.bfloat16)
    alpha = torch.ones(1, device=dev, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel = DenseGemmKernel(sf_vec_size=sf_vec, mma_tiler_mn=tile_mn,
                             cluster_shape_mn=(1, 1), split_k_slices=split_k,
                             split_k_atomic_bf16=(split_k > 1))

    a_ptr = make_ptr(F4, a_pk.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    b_ptr = make_ptr(F4, b_pk.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfa_ptr = make_ptr(E4M3, sfa_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(E4M3, sfb_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    c_ptr = make_ptr(BF16, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    al_ptr = make_ptr(F32, alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)

    @cute.jit
    def launch(a_p, b_p, sfa_p, sfb_p, c_p, al_p, s):
        mA = cute.make_tensor(a_p, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
        mB = cute.make_tensor(b_p, cute.make_layout((n, k, 1), stride=(k, 1, 0)))
        mSFA = cute.make_tensor(sfa_p, cute.make_layout((m, sf_k, 1), stride=(sf_k, 1, 0)))
        mSFB = cute.make_tensor(sfb_p, cute.make_layout((n, sf_k, 1), stride=(sf_k, 1, 0)))
        mC = cute.make_tensor(c_p, cute.make_layout((m, n, 1), stride=(n, 1, 0)))
        mAlpha = cute.make_tensor(al_p, cute.make_layout((1,)))
        kernel(mA, mB, mSFA, mSFB, mC, mAlpha, mac, s)

    # cute.compile returns a callable that only launches -- compile once, call many.
    compiled = cute.compile(launch, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, al_ptr, stream)

    def call():
        C.zero_()                             # split-K accumulates into C, so reset per launch
        compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, al_ptr, stream)

    call()
    torch.cuda.synchronize()
    return call, aq @ bq.T, C


def timeit(call, iters=50):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        call()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == "__main__":
    # split_k_slices builds its grid as (1, split_k, num_n_tiles) -- see _compute_grid:
    #     grid = (1, split_k_slices, num_ctas_mnl[1])
    # The M-tile dimension is hardcoded to 1, so the split-K path only ever launches
    # tile-row 0 of M. That is correct exactly when M <= mma_tiler_mn[0] (128) and
    # silently drops every M-tile above the first otherwise -- visible as
    # cos ~ 1/sqrt(num_m_tiles) at a fantasy TF/s (it did 1/num_m_tiles of the work).
    #
    # The shapes below pin that threshold: M <= 128 is correct for every split and every
    # N; M = 256 / 512 breaks and stays broken no matter how split_k divides K.
    for m, n, k in ((128, 128, 4096), (256, 128, 4096), (128, 256, 4096),
                    (256, 256, 4096), (512, 512, 4096)):
        m_tiles = (m + 127) // 128
        n_tiles = (n + 127) // 128
        k_tiles = k // 128
        print(f"shape {m}x{n}x{k}   {m_tiles} M-tiles x {n_tiles} N-tiles, {k_tiles} k-tiles",
              flush=True)
        for sk in (1, 2, 4, 8):
            note = ""
            if sk > 1 and m_tiles > 1:
                note = f"  BROKEN: grid M-dim=1, drops {m_tiles - 1} M-tiles"
            elif sk > 1 and k_tiles % sk:
                note = f"  DROPS {k_tiles % sk} k-tiles ({k_tiles}%{sk}!=0)"
            mac = 156
            try:
                call, ref, C = build(m, n, k, sk, mac=mac)
                ms = timeit(call)
                Cm = C.float().clone()
                md = (Cm - ref).abs().max().item()
                cos = torch.nn.functional.cosine_similarity(
                    Cm.flatten(), ref.flatten(), dim=0).item()
                tflops = 2 * m * n * k / (ms * 1e-3) / 1e12
                print(f"  split_k={sk:2d}: md_vs_ref={md:.4g} cos={cos:.6f} "
                      f"max|C|={Cm.abs().max():.4g} {ms:.4f} ms  {tflops:.1f} TF/s{note}",
                      flush=True)
            except Exception as e:
                print(f"  split_k={sk}: FAILED {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:150]}", flush=True)
    print("SPLITK DONE", flush=True)
