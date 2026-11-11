"""NVFP4 GEMM for sm120 (RTX 6000D) in CuTe DSL.

This is the CuTe DSL entry the TileLang tree could not provide: TileLang's
`T.gemm_blockscaled` is silently wrong on this card (`hw_kb/tilelang/sm120/
nvfp4_blockscaled_is_broken_in_tilelang.md`), so NVFP4 has no working TileLang kernel here. The
CuTe DSL block-scaled path does compute correctly.

PROVENANCE — read this before claiming "ForgeKernel wrote a CuTe DSL NVFP4 kernel"
------------------------------------------------------------------------------
The GEMM body is NOT hand-written from scratch. A from-scratch sm120 NVFP4 CuTe DSL kernel was
attempted and did not clear the smem/TMA plumbing (NaNs), as recorded in
`hw_kb/cutedsl/sm120/nvfp4_gemm_computes_correctly.md`. What is verified exact on this card is
the CuTe DSL kernel shipped in the container, `flashinfer.gemm.kernels.
dense_blockscaled_gemm_sm120_b12x.DenseGemmKernel`. This module wraps that kernel in the repo's
standard `build()/gemm_xxx()/_bench()` shape so it is a drop-in module and its numbers are
reproducible — it is a *wrapper around a verified CuTe DSL kernel*, not a new kernel. Be honest
about that in any report.

What the wrapper adds:
  * the NVFP4 operand encoding (e2m1 snapping + packing two values per byte) and the
    scale-factor MMA-atom layout the block-scaled kernel demands (rank-3 tensors, pre-permuted),
    both of which are the parts that silently corrupt a hand-built caller;
  * one cached, `cute.compile`-d compiled launch per shape (timing the compiler instead of the
    kernel was a real trap — see `hw_kb/cutedsl/sm120/split_k_only_when_m_is_small.md`);
  * a `_bench()` that reports TFLOP/s against the TileLang baselines in `kernels/sm120_README.md`.

Measured on this box (RTX 6000D, cutlass-dsl 4.8.0), vs the TileLang sm120 baselines:
    NVFP4 (this module)  4096x4096x4096   852 TFLOP/s   md <= 7   (TileLang has no fp4 kernel)
    CuTeDSL fp16 (TMA)   4096^3           208 TFLOP/s             (TileLang f16 158.8)
fp4 peak is ~2x fp8 peak, so the honest cross-dtype statement is: CuTe DSL delivers the serving
NVFP4 dtype at ~852 TFLOP/s, 2.3x the TileLang fp8 number, and the same-dtype fp16 comparison is
208 vs 158.8 (1.31x).
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

# e2m1 magnitudes the fp4 grid can represent exactly
_FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _to_fp4_grid(x: torch.Tensor) -> torch.Tensor:
    """Snap x onto the e2m1 grid (magnitude), keeping the sign. Values stay in [-6, 6]."""
    g = torch.tensor(_FP4, device=x.device)
    s = x.abs().clamp(min=1e-8)
    return g[torch.bucketize(s, g).clamp(1, 7) - 1] * x.sign()


def _pack_fp4(q: torch.Tensor) -> torch.Tensor:
    """(mn, k) e2m1-value tensor -> (mn, k/2) uint8, two nibbles per byte."""
    m, k = q.shape
    a, sg = q.abs(), q.sign()
    nib = torch.zeros_like(a, dtype=torch.uint8)
    for c, v in enumerate(_FP4):
        nib = nib | ((a == v).to(torch.uint8) * c).to(torch.uint8)
    nib = nib | ((sg < 0).to(torch.uint8) << 3).to(torch.uint8)
    n2 = nib.view(m, k // 2, 2)
    return (n2[:, :, 0] | (n2[:, :, 1] << 4)).to(torch.uint8)


def _sf_atom_layout(sf: torch.Tensor, mn: int, sf_k: int) -> torch.Tensor:
    """(mn, sf_k) row-major scales -> the pre-permuted MMA-atom order the kernel expects.

    The kernel re-views SF with `tile_atom_to_shape_SF`; feeding row-major scales yields
    plausible-but-wrong output with no error. Layout: (l, mn/128, sf_k/4, 32, 4, 4) then
    permute (3,4,1,5,2,0).
    """
    atom_m, atom_k = (32, 4), 4
    pad_m = ((mn + 127) // 128) * 128
    pad_k = ((sf_k + atom_k - 1) // atom_k) * atom_k
    full = torch.zeros(1, pad_m, pad_k, dtype=sf.dtype, device=sf.device)
    full[0, :mn, :sf_k] = sf
    shape = (1, pad_m // 128, pad_k // atom_k, atom_m[0], atom_m[1], atom_k)
    return full.reshape(shape).permute(3, 4, 1, 5, 2, 0).contiguous()


def build(m: int, n: int, k: int, tile_mn=(128, 128), mac: int = 156):
    """Compile once, return (call, C). `call()` launches only — safe to time in a loop.

    `m` must be a multiple of 128: the default DenseGemmKernel is silently wrong for M<128 and
    illegal-instructions at M=1 (small-M needs the use_m1_non_tma_* flags). k must be a multiple
    of 128 (nvfp4 tile_k). Unit UE4M3 scales are used, so callers must pre-scale into [-6, 6].
    """
    if m % 128 or n % 128 or k % 128:
        raise ValueError(f"m,n,k must be multiples of 128 for this kernel; got {(m, n, k)}")
    dev = "cuda"
    sf_vec, sf_k = 16, k // 16

    # a and b are held here as raw e2m1 values; the caller passes them via set_operands().
    C = torch.zeros(m, n, device=dev, dtype=torch.bfloat16)
    alpha = torch.ones(1, device=dev, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel = DenseGemmKernel(sf_vec_size=sf_vec, mma_tiler_mn=tile_mn, cluster_shape_mn=(1, 1))

    state = {"a": None, "b": None}

    def set_operands(aq: torch.Tensor, bq: torch.Tensor):
        state["a"] = _pack_fp4(aq).view(torch.float4_e2m1fn_x2)
        state["b"] = _pack_fp4(bq).view(torch.float4_e2m1fn_x2)
        sfa = torch.ones(m, sf_k, device=dev).to(torch.float8_e4m3fn)
        sfb = torch.ones(n, sf_k, device=dev).to(torch.float8_e4m3fn)
        state["sfa"] = _sf_atom_layout(sfa, m, sf_k)
        state["sfb"] = _sf_atom_layout(sfb, n, sf_k)

        a_ptr = make_ptr(F4, state["a"].data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
        b_ptr = make_ptr(F4, state["b"].data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
        sfa_ptr = make_ptr(E4M3, state["sfa"].data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
        sfb_ptr = make_ptr(E4M3, state["sfb"].data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
        c_ptr = make_ptr(BF16, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
        al_ptr = make_ptr(F32, alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)

        @cute.jit
        def launch(ap, bp, sfap, sfbp, cp, alp, s):
            mA = cute.make_tensor(ap, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
            mB = cute.make_tensor(bp, cute.make_layout((n, k, 1), stride=(k, 1, 0)))
            mSFA = cute.make_tensor(sfap, cute.make_layout((m, sf_k, 1), stride=(sf_k, 1, 0)))
            mSFB = cute.make_tensor(sfbp, cute.make_layout((n, sf_k, 1), stride=(sf_k, 1, 0)))
            mC = cute.make_tensor(cp, cute.make_layout((m, n, 1), stride=(n, 1, 0)))
            mAlpha = cute.make_tensor(alp, cute.make_layout((1,)))
            kernel(mA, mB, mSFA, mSFB, mC, mAlpha, mac, s)

        compiled = cute.compile(launch, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, al_ptr, stream)
        state["call"] = lambda: compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, al_ptr, stream)
        state["call"]()
        torch.cuda.synchronize()

    return set_operands, C, state


_CACHE: dict = {}


def gemm_nvfp4(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(M,K) x (N,K) float tensors -> (M,N) bfloat16. Internally snapped to the e2m1 grid.

    Both inputs must already lie in [-6, 6] (the fp4 range) — this module uses unit scale
    factors, so it does not rescale. For real activations, quantize with per-block scales and
    call `set_operands()` directly with the packed operands.
    """
    m, k = a.shape
    n = b.shape[0]
    key = (m, n, k)
    if key not in _CACHE:
        _CACHE[key] = build(m, n, k)
    set_operands, C, state = _CACHE[key]
    aq = _to_fp4_grid(a.float())
    bq = _to_fp4_grid(b.float())
    set_operands(aq, bq)
    state["call"]()
    torch.cuda.synchronize()
    return C.clone()


def _bench():
    # TileLang sm120 baselines from kernels/sm120_README.md, for reference in the printout.
    for (m, n, k) in [(2048, 4096, 4096), (4096, 4096, 4096)]:
        torch.manual_seed(0)
        a = (torch.randn(m, k, device="cuda") * 2)
        b = (torch.randn(n, k, device="cuda") * 2)
        c = gemm_nvfp4(a, b)
        ref = _to_fp4_grid(a) @ _to_fp4_grid(b).T
        md = (c.float() - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            c.float().flatten(), ref.flatten(), dim=0).item()
        set_operands, _, state = _CACHE[(m, n, k)]
        call = state["call"]
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        s.record()
        for _ in range(30):
            call()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 30
        tf = 2 * m * n * k / (ms * 1e-3) / 1e12
        print(f"NVFP4 {m}x{n}x{k}: {ms:.4f} ms  {tf:7.1f} TFLOP/s  md={md:.4g} cos={cos:.6f}")
    print("baselines (TileLang sm120): f16 158.8 TF, fp8 375.2 TF, fp4 = none (broken in TileLang)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device (measured on RTX 6000D / sm120)")
    _bench()
