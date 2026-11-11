# sm120 kernels — the recipes that won, as reusable modules

These are not new kernels. Each file turns one measured recipe in `hw_kb/tilelang/sm120/`
into an importable module with the verdict baked into the API, so the next person does not
have to re-derive the tuning constants or repeat the correctness traps. Nine kernels cover
the Qwen3.8 serving path on this card.

Every file was run on the actual RTX 6000D (tilelang 0.1.15, torch 2.15.0+cu132) and
reproduced its recipe's numbers with `cos ~ 1.0` against a torch reference.

| kernel | recipe | what it wins |
|---|---|---|
| `gemm/gemm_splitk/sm120` | `split_k_fills_a_starved_grid.md` | fp8 split-K when the grid is starved; `pick_splitk()` refuses to split a full grid |
| `gemm/gemm_fp16/sm120` | `gemm_dtype_throughput.md`, `gemm_operand_layout_and_policy.md` | f16/bf16 baseline; `Square` policy (25 % fewer smem wavefronts) |
| `gemm/fp8_gemm/sm120` | `gemm_dtype_throughput.md` | fp8 e4m3 at BK=128 — the 2.1x lever over f16 |
| `gemm/lm_head/sm120` | `lm_head_is_wide_not_starved.md` | wide-N vocab projection; **no** split-K (measured to lose) |
| `attention/attn_fwd_causal/sm120` | `causal_attention_skips_the_masked_half.md` | bound the k-loop to the diagonal (~1.95x over full-square) |
| `attention/mla_paged_prefill/sm120` | `flash_attention_is_compute_bound_too.md`, `head_dim_decides_the_stage_budget.md` | prefill; stage count derived from head dim |
| `attention/gqa_paged_decode/sm120` | `decode_is_the_one_bandwidth_bound_case.md` | fp16 KV decode at dram ~90 %; fp8 KV is a loss |
| `attention/mla_paged_decode/sm120` | `mla_absorbed_decode_beats_mha.md` | one shared latent instead of per-head KV; 256x smaller cache |
| `linear_attention/rmsnorm/sm120` | `memory_bound_elementwise_saturates_hbm.md` | the one HBM-saturating regime; small tile (BM=2) wins |

## Readings on this box

    gemm fp8   512^2x16384   SK8   182.8 TF   (plain 47.6 TF)     md=0
               1024^2x8192   SK2   241.4 TF   (plain 169.2 TF)    md=0
               4096^3        SK1   374.7 TF   (split-K loses)     md=0
    gemm f16   4096^3 / 8192^3     158.8 / 139.9 TF              cos=1.000000
               bf16 same shape     159.1 / 141.7 TF              cos=0.999999
    fp8 gemm   4096^3 BK128        375.2 TF                      cos=1.000000
    lm head    M64 N128256 K4096   917.5us  73.3 TF   (fp8 133.2 TF, 1.8x)
               M256 same          1808.8us 148.7 TF
    prefill    H16 SEQ2048 D128     238.9us 143.8 TF vs SDPA 197.6us (0.83x)
    causal     2048/4096/8192      113.4 / 149.2 / 145.2 TF      cos=1.00000
    gqa decode 32h x 8kv, kv8192   147.3us  228 GB/s             cos=1.00000
    mla decode HQ128 kv8192 Dc128  123.3us   69.7 TF  2.1MB cache cos=1.00000
    rmsnorm    8192x2560 BM2       65.7us   1277.7 GB/s           cos=1.000000

## Three traps these modules already absorb

1. **`out_idx` cannot drive an accumulating kernel.** `gemm_splitk` compiles without
   `out_idx` and takes `C` as an explicit in/out argument. With `out_idx=[2]` the kernel
   returns a fresh buffer and never writes the `C` you zeroed, so the second launch
   accumulates onto the first (`md=455`, then `1.49e5`). The archived probe did this.
2. **Never put `tilelang.compile` inside a timing lambda.** An early version of the causal
   benchmark called the wrapper (which compiled) inside `do_bench` and reported 29686us for
   work that is 1900us. Each module caches its compiled kernel per shape.
3. **`from __future__ import annotations` breaks tilelang.** It stringifies annotations and
   tilelang's `get_type_hints()` then eval's them without the enclosing function's locals,
   giving `NameError: name 'k' is not defined`. None of these files import it.
4. **`do_bench` returns milliseconds, not seconds.** Two of the new benchmarks printed
   `t*1e6` and showed 1000x the real time; the GB/s figures were right because they used
   `t*1e-3`. If a latency looks impossible, check the unit before the kernel.
5. **`T.atomic_add` on a fragment is not a row reduction.** The first rmsnorm compiled and
   then died with `unspecified launch failure`; the fix is a full `(BM,D)` fragment plus
   `T.reduce_sum(..., dim=1)`, the same pattern the attention kernels use.

## Shape limits that are real, not incidental

- `gemm_splitk`: fp8 only; tile is fixed at 128x128x128 / stages 2.
- `gemm_fp16`: BK is 32 on purpose — BK is flat for 2-byte operands (141 -> 142 from 32 to
  48). Do not enlarge it expecting speed; use fp8.
- `fp8_gemm`: BK=128 is the measured best; BK=256 asks for 128 KiB and fails the 99 KiB
  opt-in ceiling. Precision is `cos=1.0` only on unit-normal inputs — re-measure on real
  activations with outliers.
- `lm_head`: no split-K. `N = vocab` makes the grid wide (2.5-6.4 waves), not starved.
- `attn_fwd_causal`: `SEQ` must be a multiple of `BM=64`; `BN=128` with `stages=2` exceeds
  dynamic smem, so stages is 1.
- `mla_paged_prefill`: `stages` is derived from `D` (2 if `D<=64`, else 1); stages=2 at D=128
  is an SMEM-cap failure, not a slow path.
- `gqa_decode`: fp16 KV on purpose (fp8 KV upcasts in SMEM and is slower). `BN` must stay 64;
  128 asks for 107520 B and fails.
- `mla_decode`: absorbed form with `Dc=128`. The production `Dc=512` does **not** fit one
  kernel — the up-projection needs a second pass. `QLEN` must be 16.
- `rmsnorm`: `BM=2` is deliberate. Larger tiles collapse it (BM=32 is 2.7x slower) because
  this is the one memory-bound regime and fatter blocks leave HBM idle.
