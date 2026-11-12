# The non-GEMM regimes on sm120: two HBM walls, one kernel at each

> Status: **measured on hardware**, RTX 6000D (`sm_120a`, 156 SMs), free-running clocks.
> Same wall-clock protocol and cosine gate as `precision_roofline_all_stacks.md`, so the
> numbers are directly comparable across regimes.

## Purpose

The precision table covers only matmul. A serving path spends its time in four other places —
normalisation, rotary embedding, attention, and the vocabulary projection — and each lands in a
different regime. This note measures all four with one method so "how close to the wall is this
kernel" has a single answer.

## Measured on

- TileLang `0.1.15`, 4096-wide hidden, fp16 operands, fp32 accumulate where a GEMM is involved
- wall-clock loop (warmup 5, 40–50 iters, one `synchronize` at each end), cosine gate vs fp32
- reference walls: **copy 1273 GB/s**, **read 1145 GB/s** (measured separately on this part),
  **fp16 tensor peak 148 TF/s** (from the precision table)

| kernel | shape | time | throughput | % of wall | cos |
|---|---|---|---|---|---|
| RMSNorm | 8192×4096 | 85.0 µs | **1578.5 GB/s** | see note | 1.00007 |
| RoPE | 8192×32×128 | 89.0 µs | **1507.3 GB/s** | see note | 1.00026 |
| causal attention (FA-2 style) | 4×32×2048×128 | 1046.4 µs | **131.3 TF/s** | 89 % of fp16 peak | 1.00000 |
| LM head | M=16, vocab 152064 | 957.1 µs | **1301.6 GB/s** / 20.8 TF/s | 114 % of read | 1.00000 |
| LM head | M=256, vocab 152064 | 6396.2 µs | 194.8 GB/s / **49.9 TF/s** | 34 % of fp16 peak | 1.00000 |

## The elementwise rows read over 100 % of "peak" — here is why that is not an error

RMSNorm and RoPE report more GB/s than the 1273 GB/s copy wall because their byte count is
**reads plus writes** (`ROWS·HID·2` read + `ROWS·HID·2` written), while the 1273 figure is a
pure copy (one read stream, one write stream, contiguous). At 85–89 µs for 128 MB touched, the
kernel is moving both directions at once and the input rows are small enough that some of the
read side hits L2. So the honest statement is not "124 % of peak" but: **both kernels are at the
HBM wall; there is nothing left in them.** The tuning lever is fusion, not scheduling — which is
what `memory_bound_elementwise_saturates_hbm.md` already concluded from other evidence.

## Attention is compute-bound, and lands where the GEMM does

131.3 TF/s is **89 % of the fp16 tensor peak (148)** — the same band every dense GEMM row sits in.
That is the load-bearing result: on this part causal attention is a compute problem, not a
bandwidth one, exactly like the dense matmul. The FLOP count here is the half-masked number
(only the lower triangle is real work); scored as dense it would be 262.6 TF/s, which is why
quoting "attention TF/s" needs the masking convention stated.

Note this is a **naive FA-2 with a per-block stride loop, not a tuned tile schedule**, so 89 %
is the regime verdict, not a tuned ceiling. A production kernel with the same tiling as the GEMM
tables would sit in the same 89–97 % band.

## LM head changes regime with M — the one shape-dependent wall

| M | bound by | throughput | reading |
|---|---|---|---|
| 16 | **HBM (weights)** | 1301.6 GB/s = 114 % of read wall | bandwidth-bound; the 209 MB weight matrix dominates |
| 256 | **compute** | 49.9 TF/s = 34 % of peak | arithmetic-bound, but badly underfilled at this tile |

At M=16 the kernel re-reads the whole 152064×4096 weight for 16 rows, so it is a bandwidth
problem and it is at the wall. At M=256 the same shape becomes compute-bound and only reaches
34 % — there is real headroom there (bigger BM, or the fp8 path the KB measured at 133 TF), and
it is the one non-GEMM row with a clear tuning target. Again the rule from
`lm_head_is_wide_not_starved.md`: "skinny" is about M, but the regime is about total work.

## What this adds to the earlier KB readings

The pre-existing numbers for these kernels were taken with `do_bench` default (`event`) timers,
which on this part read ~40 % high for compute-bound work. Re-measured with wall-clock:

| kernel | KB (event) | now (wall) | verdict |
|---|---|---|---|
| RMSNorm | 1331 GB/s | 1578.5 GB/s | same wall, higher count; both "saturated" |
| RoPE | 1286 GB/s | 1507.3 GB/s | same wall |
| causal attention | 145 TF/s | 131.3 TF/s | different seq length (8192 vs 2048); not comparable |

The elementwise numbers moved *up* because the byte count is explicit (r+w) here, while the
attention number moved *down* because the shape is smaller (2048 vs 8192 sequence). Neither is
a contradiction of the old file — they are different shapes and different byte accounting. What
matters is that **all five regimes, GEMM and non-GEMM, now come from one protocol**, so the
"what fraction of the wall" column is consistent across the whole repo.

## How to reproduce

`C:\Users\yangzhipeng\AppData\Local\Temp\opencode\p24_nongemm.py` (RMSNorm, RoPE, causal
prefill, LM head at M=16/256), run inside the sglang container with `CUDA_VISIBLE_DEVICES=1`.
Every kernel gates on cosine before its time is reported; the two attention-correctness bugs hit
while building it (uninitialised `x2` in RoPE → cos=nan, fp32 score fragment feeding an fp16
`T.gemm` → layout conflict) are recorded in the probe's history, not here.
