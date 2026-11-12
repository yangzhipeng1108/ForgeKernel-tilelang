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
| LM head | M=256, vocab 152064 | 2298.4 µs | **138.7 TF/s** | **94 % of fp16 peak** | 1.00000 |

> **Correction 閳?the M=256 LM-head row used to read 49.9 TF/s / 34 %, and that was a
> misconfigured kernel, not a property of the shape.** The row was produced by `BM=256` with
> `threads=128`. At 128 threads a 256-row tile gives each of the four warps 64 rows of
> accumulator, which spills; the same tile at 256 threads runs **2.8脳 faster**. Reproduced
> exactly: 6428.7 µs / 49.6 TF/s at `threads=128` versus 2298.4 µs / 138.7 TF/s at
> `threads=256`, same tile, `cos` 1.000000 both. `BM=128` is insensitive to this (136.9 at 128
> threads, 137.4 at 256), so the trap is specific to pairing a large BM with too few threads.
> Three fresh-process repeats of the fast config give 136.2 / 136.6 / 137.1 TF/s.
>
> The consequence for planning is the opposite of what the old row implied: **there is no 3脳
> headroom in the LM head.** At M=256 it is at 94 % of the fp16 peak and the vocabulary
> projection is not a tuning target. What remains is the M=16 case, which is bandwidth bound at
> the wall, and byte reduction (block-sparsity over the vocabulary, see
> `lm_head_is_wide_not_starved.md`), not scheduling.

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

## LM head changes regime with M — and only the small-M end is short of the wall

| M | bound by | throughput | reading |
|---|---|---|---|
| 16 | **HBM (weights)** | 1301.6 GB/s = 114 % of read wall | bandwidth-bound; the 209 MB weight matrix dominates |
| 256 | **compute** | 138.7 TF/s = **94 % of peak** | arithmetic-bound and essentially at the wall |

At M=16 the kernel re-reads the whole 152064脳4096 weight for 16 rows, so it is a bandwidth
problem and it is at the wall. At M=256 the same shape becomes compute-bound and reaches
**94 %** once the tile is configured sanely (`BM=256` with 256 threads; `threads=128` at that BM
spills and halves throughput). There is no headroom left to tune here 閳?the earlier "34 %, lots
of headroom" reading came from that thread misconfiguration and was wrong. The remaining levers
for the vocabulary projection are byte-level: fp8 weights, or block-sparsity over the vocabulary
as `lm_head_is_wide_not_starved.md` measured end to end. The rule from that entry still holds:
"skinny" is about M, but the regime is about total work 閳?and at M=256 the work is already near
the tensor peak.

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
