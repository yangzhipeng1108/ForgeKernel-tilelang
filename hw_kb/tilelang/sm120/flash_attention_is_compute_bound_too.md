# FlashAttention on sm120: compute-bound too, and the GEMM lessons do (and do not) transfer

> Status: **measured on hardware, clock-locked at 1995 MHz**, fp16/fp8, NCU census included.
> Correctness `cos = 1.00000` (f16) against `torch.nn.functional.scaled_dot_product_attention`.

## Purpose

The GEMM map closed 11 of 15 axes on the premise that a dense GEMM is compute-bound
(`dram_pct ≈ 13 %`). Attention adds a **softmax between two GEMMs** and an **online rescale** that
writes every accumulator row each K-block. The question this file answers: does that open a new
compute/bandwidth balance, and which GEMM lessons carry over?

Short answer: **attention on this arch is still compute-bound** (`dram_pct ≈ 8.8 %`), so the
scheduling axes stay dead — but the *reason* the kernel is below the tensor peak is different
(register pressure → 1 block/SM → 16 % warps active → 22 % `stall_wait`), and the two GEMM wins
that carry over are **fp8** and **short-sequence grid shaping**.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 1995 MHz, 156 SMs
- `H = 16` heads, `SEQ = 2048`, `D = 128`, fp16 (and fp8 e4m3), causal = off
- Baseline: `torch.nn.functional.scaled_dot_product_attention` (SDPA), same f16 inputs
- TileLang: FlashAttention-2 forward (no warp specialization, single consumer warpgroup)

## The kernel (FA-2 forward, the only gotcha is exp2's base)

```python
scale = 1.0 / sqrt(D) * log2(e)          # <- exp2 needs the base-2 scaling, not 1/sqrt(D)
...
S = T.alloc_fragment((BM, BN), "float32")
T.gemm(Qs, Ks, S, transpose_B=True, clear_accum=True)
# online softmax: rowmax -> m_i, rescale acc by exp2(m_prev - m_i), P = exp2(S - m_i)
T.reduce_max(S, rowmax, dim=1, clear=True)
for i, j in T.Parallel(BM, BN):
    Ps[i, j] = T.exp2(S[i, j] - m_i[i])   # Ps is fp16 shared, fed to the PV gemm
T.gemm(Ps, Vs, acc)
```

**Forgetting `log2(e)` is the classic bug**: with `scale = 1/sqrt(D)` the kernel compiles, runs, and
returns `cos ≈ 0` (a plausible-magnitude but wrong matrix). With it, `md = 1.7e-4`, `cos = 1.0`.

## Finding 1: it is compute-bound, so the scheduling axes stay closed

NCU on the best config (`BM=64, BN=128`, 2048²·D=128, fp16):

| metric | value |
|---|---|
| `dram__throughput` | **8.77 %** |
| `tensor_pipe_active` (`sm__pipe_tensor_cycles_active`) | **56.6 %** |
| `sm__throughput` | 46.4 % |
| `sm__warps_active` | **15.8 %** |
| `launch__occupancy_limit_registers` | **1 block** |
| `launch__registers_per_thread` | **240** |
| `stall_wait` | **22.4 %** |
| `stall_short_scoreboard` | 4.6 % |
| `stall_barrier` | 2.4 % |
| `stall_mio_throttle` | 0.7 % |

So `dram_pct ≈ 8.8 %` — even lower than GEMM's 13 %, and L2-block scheduling, swizzle, persistent,
eviction are all still ties (the GEMM results in `block_scheduling_is_a_tie.md` carry over
unchanged). **The softmax does not open a bandwidth problem on this arch.**

What *is* new is the ceiling's cause: **240 registers/thread forces 1 block per SM**, warps_active
is only 15.8 %, and the dominant stall is `stall_wait` at 22.4 % — waiting on fixed-latency
dependent instructions (the `exp2` → subtract → reduce chain). That is a **latency-coverage** problem,
not a bandwidth one. This is the attention-specific axis, and it is not the same knob as anything in
the GEMM map.

## Finding 2: tile size — the opposite instinct from GEMM helps a little, then hurts

fp16, `SEQ=2048`, TFLOP/s (SDPA baseline = 347):

| BM | BN | thr | stages | TFLOP/s | vs SDPA | md |
|---|---|---|---|---|---|---|
| **64** | **128** | 128 | **1** | **286** | **0.82×** | 8.2e-5 |
| 64 | 64 | 128 | 2 | 239 | 0.69× | 7.9e-5 |
| 64 | 64 | 128 | 1 | 251 | 0.72× | 7.9e-5 |
| 128 | 64 | 128 | 1 | 244 | 0.70× | 7.9e-5 |
| 48 | 128 | 128 | 1 | 245 | 0.71× | 8.2e-5 |
| 32 | 128 | 128 | 1 | 238 | 0.69× | 8.2e-5 |
| 16 | 128 | 128 | 1 | 133 | 0.38× | 8.2e-5 |

Two things differ from GEMM:

- **`num_stages = 1` wins** (251 vs 239 at BM=64/BN=64). Attention's shared footprint is Q + K + V +
  P, so `num_stages=2` does not fit the bigger tiles at all (`BK`-doubling overflows 100 KiB:
  `BM128 BN64 → 115712 B`). GEMM wanted stages=2; attention wants 1 and a wider `BN`.
- **`BN > BM` is the right shape** (`BM=64, BN=128` beats `BM=128, BN=64`): the score matrix
  `S (BM×BN)` and the `exp2` work scale with `BN`, but a wide `BN` amortizes the per-iteration
  rescale of `acc (BM×D)` over more columns.

The 0.82× gap to SDPA is **not** a tile-sweep problem — SDPA uses warp specialization (separate
softmax warps) that this hand-written kernel does not, which is exactly the mechanism that is
counterproductive on the *GEMM* here (`warp_specialization_is_already_on.md`) but is the standard
attention design. Not closed in this measurement.

## Finding 3: fp8 transfers — +14 % (and it is the only positive axis ported from GEMM)

fp8 e4m3 Q/K/V, fp32 softmax, best tile:

| dtype | tile | TFLOP/s | cos | md |
|---|---|---|---|---|
| fp16 | 64×128, st1 | 286.1 | 1.00000 | 8.2e-5 |
| **fp8 e4m3** | 64×128, st1 | **324.9** | 0.99970 | 5.3e-3 |
| fp8 e4m3 | 128×64, st1 | 315.2 | 0.99970 | 5.2e-3 |

**+14 %**, reaching 0.94× of SDPA. The `cos = 0.9997` / `md ≈ 5e-3` is the fp8 operand rounding in
QK^T (softmax itself stays fp32); for attention that is a normal trade, but validate on real
activations with the project's scaling, not on `randn` — the same caveat as `gemm_dtype_throughput.md`.
Note fp8 is a **smaller** win here than in GEMM (2.4×) because the kernel is at 56 % tensor pipe
with a latency bottleneck, not a pure math-throughput one.

## Finding 4: short sequences — the grid-shaping win flips to our favour

At `SEQ = 512` (16 heads, D=128):

| | TFLOP/s |
|---|---|
| SDPA | 110 |
| hand-written FA (64×128, st1) | **150** |
| ratio | **1.36×** |

At small `SEQ` the grid `(SEQ/BM) × H` is tiny and SDPA's fixed per-launch overhead dominates, while
the hand-written kernel's grid is exactly `(SEQ/BM, H)` and carries no scheduler. This is the
**same lesson as split-K / block-tile skipping**: at small shapes, *grid shaping* beats a general
kernel — but here it comes for free because we wrote the grid by hand, not from a scheduling knob.

## Finding 5: the large-SEQ gap to SDPA is structural — neither WS nor register pressure closes it

The obvious next move after Finding 1 (240 regs → 1 block/SM → 15.8 % warps active) is to add
warps or shrink registers. Both were measured and **both fail**:

**Warp specialization — a tie, not a win.** Enabling `tl.enable_auto_warp_specialization: "role_based"`
(which *hurts* the GEMM by 2–8×, `warp_specialization_is_already_on.md`) on the attention kernel:

| config | `__launch_bounds__` | `reg_dealloc` | TFLOP/s | vs SDPA |
|---|---|---|---|---|
| no WS, thr=128 | 256 | 1 | 285.7 | 0.82× |
| WS, thr=128 | 256 | 2 | **286.3** | 0.83× |
| WS, thr=256 | 384 | 2 | 261.4 | 0.75× |
| WS, thr=256, BM128 | 384 | 2 | 155.7 | 0.45× |

The `role_based` scheduler is a generic producer/consumer split; it gives the attention kernel the
same ties-or-losses it gives the GEMM. **The "same mechanism, opposite conclusion" hypothesis is
false** — WS is not the thing SDPA has that this kernel lacks. (SDPA's softmax specialization is a
hand-written producer/consumer *role assignment specific to attention*, not what `role_based`
derives.)

**Register pressure — the tile needs the registers.** Forcing a smaller consumer budget:

| `annotate_consumer_reg_alloc` | TFLOP/s | vs SDPA |
|---|---|---|
| none (natural: 240) | **285.4** | 0.82× |
| 240 (explicit) | 286.5 | 0.83× |
| 160 | 268.0 | 0.77× |
| 128 | 211.1 | 0.61× |
| 112 | 178.9 | 0.52× |

Squeezing registers is strictly worse: `S (64×128 fp32)` already costs ~64 regs/thread and
`acc (64×128 fp32)` another ~64, so below ~240 the compiler spills. **The 240 registers are the tile,
not slack** — occupancy cannot be bought by lowering them.

So the 0.82× f16 / 0.94× fp8 gap to SDPA is **structural for this tile geometry**, not a knob. It
would need either a different softmax placement (a hand-written attention-specific role split) or a
smaller `BM` (which the tile sweep already showed loses: `BM=32` → 0.69×). Recorded as a closed axis
rather than a TODO.

## When needed (trigger)

- **Use the hand-written FA-2 for small/medium shapes** (`SEQ ≤ ~1k`, decode, or any case where
  `(SEQ/BM)*H` is small): it beats SDPA (1.36× at SEQ=512) with no extra machinery.
- **For large `SEQ` (2k+), expect ~0.82× f16 / 0.94× fp8 vs SDPA and stop there.** Both levers people
  reach for (warp specialization, register reduction) measured as a tie and a loss (Finding 5); the
  gap is structural for this tile, not a missing knob.
- **Do not reach for L2 scheduling / swizzle / persistent / eviction on attention** — `dram_pct` is
  8.8 %, so those stay ties exactly as in the GEMM map.
- **fp8 QK^T is the free ~14 %** if numerics allow; keep softmax in fp32.
- If closing the large-SEQ gap matters, the remaining option is a **hand-written attention-specific
  warp role** (pin the softmax to its own warps and the two GEMMs to the tensor warps) — the generic
  `role_based` scheduler does not do this, and neither does dropping registers.

## Gotchas

- **`exp2` needs `scale = 1/sqrt(D) * log2(e)`.** Without it the kernel runs and is wrong
  (`cos ≈ 0`). This is the single most common FA reimplementation bug.
- **`T.gemm` cannot mix fp32 and fp16 operands** (`Mixed-dtype GEMM only supports f8f6f4 family`).
  Keep `S` fp32 for the softmax, but write `P` into an **fp16 shared** buffer and do the PV GEMM
  from that.
- **`num_stages=2` does not fit**: Q+K+V+P at `BM=128, BN=64` is 115 712 B, over the 100 KiB cap.
  Use `stages=1` and a wider `BN`.
- **The two GEMMs in one body do not pipeline the same way**: the first `T.gemm(Qs, Ks, S)` is
  `clear_accum=True` (fresh scores), the second accumulates into `acc`. Do not `clear` `acc` inside
  the loop.
- **SDPA's `scaled_dot_product_attention` uses its own flash/mem-efficient backend**; comparing
  against it is fair only at the same dtype and layout (both had fp16 inputs here).

## Sources

1. TileLang 0.1.15: `T.reduce_max` / `T.reduce_sum` / `T.exp2` / `T.gemm(clear_accum=)`
   (`tilelang/language/*.py`); the mixed-dtype GEMM restriction from the runtime error text.
2. `block_scheduling_is_a_tie.md` (the scheduling axes that carry over as ties),
   `gemm_dtype_throughput.md` (fp8, and the randn-accuracy caveat),
   `split_k_fills_a_starved_grid.md` (grid shaping at small shapes — the same lesson as Finding 4).
3. This repo's probes on an RTX 6000D, clock-locked, NCU census via `ncu -k regex:main_kernel`.
