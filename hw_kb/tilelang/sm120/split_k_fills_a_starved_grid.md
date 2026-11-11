# Split-K on sm120: the one scheduling axis that pays — when the tile grid is starved

> Status: **measured on hardware, clock-locked**, fp8 e4m3, tile `(128,128,128)`, `num_stages=2`,
> `do_bench(warmup=20, rep=60)`. Correctness checked with `cosine = 1.000000` on every row
> (`md ≤ 1e-3`, fp32 accumulate).

## Purpose

A plain GEMM launches one block per `(M/BM)×(N/BN)` tile. When M and N are small, that grid is
smaller than the SM count and most of the machine sits idle — the kernel is **tile-starved**. Split-K
cuts the K axis into `SK` slices, gives each `(tile, slice)` its own block, and reduces the partials
with an atomic add. It trades a reduction for occupancy.

This is the follow-on to `block_scheduling_is_a_tie.md`: raster / L2-swizzle / persistent do **not**
create work, so they cannot help a starved grid. Split-K can, because it *multiplies the block count*.

## Measured on

- TileLang `0.1.15`, `sm_120a`, RTX 6000D, SM clock locked at 2000 MHz, **156 SMs**
- Grid fill (`tiles × SK / 156`):

  | shape | tiles | fill | SK2 | SK4 | SK8 | SK16 |
  |---|---|---|---|---|---|---|
  | 512×512×16384 | 16 | **10 %** | 21 % | 41 % | 82 % | 164 % |
  | 1024×1024×8192 | 64 | **41 %** | 82 % | 164 % | 328 % | 656 % |
  | 4096³ | 1024 | **656 %** | 1313 % | … | … | … |

## The finding: split-K is a 2–4× win exactly where fill is under ~100 %, a loss where it is over

TFLOP/s (all `cos = 1.0`):

| shape | plain | SK=2 | SK=4 | SK=8 | SK=16 |
|---|---|---|---|---|---|
| **512×512×16384** (10 % fill) | 46.4 | 84.4 | 135.8 | **183.5** | 178.5 |
| **1024×1024×8192** (41 % fill) | 171.8 | **247.0** | 219.7 | 202.8 | 183.3 |
| 4096³ (656 % fill) | **372.8** | 308.0 | 276.2 | 209.2 | 144.5 |

- Starved shape, 10 % fill: **46 → 183.5 = 4.0×** at SK=8 (fill 82 %). SK=16 overshoots (164 %) and
  *drops* to 178.5 — the extra slices add phases and atomic traffic faster than they add occupancy.
- Half-starved shape, 41 % fill: best is **SK=2** (247, fill 82 %); more slices already lose, because
  the reduction cost grows with SK while the fill gain is already spent by 82 %.
- Large shape, 656 % fill: split-K is **strictly worse** (372.8 → 144.5). There is nothing to fill;
  every extra slice is pure reduction overhead and extra global write traffic.

**The optimal SK is the one that brings fill to roughly one wave (≈ 100 %), no more.** Both
starved shapes peak near 82 % fill and fall off past it.

## Why the plain kernel cannot do this itself

It can't — the grid is `(N/BN, M/BM)` and there is no third axis. The block-scheduling recipes
(raster, `T.use_swizzle`, `T.PersistentTileScheduler`) only reorder or pin the *same* 16 blocks
onto 156 SMs; they measured within ±2 % on the same starved shape (see
`block_scheduling_is_a_tie.md`). Split-K is the only lever that changes the block **count**.

## Correctness: the reduction, and the one trap

There is **no first-class `T.SplitK`** in 0.1.15 — write it by hand:

```python
with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), SK, threads=256) as (bx, by, sz):
    k0 = sz * per                       # per = ceildiv(ceildiv(K, BK), SK)
    ...                                 # accumulate this slice into acc
    T.atomic_add(C[by * BM, bx * BN], acc)
```

`T.atomic_add(dst_slice, fragment)` lowers to a per-element atomic (a TMA `cp.reduce` path exists
but the elementwise form is what the tile-region call here emits). C must be **float32** and
**zeroed before the kernel**.

> **The trap that cost a wrong answer:** `out_idx=[2]` zeroes the output **once** (on the first
> call) and then reuses the same buffer. A split-K kernel accumulates with `atomic_add`, so the
> second launch adds onto the first launch's result. The first run is correct, every run after it
> is garbage (`cos` collapses to 0.5). Symptoms: `run1 md=3e-4, run2 md=6e2`. **For any
> `atomic_add` kernel, pass an explicitly zeroed C every call** (or `zero_()` it inside the timed
> region) — never rely on `out_idx` allocation to reset an accumulating output.

With C zeroed each call, all rows above are `cos = 1.000000`, `md ≤ 1e-3`.

### Revalidation: the archived probe had this bug, and `zero_()` alone does not fix it

`_probes/fk_probes_20261111_0247/sk.py` called `C.zero_()` before each launch and still
produced `md=455` / `md=1.49e5`. The reason is sharper than "forgot to zero": with
`out_idx=[2]` the kernel **returns a fresh buffer** and never writes the `C` the caller
zeroed, so `C.zero_()` resets an object the kernel does not read. The fix is to compile
**without** `out_idx` and pass `C` as an explicit in/out argument:

```python
k = tilelang.compile(builder, target="cuda")     # no out_idx
C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
def run():
    C.zero_()                                    # now this is the buffer the kernel accumulates into
    k(A, B, C)
```

`_probes/fk_probes_20261111_0247/sk_fixed.py` is that fix, and its readings on the RTX 6000D
(tilelang 0.1.15) agree with the table above:

| shape | plain | SK=2 | SK=4 | SK=8 |
|---|---|---|---|---|
| 512²×16384 | — | 84.4 TF | 133.6 TF | **187.5 TF** |
| 1024²×8192 | 170.4 TF | **247.3 TF** | 215.7 TF | 204.7 TF |
| 4096³ | **375.3 TF** | 310.6 TF | 277.0 TF | 208.4 TF |

Every row `md ≤ 2.8e-4`. The shape of the result is the entry's claim: split-K wins on the
starved `512²×16384` (10 % fill) and loses once the grid is full (4096³).

## When needed (trigger)

- **Tile grid under ~100 % of the SMs** (`(M/BM)×(N/BN) < num_sms`) **and K large enough to slice**
  (`K/BK ≥ SK`, and each slice still a few pipeline iterations deep). This is the LM-head /
  skinny-GEMM / `512²×16k` regime.
- Formula to pick SK: `SK ≈ ceil(num_sms / num_tiles)`, then round down if that overshoots a wave.
  Grid fill between ~80 % and ~120 % is the sweet spot; do not push past it.
- **Do not split-K a well-filled GEMM.** Above ~100 % fill it is a monotone loss (4096³: 372 → 144).
- Split-K pairs badly with the block schedulers here (they're already a tie); it pairs *well* with
  the fp8 dtype win (`gemm_dtype_throughput.md`) — the starved shape is fp8 throughout above.

## Minimal runnable example

```python
import tilelang
import tilelang.language as T

M, N, K = 512, 512, 16384
BM = BN = BK = 128
SK = 8                                   # ~ceil(156 / 16) -> fill 82 %
PER = (K // BK + SK - 1) // SK
DT = "float8_e4m3"


@T.prim_func
def splitk(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT),
           C: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), SK, threads=256) as (bx, by, sz):
        a_s = T.alloc_shared((BM, BK), DT)
        b_s = T.alloc_shared((BK, BN), DT)
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)
        k0 = sz * PER
        for kk in T.Pipelined(PER, num_stages=2):
            k = k0 + kk
            if k < K // BK:
                T.copy(A[by * BM, k * BK], a_s)
                T.copy(B[k * BK, bx * BN], b_s)
                T.gemm(a_s, b_s, acc)
        T.atomic_add(C[by * BM, bx * BN], acc)


kernel = tilelang.compile(splitk, target={"kind": "cuda", "arch": "sm_120a"})   # no out_idx
C = __import__("torch").zeros(M, N, device="cuda", dtype=__import__("torch").float32)
kernel(A, B, C)                          # zero C before EVERY call
```

## What TileLang decides for you here (and what it does not)

**Decided by you:** everything — the K split, the slice bounds, the `atomic_add` (or a two-pass
workspace reduction), and zeroing C. There is no scheduler primitive for this.

**Decided by the pass:** the ordinary pipeline / TMA / warpgroup machinery of each slice's inner
`T.Pipelined` — unchanged from a plain GEMM.

## Gotchas

- **`out_idx` + `atomic_add` = wrong after the first call.** Zero C yourself every launch. This is
  the whole reason split-K results looked broken at first.
- **C must be float32.** Accumulating `float16` partials across slices loses precision badly; the
  fp8 inputs already carry the operand rounding, keep the sum in fp32.
- **`per = ceil(ceil(K/BK) / SK)` and guard `k < ceil(K/BK)`** — a non-divisible K leaves a short
  last slice; without the guard it reads out of bounds of `A`/`B`.
- **Over-splitting reverses the win** (SK=16 at 10 % fill drops 183 → 178 and at 41 % fill drops
  far more). Stop at ~one wave of fill.
- **The reduction cost is real** — it is why the large GEMM loses. `atomic_add` over a whole
  `(128,128)` fp32 tile × SK slices is global write traffic that a plain GEMM does not pay.

## Sources

1. TileLang 0.1.15: `tilelang/language/atomic.py` (`atomic_add`, the optional TMA `cp.reduce`
   lowering), `tilelang/carver/roller/hint.py` (`split_k_factor` — an auto-scheduler hint only,
   not a codegen primitive; there is no `T.SplitK`).
2. `block_scheduling_is_a_tie.md` (the schedulers that *cannot* fill a starved grid),
   `gemm_dtype_throughput.md` (fp8 = 2.4×, the dtype these shapes run in).
3. This repo's probes on an RTX 6000D, clock-locked, grid-fill computed against 156 SMs.
