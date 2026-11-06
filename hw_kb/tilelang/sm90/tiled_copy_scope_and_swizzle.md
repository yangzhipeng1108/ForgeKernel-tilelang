# `T.copy` between scopes, and what `T.annotate_layout` does and does not pin

> Status: **skeleton — the mechanism is documented, the readings are not taken yet.**
> Every `[to be measured]` below must be produced on a real box before this counts as
> evidence. The API surface and its parameters are read off the installed 0.1.15 source.

## Purpose

Move a tile between global, shared and `fragment` memory with one call the compiler maps onto
threads, and — when the default layout is the wrong one — pin the layout of a *named* buffer
so the copy and the consumer agree on it.

## Measured on

- TileLang version: `0.1.15`
- arch: `[to be measured]` (this file is under `sm90/`)
- Certification: `[to be measured]` — a `PROBE PASS` job id, or the NCU census axis that moved
  (`smem_wavefronts_st`/`_ld`, bank-conflict counts, before → after).

## When needed (trigger)

- **smem traffic over the line**: `smem_wavefronts_st` / `smem_wavefronts_ld` above what the
  tile volume implies — the first question is whether the layout is the one you asked for.
- **A layout you annotated appears to be ignored.** `T.annotate_layout` is consulted only for
  the buffers it names, and index inference may still choose another for everything else.
- **A `T.copy` raises a shape or dtype complaint** that reads like your source is wrong.

## Minimal runnable example

```python
import tilelang
import tilelang.language as T


@T.prim_func
def copy_and_pin(A: T.Tensor((4096, 4096), "float16"),
                 B: T.Tensor((4096, 4096), "float16")):
    BM, BN, BK = 128, 128, 32
    with T.Kernel(T.ceildiv(4096, BN), T.ceildiv(4096, BM), threads=128) as (bx, by):
        a_s = T.alloc_shared((BM, BK), "float16")      # smem.dyn
        a_f = T.alloc_fragment((BM, BK), "float16")    # local.fragment
        T.copy(A[by * BM, 0], a_s)        # global  -> shared
        T.copy(a_s, a_f)                  # shared  -> fragment
        # T.annotate_layout({"a_s": ...}) -- pin the smem layout by the buffer's NAME
        T.copy(a_f, a_s)                  # fragment -> shared
        T.copy(a_s, B[by * BM, 0])        # shared  -> global
```

`T.copy` takes `coalesced_width`, `disable_tma`, `eviction_policy`, `prefer_instruction`,
`loop_layout` and `annotations` — see Sources for the exact 0.1.15 signature. The four common
scope pairs are all legal: global↔shared, shared↔fragment, global↔fragment (but read Gotchas),
and fragment↔shared.

## What TileLang decides for you here (and what it does not)

**Decided by the pass:**

- how the copy is mapped onto threads (that is what "operates at the block level" means);
- the instruction it picks for a given pair of scopes and shapes (vectorized load, `cp.async`,
  bulk copy, TMA) — `prefer_instruction` and `disable_tma` bias that choice, they do not command it;
- the default smem layout, and the index map for every buffer `T.annotate_layout` did not name.

**Decided by you:**

- the source and destination buffers and their shapes (they have to be compatible, or the call
  refuses);
- which buffers you name in `T.annotate_layout`;
- `eviction_policy` (`evict_normal` / `evict_first` / `evict_last`) when the L2 residency of
  the tile is the thing being priced.

## Gotchas

- **`T.annotate_layout` is a dict from buffer to layout, and it only touches the buffers whose
  names are in it.** Annotating one buffer and expecting its neighbours to follow is the most
  common misreading.
- **Annotation is a request, and index inference still runs.** If the emitted layout differs
  from the one you asked for, that is a reading to take, not a bug to report — prove the
  emitted layout with a probe before rewriting the copy.
- **A `T.copy` that names the same buffer on both sides is a no-op**, and it lowers silently
  (`T.copy(x[...], x)`). If a staging buffer was named after a kernel parameter, check the
  names: this is the failure the starter kernel's `_tile_in` / `_tile_out` naming exists to
  avoid.
- **The `fragment` scope is the one a `T.reduce_*` reads**, and a CPU-target lowering refuses a
  reduce whose source is `local.fragment` ("CPU reduce only supports local src and local/local.var
  dst buffers"). That is a **lowering-target** limit, not a rule about the GPU kernel.
- **Global ↔ fragment directly is legal but usually the wrong shape**: it skips the smem stage,
  so a swizzle you annotated on a shared buffer never applies.

## Sources

1. TileLang 0.1.15, installed package: `tilelang/language/copy.py` (`T.copy` signature and
   parameters), `tilelang/language/allocate.py` (`T.alloc_shared` / `T.alloc_fragment`),
   `tilelang/language/annotations.py` (`T.annotate_layout`).
2. TileLang docs: `docs/` in the `tile-ai/tilelang` repository; <https://tilelang.com>.
3. NVIDIA PTX ISA, for the instruction a given scope pair lowers to:
   <https://docs.nvidia.com/cuda/parallel-thread-execution/>.
4. This repo's probe ladder: `hw_kb/tilelang/sm90/gpu_probe_ladder.md`.
