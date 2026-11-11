# Split-K on sm120 is correct only while M fits in one tile

Split-K is the usual fix for a starved grid: when K is long but M×N yields fewer CTAs than the
156 SMs, slicing K raises occupancy. On this kernel it **speeds the starved case up ~1.9x — and
silently returns wrong numbers the moment M needs more than one 128-row tile.** That asymmetry
is the whole entry, because the wrong case *looks* correct: it runs fast, `max|C|` matches the
reference, and only a cosine check against ground truth exposes it.

## The result

`DenseGemmKernel(split_k_slices=…, split_k_atomic_bf16=True)`, NVFP4, C = bf16, K = 4096
(32 k-tiles), tile (128, 128):

| M | N | M-tiles | split_k=1 | split_k=2 | split_k=4 | split_k=8 |
|---|---|---|---|---|---|---|
| 128 | 128 | 1 | cos 1.0000, 22.5 µs | **1.0000**, 12.9 µs | **1.0000**, 12.2 µs | **1.0000**, 12.6 µs |
| 256 | 128 | 2 | cos 1.0000 | **0.7076** ✗ | **0.7076** ✗ | **0.7076** ✗ |
| 128 | 256 | 1 | cos 1.0000 | **1.0000** | **1.0000** | **1.0000** |
| 256 | 256 | 2 | cos 1.0000 | **0.7087** ✗ | **0.7087** ✗ | **0.7087** ✗ |
| 512 | 512 | 4 | cos 1.0000 | 0.7058 ✗ | **0.4986** ✗ | **0.4986** ✗ |

The rule is exact: **`split_k_slices > 1` is correct iff `num_M_tiles == 1`, i.e.
`M <= 128`.** N may be any size (the 128×256 row is correct for every split). When M > 128 the
cosine collapses to `1/sqrt(num_M_tiles)` — the signature of computing only the first M-tile
row — and the "TF/s" the probe prints for those rows is fiction, because it only did
`1/num_M_tiles` of the work.

## Why: the grid drops the M dimension

`DenseGemmKernel._compute_grid` builds the split-K grid as

```python
grid = (1, split_k_slices, num_ctas_mnl[1])
```

`num_ctas_mnl` is `(M_tiles, N_tiles, 1)`. The first tuple element is a literal **1**, and
`block_idx[1]` is used inside the kernel as the *split* index (to pick which K-slice to run),
`block_idx[2]` as the N-tile. There is nowhere for an M-tile index to live, so only M-tile 0 is
ever launched. That is why N is free to grow (it rides in `block_idx[2]`) and M is not.

Compare the non-split path, which uses the full persistent scheduler:

```python
grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params,
                                                          max_active_clusters)
```

and covers every (M_tile, N_tile). Split-K trades that away for the K-parallel grid.

**Practical rule: enable split-K only when `M <= mma_tiler_mn[0]` (128).** For a decode-class
GEMM (M = batch × 1 token, so M ≤ 32) that is always true and split-K is a real win. For a
prefill or a batched GEMM (M in the thousands) split-K will corrupt the output, so leave it at 1.

## Two more things that made an earlier attempt wrong

- **C must be bf16 and zeroed.** The reduction is `scatter_add_bf16`, hardcoded, regardless of
  the C element type passed to the kernel. Passing an fp16 C and reading it as fp16 gives
  `max|C| = 4.2` against a 729.5 reference — the atomics wrote bf16 bit patterns into an fp16
  buffer. And because the atomics accumulate, C must be freshly zeroed before every launch;
  reusing a dirty C across timing iterations sums them.
- **The k-split must divide the k-tile count.** `k_tiles_per_split = k_tile_cnt // split_k` is an
  integer divide. K = 4096 is 32 k-tiles of 128, so split_k ∈ {1,2,4,8,16} divides evenly and
  loses nothing — but e.g. split_k = 3 or any split that does not divide 32 drops the tail
  k-tiles with no error.

## A timing trap worth repeating

Timing a `@cute.jit` host function with a loop that calls it re-runs the **compiler** each
iteration, not the kernel: an early version reported 162 ms per launch for a job that takes
22 µs. Compile once with `cute.compile(...)`, then loop the returned callable. And for the
`split_k = 1` baseline, pass the same `max_active_clusters` a real launch would use — with
`max_active_clusters = 1` the persistent scheduler serializes the whole GEMM onto one CTA and
the baseline is not comparable.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB, 156 SMs, no NVLink
- `lmsysorg/sglang:dev`: cutlass **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- Probe: `_probes/probe_sm120_splitk.py`
