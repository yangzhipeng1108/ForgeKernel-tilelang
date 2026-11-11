# The TMA path end to end: fp16 GEMM through `make_tiled_tma_atom`

`tma_and_copy_family.md` established that every TMA symbol *constructs* on sm120. This closes
the loop: the shipped GeForce example drives a full GEMM through TMA and is **numerically
correct at every size tested**, so TMA on this card is not merely present, it works.

## The result

`blackwell_geforce/dense_gemm.py`, fp16 in / fp16 out / fp32 accumulate, tile (128, 128, 64),
20 timed iterations:

| M = N = K | time / iteration | achieved | verdict |
|---|---|---|---|
| 1024 | 25.4 µs | 84.5 TFLOP/s | **PASS** |
| 2048 | 82.5 µs | 208.0 TFLOP/s | **PASS** |
| 4096 | 962 µs | 142.8 TFLOP/s | **PASS** |

The 2048 point is the peak of the three; 4096 is slower per-FLOP because the 128×128 tile gives
fewer, longer-running CTAs than the grid wants. These are not tuned numbers — one warmup, one
tile shape — but they establish that TMA ingestion sustains a real fraction of the card, and
that the numbers match a reference at every size.

## What "through TMA" actually means here

The example is not using `cp.async`; it uses the full tensor-memory-accelerator path. The
mechanism, in the order it appears:

1. **Build the descriptor** — `cpasync.make_tiled_tma_atom(...)` in
   `_make_tma_atoms_and_tensors`, once for A and once for B. (There is still no
   `cute.make_tma_copy`; that name does not exist in this cutlass.)
2. **Prefetch it** — `cpasync.prefetch_descriptor(tma_atom_a)` on entry.
3. **Partition across threads** — `cpasync.tma_partition(tma_atom_a, …)` yields the
   `(tensor, tensor)` pair `tAsA, tAgA` that maps the smem tile to the gmem window.
4. **Issue the copy** — `cute.copy(tma_atom_a, tAgA, tAsA)` inside the mainloop, hidden behind
   a multi-stage pipeline.
5. **Store** — the epilogue builds a third TMA atom for C and partitions it the same way
   (`tma_partition` on `sC`/`gC`).

So a hand-written sm120 GEMM should follow the same five steps. The one difference from sm100
worth flagging: the smem budget is **101376 B**, not 232448, so the stage count the descriptor
partitioning is pipelined over must be chosen for 99 KB (see `tma_and_copy_family.md`).

## Why this matters for a tiled epilogue

The C-store being a TMA atom rather than a plain `cute.copy` from registers means the epilogue
writes smem and lets TMA drain it — which is why the C tile shape (`epi_tile`) is a separate
parameter from the MMA tile. On a 99 KB part, sizing the epi stage is part of the smem budget,
not free.

## Measured on

- RTX 6000D, compute capability **(12, 0)** = sm_120a, 85651 MiB, 156 SMs, no NVLink
- `lmsysorg/sglang:dev`: cutlass **4.8.0**, torch 2.14.1+cu130, CUDA 13.0.3
- `dense_gemm.py` from `flashinfer/data/cutlass/examples/python/CuTeDSL/blackwell_geforce/`
- Reproduce: `python3 dense_gemm.py --mnkl N,N,N,1 --tile_shape_mnk 128,128,64
  --a_dtype Float16 --b_dtype Float16 --c_dtype Float16 --acc_dtype Float32
  --a_major k --b_major k --c_major n --iterations 20`
