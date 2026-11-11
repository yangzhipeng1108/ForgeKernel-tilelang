# sm120 kernels — the four probes that won, as reusable modules

These are not new kernels. Four of the archived sm120 probes (`sk`, `ca2`, `g1`'s GQA half,
`mla5`) each earned one recipe in `hw_kb/tilelang/sm120/`. This directory is those four
probes turned into importable modules with the measured verdict baked into the API, so the
next person does not have to re-derive the tuning constants or repeat the correctness traps.

Every file was run on the actual RTX 6000D (tilelang 0.1.15, torch 2.15.0+cu132) and
reproduced its recipe's numbers with `cos = 1.00000` / `md = 0` against a torch reference.

| kernel | recipe | what it wins |
|---|---|---|
| `gemm/gemm_splitk/sm120` | `split_k_fills_a_starved_grid.md` | fp8 split-K when the grid is starved; `pick_splitk()` refuses to split a full grid |
| `attention/attn_fwd_causal/sm120` | `causal_attention_skips_the_masked_half.md` | bound the k-loop to the diagonal (~1.95x over full-square) |
| `attention/gqa_paged_decode/sm120` | `decode_is_the_one_bandwidth_bound_case.md` | fp16 KV decode at dram ~90 %; fp8 KV is a loss |
| `attention/mla_paged_decode/sm120` | `mla_absorbed_decode_beats_mha.md` | one shared latent instead of per-head KV; 256x smaller cache |

## Readings on this box

    gemm fp8   512^2x16384   SK8   182.8 TF   (plain 47.6 TF)     md=0
               1024^2x8192   SK2   241.4 TF   (plain 169.2 TF)    md=0
               4096^3        SK1   374.7 TF   (split-K loses)     md=0
    causal     2048/4096/8192      113.4 / 149.2 / 145.2 TF      cos=1.00000
    gqa decode 32h x 8kv, kv8192   147.3us  228 GB/s             cos=1.00000
    mla decode HQ128 kv8192 Dc128  123.3us   69.7 TF  2.1MB cache cos=1.00000

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

## Shape limits that are real, not incidental

- `gemm_splitk`: fp8 only; tile is fixed at 128x128x128 / stages 2.
- `attn_fwd_causal`: `SEQ` must be a multiple of `BM=64`; `BN=128` with `stages=2` exceeds
  dynamic smem, so stages is 1.
- `gqa_decode`: fp16 KV on purpose (fp8 KV upcasts in SMEM and is slower). `BN` must stay 64;
  128 asks for 107520 B and fails.
- `mla_decode`: absorbed form with `Dc=128`. The production `Dc=512` does **not** fit one
  kernel — the up-projection needs a second pass. `QLEN` must be 16.
