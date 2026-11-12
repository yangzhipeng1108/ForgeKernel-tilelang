# The Qwen3.8 sglang config is real: every flag parses and the flashinfer GDN kernel binds on sm120

## Purpose

The deployment config for `Qwen3.8-Flash-Next-NVFP4` carries a long tail of `--mamba-*` and
`--linear-attn-*` flags, and the measured `730 tokens/s` depends on them doing something. This
entry records the check that they are not decoration: that every flag is a real argparse option,
that the values are legal, and that the kernel they select actually binds on this part rather
than silently falling back to Triton.

This matters because a fallback is logged as a warning and nothing else, so a config that looks
tuned can be running the default path and report the default number.

## Measured on

RTX 6000D (sm_120a, capability `(12, 0)`), container `lmsysorg/sglang:dev`, sglang
`0.0.0.dev1+g38d9e4a3f` installed from source at `/sgl-workspace/sglang`.

## Every flag is a real option

Server arguments are no longer defined in `srt/server_args.py` (908 lines, no `mamba` string in
it). They are registered modularly under `srt/arg_groups/`. Building the parser the way the CLI
does and enumerating `option_strings` gives **569 options**, of which the config's tail is all
present:

| flag | present | default | legal choices |
|---|---|---|---|
| `--mamba-track-interval` | yes | 256 | unrestricted |
| `--mamba-ssm-dtype` | yes | None | float32 / bfloat16 / float16 |
| `--mamba-radix-cache-strategy` | yes | auto | auto / no_buffer / extra_buffer / extra_buffer_lazy |
| `--max-mamba-cache-size` | yes | — | — |
| `--linear-attn-prefill-backend` | yes | None | triton / cutedsl / flashinfer / flashkda / nvidia_kda / ptx_kda / helion / intel_xpu |
| `--linear-attn-decode-backend` | yes | None | same |
| `--linear-attn-verify-backend` | yes | None | same + nv_cutedsl |
| `--ple-offload-embedding` | yes | — | — |

Every value in the shipped config is in its `choices` list, so nothing is rejected.

**A wrong flag would abort, not be ignored:** the CLI does not use `parse_known_args`, so a typo
in this tail would have failed the launch outright rather than being dropped. The config boots,
therefore the tail is being applied.

## The flashinfer kernel binds on sm120

The config asks for `--linear-attn-prefill-backend flashinfer` and
`--linear-attn-decode-backend flashinfer`. Several of the other backends are unavailable here —
`kda_backend.py` gates cutedsl, PTX-KDA and NVIDIA-KDA prefill behind **SM100** and falls back to
Triton otherwise, and this part is sm120. So the question is whether flashinfer survives its own
gate.

It does, checked by calling the gate rather than reading it:

```
cap: (12, 0)
flashinfer gdn prefill available: True
raw available: True prefill_fn: <function chunk_gated_delta_rule at ...>
```

`is_flashinfer_gdn_prefill_available()` returns **True** and the bound kernel is
`chunk_gated_delta_rule`. This is a real binding, not a fallback.

Two sm120-specific branches exist in the same file and are worth knowing about:

- `sm_major >= 12` sets `_prefill_needs_fp32_state = True`, so the prefill state is fp32 on this
  part regardless of `--mamba-ssm-dtype`. The `bfloat16` in the config governs the cache, not
  the prefill scratch.
- `supports_target_verify = sm_major in (9, 10)` — **sm120 is not in that set**, so the flashinfer
  path cannot serve MTP verify here. The config correctly routes
  `--linear-attn-verify-backend triton`, which is the only consistent choice; pointing verify at
  flashinfer would hit a path the file says this part does not support.

## `--mamba-track-interval` is consumed as an LCM modulus

Parsing proves a flag exists; it does not prove the value reaches computation. For
`--mamba-track-interval` the value is read in `runtime_context.py` through a two-level LCM:

```python
def mamba_cache_chunk_size() -> int:
    # max(model mamba_chunk_size or FLA_CHUNK_SIZE, page_size)
    ...

def mamba_checkpoint_grid(tree_page: int) -> int:
    return math.lcm(mamba_cache_chunk_size(), tree_page)

def mamba_track_grid(tree_page: int) -> int:
    return math.lcm(
        mamba_checkpoint_grid(tree_page), get_exec().mamba.mamba_track_interval
    )
```

`mamba_track_grid` is called on the speculative path (`spec_utils.py`,
`eagle_worker_common.py`, `dspark_worker_v2.py`, `eagle_post_verify_graphs.py`) with
`batch.tree_cache.page_size`, so the configured interval is a real operand, not a stored-looking
option. `mamba_track_grid` returns the depth granularity a decode-donated checkpoint must land on
for the radix tree to name it; the interval sets how finely those checkpoints are spaced.

### The concrete numbers for this model and config

The model's `config.json` has no `mamba_chunk_size`, so the chunk size falls back to
`FLA_CHUNK_SIZE`, which resolves to **64** (read out of
`sglang.kernels.ops.attention.fla.chunk_delta_h.CHUNK_SIZE`). With `--page-size 32`:

- `mamba_cache_chunk_size = max(64, 32) = 64`
- `mamba_checkpoint_grid(32) = lcm(64, 32) = 64`
- `mamba_track_grid(32) = lcm(64, interval)`

| `--mamba-track-interval` | `mamba_track_grid(32)` | vs default 256 |
|---|---|---|
| 64 | 64 | 1/4 |
| 128 (shipped) | 128 | 1/2 |
| 256 (default) | 256 | 1 |
| 512 | 512 | 2x |

So the config's `128` halves the checkpoint spacing relative to the default `256`: it donates
checkpoints twice as often, which keeps a finer history for MTP look-back at a higher per-step
cost. The assertion in `mamba_cache_chunk_size` (`max % min == 0`) is satisfied because both 64
and 32 divide evenly; a page size that does not divide 64 would fault at startup.

**Not yet measured: whether 128 is faster than 256 here.** The mechanism above is confirmed
from source and from a live import, but no sweep of `interval ∈ {64, 128, 256, 512}` against
`tokens/s` has been run. Until it is, treat `128` as "a live knob with a known halving effect on
checkpoint spacing", not as an established optimum.

## What this changes

Nothing about the numbers — the earlier `PG32 = 730` stands, and this is the evidence that it was
measured on the path it claims. What it removes is the doubt: the tuning tail is applied, the
backend is the requested one, and the verify routing is deliberate rather than accidental.
The `--mamba-track-interval 128` additionally has a known consumption path (the LCM above) rather
than being assumed live, but its optimality is still open.

## Sources

- `sglang/srt/arg_groups/` — modular server-argument registration (569 options)
- `sglang/srt/arg_groups/mamba_hook.py` — mamba/linear-attention argument resolution
- `sglang/srt/arg_groups/overrides.py:mamba_cache_chunk_size` — `max(FLA_CHUNK_SIZE, page_size)`
- `sglang/srt/runtime_context.py:mamba_checkpoint_grid,mamba_track_grid` — the two-level LCM that
  consumes `mamba_track_interval`
- `sglang/kernels/ops/attention/fla/chunk_delta_h.py:CHUNK_SIZE` — the `64` used when the model
  config has no `mamba_chunk_size`
- `sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` — the availability gate and the
  `chunk_gated_delta_rule` prefill kernel
- `sglang/srt/layers/attention/linear/kda_backend.py` — the SM100-only gates that rule out the
  KDA backends on this part
