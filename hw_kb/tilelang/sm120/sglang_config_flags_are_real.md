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

## What this changes

Nothing about the numbers — the earlier `PG32 = 730` stands, and this is the evidence that it was
measured on the path it claims. What it removes is the doubt: the tuning tail is applied, the
backend is the requested one, and the verify routing is deliberate rather than accidental.

## Sources

- `sglang/srt/arg_groups/` 閳?modular server-argument registration (569 options)
- `sglang/srt/arg_groups/mamba_hook.py` 閳?mamba/linear-attention argument resolution
- `sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` 閳?the availability gate and the
  `chunk_gated_delta_rule` prefill kernel
- `sglang/srt/layers/attention/linear/kda_backend.py` 閳?the SM100-only gates that rule out the
  KDA backends on this part
