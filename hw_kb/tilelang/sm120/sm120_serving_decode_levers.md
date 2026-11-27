# SM12X serving decode: prefix cache (TTFT 38–54×) and CUDA Graph (steps/s 2.7×) are the two levers

> State: PARTIAL — the ratios come from an 8×RTX 6000D (SM120, PCIe-only) deployment measured with
> vLLM, not from this repo's own probe; treat them as the reason a serving reading must record these
> two run facts and a decode gate must demand graph capture, not as certified timings to quote.

> Applies to the whole sm12X family. A **serving** fact, not a TileLang kernel recipe: it says what a
> served-model reading is comparable across, and why the CUDA-graph deploy gate exists.

## Purpose

Two serving-side levers move a decode/TTFT reading by integer multiples on SM12X, far more than any
kernel micro-optimisation — and both change what a measurement *means*. A reading that ignores them
is not comparable to one that did. This entry records the numbers behind two guards the serve layer
enforces: **a reading must state its prefix-cache state and context length**, and **decode must be
CUDA-graph capturable**.

## Finding 1: prefix caching moves long-context TTFT by 38–54×

TTFT with and without prefix/KV-prefix caching, same box, 20 concurrent (8 at 512K):

| context | concurrency | TTFT no cache | TTFT with cache | speedup |
|---|---|---|---|---|
| ~32K | 20 | 27.98 s | 0.73 s | 38× |
| ~128K | 20 | 118.15 s | 2.26 s | 52× |
| ~512K | 8 | 239.36 s | 4.41 s | 54× |

The same ~10 s of per-request compute sits behind a cache-hit first token regardless of context
length; the cache turns an unusable long-context request into a usable one. **The consequence for
measurement:** a TTFT or tok/s taken with the cache on is not comparable to one taken with it off,
because they are different workloads. That is exactly why `serve.e2e_record.ServingReading` carries
`prefix_cache` and `context_len`, and `compare_to_baseline` refuses a ratio across a differing
prefix-cache state or a differing context length (the same discipline as refusing a cross-concurrency
ratio).

## Finding 2: CUDA Graph capture lifts decode ~2.7× — and is a deploy gate

On the same box, decode-only benchmark:

| config | decode steps/s |
|---|---|
| eager (`--enforce-eager`) | 10.97 |
| CUDA graphs (default) | 29.23 |

Disabling graph capture silently drops decode to roughly a third — the eager launch overhead is paid
every step. This is why the serve layer's `cuda_graph` gate (`serve.checks.graph_capturable`) exists:
a raw kernel that cannot be captured in a `torch.cuda.CUDAGraph` and replayed will fall back to eager
and hand back a third of the win, however fast its isolated launch measures. **Capture it or do not
call the comparison a win.**

## What this changes in the serve layer (the guards, not prose)

- `ServingReading.prefix_cache: bool | None`, `.context_len: int | None` — recorded on every reading;
  `context_len` defaults to the run's input length.
- `compare_to_baseline` returns a `reason` (no ratio) when the two readings differ in prefix-cache
  state or context length, mirroring the existing cross-concurrency refusal.
- The `cuda_graph` deploy gate (`graph_capturable`) is the measurement-side twin of Finding 2: an
  uncapturable kernel is not deployable regardless of its offline speed.

## When to use / not use

- **Use:** whenever you take or compare a served-model reading on SM12X — always record prefix-cache
  state + context length, and always run decode under CUDA graphs.
- **Not:** do not compare a cached reading to an uncached one and call the difference a kernel win;
  do not benchmark a decode kernel in eager and quote the number as its serving speed.

## Sources

1. 8×RTX 6000D (SM120) deployment measurements reported for vLLM: prefix-cache TTFT table and the
   eager-vs-CUDA-graph decode table.
2. `decode_is_the_one_bandwidth_bound_case.md` (this tree) — decode is the bandwidth-bound case, so
   launch overhead and graph capture dominate differently than for prefill.
