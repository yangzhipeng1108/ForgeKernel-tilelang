# SM12X PCIe collectives: the backend that wins depends on message size and on not trusting the default

> State: PARTIAL — the numbers and the rules come from an 8×RTX 6000D (SM120, PCIe-only, no NVLink)
> deployment measured with vLLM and a vendor collective library, not from this repo's own probe;
> read them as a selection guide to confirm on the box, not as a certified timing to quote as-is.

> Applies to the whole sm12X family (SM120 workstation Blackwell: RTX PRO 6000/6000D, 5090). It is a
> **serving / communication** fact, not a TileLang kernel recipe: it tells you which collective
> backend an SM12X PCIe box should run, and how to decide, before you blame the kernel.

## Purpose

On a PCIe-only, no-NVLink box every all-reduce crosses the PCIe fabric, and three facts decide the
outcome: (1) the **default backend is not the best backend**; (2) the **message size** picks the
winner; (3) a single env var removes most of the small-message latency. A reader who tunes the
kernel while the collective path is the bottleneck is optimising the wrong layer.

## Finding 1: the default backend is not the best backend (Cake → NCCL, ~6.6–8×)

A recent inference framework shipped a MoE/communication backend (Cake) that is excellent on a
box with fast NICs but degrades badly on a pure-PCIe box:

| backend | prefill tok/s/GPU | decode steps/s |
|---|---|---|
| default (Cake) | 265.6 | 0.658 |
| NCCL | 2139.3 | 4.355 |

That is **8.1× prefill / 6.6× decode** from changing only the backend. Its own profile blames
**per-transfer round-trip latency**, not bandwidth or topology: a 4 KB round trip is ~9.5 µs, the
backend splits work into ~128 blocks × 7 peers ≈ 900 round trips ≈ 34 ms per layer. The fabric
itself was healthy (16 MB at ~52 GB/s across NUMA domains).

**Rule:** on a PCIe-only SM12X box, check what the collective backend actually does before tuning
anything else. Plain NCCL frequently beats a fancy default whose design assumed RDMA/NVLink.

## Finding 2: message size picks the collective backend

Measured speedup over NCCL, TP8, 8×RTX 6000D:

| message size | NCCL | small-message IPC path | FP8-wire collective |
|---|---|---|---|
| 4 MiB | 1.0× | **1.59× (fastest)** | ~0.15× |
| 8–16 MiB | 1.0× | ~1.35× / ~1.08× | ~0.12× / ~0.4× |
| 64 MiB | 1.0× | ~0.92× | ~1.38× |
| 128–256 MiB | 1.0× | ~0.9× | **~1.75× (peak 1.76×)** |
| 512 MiB | 1.0× | ~0.88× | ~1.0× (falls back to NCCL) |

The pattern: small messages are **latency-bound** (a one-sync push to peers wins), large messages are
**bandwidth-bound** (an FP8 wire format that cuts line bytes 48.4 % — 132 bytes / 128 values vs
BF16's 256 — wins), and at the extremes NCCL is the safe default.

**The SM12X-specific caveat:** the small-message IPC path uses multicast, and **SM12X has no working
multicast** (see `clusters_free_multicast_broken.md`). So on an RTX 6000D the realistic choice for
most message sizes collapses to **NCCL vs the FP8-wire collective** — the IPC column is unavailable,
not merely slower. Do not copy a config that relies on it.

## Finding 3: `NCCL_P2P_LEVEL=SYS` removes small-message latency (+18 % single-stream)

On 8×RTX 6000D a single-stream decode rose 158 → 187 tok/s from one environment variable:

```bash
export NCCL_P2P_LEVEL=SYS     # +18% single stream on a no-NVLink PCIe box
```

PCIe small messages pay per-hop latency; raising the P2P level lets NCCL use the system path that
avoids the slow hop. This is the single cheapest SM12X decode lever and costs nothing to try.

## Finding 4: the two-tier probe that tells latency-bound from bandwidth-bound

Before choosing a backend, run a 4 KB / 16 MB P2P micro-benchmark (two message sizes, one peer pair,
one cross-NUMA pair):

- **4 KB round trip** (expect ~9.5 µs) is the **per-transfer latency**; if it dominates your step
  time, the problem is round-trip count × latency, not bandwidth — reduce trips or fix the backend.
- **16 MB** (expect ~50 GB/s on PCIe) is the **bandwidth**; if the step is at this wall, only fewer
  bytes (FP8 wire) or more links help.

The 6000D box showed both healthy (~9.5 µs, ~52 GB/s) while a default backend still lost 8× — the
fabric was fine and the **backend's transfer count** was the bug. Probe the fabric before blaming it.

## Minimal reproduction (the probe, to run on the target box)

```bash
# 4 KB round trip (latency) and 16 MB (bandwidth), one peer pair
python - <<'PY'
import torch, time
dev = "cuda"
def rt(bytes_, iters=200):
    x = torch.empty(bytes_, dtype=torch.uint8, device=dev)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        y = x.clone()              # a device-to-device move stands in for one hop
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6   # microseconds
print("4KB  us/hop:", rt(4 * 1024))
print("16MB GB/s  :", 16 * 1024 * 1024 / (rt(16 * 1024 * 1024, 50) * 1e-6) / 1e9)
PY
```

(A true cross-peer round trip needs the real collective library; this device-local form only measures
the per-hop cost the scheduler pays, enough to tell latency-bound from bandwidth-bound.)

## When to use / not use

- **Use:** deciding the SM12X collective backend for prefill (large message → FP8 wire) vs decode
  (small message → NCCL + `NCCL_P2P_LEVEL=SYS`); diagnosing "the kernel is fine but the step is slow".
- **Not:** do not expect the small-message multicast IPC path on SM12X — it is unavailable (no working
  multicast). Do not treat the Cake→NCCL ratio as a kernel number; it is a backend-selection number.

## Sources

1. 8×RTX 6000D (SM120) deployment measurements reported for vLLM (PCIe-only, no NVLink): backend
   selection, message-size sweep, P2P micro-benchmark, `NCCL_P2P_LEVEL=SYS`.
2. `clusters_free_multicast_broken.md` (this tree) — why the multicast IPC path is absent on SM12X.
3. `sglang_config_flags_are_real.md` (this tree) — the same principle that a config flag must be
   confirmed to bind, not assumed.
