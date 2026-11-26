"""Deployment-side package: drive a served model (sglang) and decide "optimal in deployment".

Deliberately outside the optimisation loop. The loop's contract (``loop/goal_ref``) forbids a
*student kernel* from calling sgl_kernel / flashinfer / flash_mla and requires the student to
write its own kernel; this package is the *other* side -- proving a written kernel is deployable
end to end (bench_serving tok/s, CUDA-graph capture, paged-KV fit, parity) when swapped into a
real server. Keeping it in its own namespace is what lets it reference serving infrastructure
without weakening the loop's "do not call the library" rule.

Nothing here imports torch or a server at module load: the GPU/server work hides behind
injected callables, so the assembly and parsing logic is unit-testable with no GPU.
"""

from __future__ import annotations

from .e2e_record import (
    GATE_GRAPH,
    GATE_PAGED,
    GATE_PARITY,
    GATES,
    PARITY_MIN_COS,
    ServingReading,
    compare_to_baseline,
    serving_reading_from_bench,
)
from . import runtime

__all__ = [
    "GATE_GRAPH",
    "GATE_PAGED",
    "GATE_PARITY",
    "GATES",
    "PARITY_MIN_COS",
    "ServingReading",
    "serving_reading_from_bench",
    "compare_to_baseline",
    "runtime",
]
