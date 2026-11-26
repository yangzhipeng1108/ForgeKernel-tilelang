"""Deploy gates: the three things a fast raw kernel must pass to be deployable in a server.

Each gate answers one question, and returns True / False / None (unmeasured):

* **cuda_graph** -- can the kernel be captured in a ``torch.cuda.CUDAGraph`` and replayed? A
  server's decode path runs under CUDA graph capture; a kernel whose launch synchronises the
  host or invalidates the capture (the TileLang JIT case the porting docs record:
  ``cudaErrorStreamCaptureInvalidated``) silently falls back to eager and loses the win.
* **paged_kv** -- does the kernel consume the server's paged KV layout (block_table / page_size)
  directly, rather than needing a gather that eats the memory saving?
* **parity** -- is the kernel's output numerically the same as the server's native kernel
  (cosine >= ``PARITY_MIN_COS``)?

The GPU work (capture, run, compare) hides behind injected callables so the gate *logic* -- what
counts as pass/fail/None, the parity threshold, the block-table shape check -- is unit-testable
with no GPU. torch is imported lazily, only inside the default callables.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..e2e_record import GATE_GRAPH, GATE_PAGED, GATE_PARITY, PARITY_MIN_COS


@dataclass
class GateResult:
    name: str
    passed: bool | None          # True / False / None (unmeasured)
    detail: str = ""

    def as_dict(self) -> dict:
        return {self.name: self.passed}


# ---------------------------------------------------------------- cuda graph

def graph_capturable(capture_fn) -> GateResult:
    """Run ``capture_fn()`` which performs a warmup + graph capture + replay; pass iff no raise.

    ``capture_fn`` is the real GPU work (capture into a ``CUDAGraph`` and replay once). It raises
    on any failure -- a kernel that cannot be captured raises there and the gate fails with the
    reason. This function is pure: it only maps "raised or not" to the verdict, so it is testable
    with a fake capture_fn."""
    try:
        capture_fn()
    except Exception as e:  # noqa: BLE001 - any capture failure means "not capturable"
        return GateResult(GATE_GRAPH, False, f"graph capture failed: {type(e).__name__}: {e}")
    return GateResult(GATE_GRAPH, True, "captured and replayed")


def make_graph_capture(fn, args, *, torch=None):
    """The default GPU capture for ``fn(*args)``: warmup once, then capture + replay.

    torch is imported lazily. Warmup runs *outside* the capture so any one-time JIT/compile does
    not land inside the graph (a compile inside a capture is exactly what invalidates it)."""
    if torch is None:
        import torch as _torch
        torch = _torch

    def capture():
        fn(*args)                       # warmup: compile/alloc outside the capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(*args)
        g.replay()
        torch.cuda.synchronize()

    return capture


# ---------------------------------------------------------------- paged kv

def paged_kv_ok(kernel_inputs) -> GateResult:
    """Pass iff the kernel was fed the server's paged layout, not a gathered dense copy.

    The check is structural: the inputs bundle must carry a ``block_table`` (a 2-D int index of
    KV pages) and a positive ``page_size``. A dense KV tensor with no block table is the signal
    that the caller gathered the pages itself -- which means the kernel does not speak the
    server's paging and the 256x KV saving (MLA) is thrown away before the kernel runs.

    ``kernel_inputs`` is any object exposing ``block_table`` and ``page_size`` (a dict or a
    dataclass); a missing field is a fail, not a crash."""
    def get(name):
        if isinstance(kernel_inputs, dict):
            return kernel_inputs.get(name)
        return getattr(kernel_inputs, name, None)

    bt = get("block_table")
    page = get("page_size")
    if bt is None:
        return GateResult(GATE_PAGED, False, "no block_table: kernel got dense KV, not paged")
    if not page or int(page) <= 0:
        return GateResult(GATE_PAGED, False, f"page_size must be positive, got {page!r}")
    if getattr(bt, "dim", lambda: None)() not in (2, None):
        return GateResult(GATE_PAGED, False, "block_table is not 2-D")
    return GateResult(GATE_PAGED, True, f"paged: block_table 2-D, page_size={int(page)}")


# ---------------------------------------------------------------- parity

def parity_ok(a, b, *, min_cos: float = PARITY_MIN_COS, cosine_fn=None) -> GateResult:
    """Pass iff kernel output ``a`` matches server output ``b`` by cosine >= ``min_cos``.

    ``cosine_fn`` is the real cosine (imports torch lazily by default); injectable so the
    threshold logic is testable without tensors. A flat return of None from cosine_fn (e.g. a
    zero vector) fails the gate -- a degenerate comparison is not a pass."""
    if cosine_fn is None:
        cosine_fn = _torch_cos
    cos = cosine_fn(a, b)
    if cos is None:
        return GateResult(GATE_PARITY, False, "cosine undefined (degenerate input)")
    passed = cos >= min_cos
    return GateResult(GATE_PARITY, passed,
                      f"cos={cos:.6f} {'>=' if passed else '<'} {min_cos}")


def _torch_cos(a, b) -> float | None:
    import torch
    av = a.flatten().float()
    bv = b.flatten().float()
    denom = av.norm() * bv.norm()
    if float(denom) == 0.0:
        return None
    return float(torch.dot(av, bv) / denom)


# ---------------------------------------------------------------- bundle

def run_all_checks(kernel_inputs, *, capture_fn=None, kernel_out=None,
                   server_out=None, cosine_fn=None) -> dict[str, bool | None]:
    """Run the three gates and return ``{gate: True/False/None}``.

    A gate whose inputs are absent is None (unmeasured) -- never a fabricated pass. This is what
    the harness threads into :class:`ServingReading.gates`."""
    out: dict[str, bool | None] = {}
    out[GATE_GRAPH] = graph_capturable(capture_fn).passed if capture_fn is not None else None
    out[GATE_PAGED] = paged_kv_ok(kernel_inputs).passed if kernel_inputs is not None else None
    out[GATE_PARITY] = (parity_ok(kernel_out, server_out, cosine_fn=cosine_fn).passed
                        if (kernel_out is not None and server_out is not None) else None)
    return out
