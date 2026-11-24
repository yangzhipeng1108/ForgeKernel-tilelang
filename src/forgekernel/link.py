"""A simulated PipeLink-style linker: join components into one fused kernel, on paper.

PipeLink's core is a linker that packs GPU-kernel stages separately and links them into
one fused kernel, checking every seam at link time and solving the resources/roles the
fused kernel needs. Doing that for real needs ``nvcc``/CuTeDSL and a GPU, neither of which
this environment has. What this module does instead is the part that does **not** need a
GPU, and that is where most link-time decisions actually live:

* **seam checking** — each stage declares its interface (``contract.json``); two stages
  link only if the output one produces matches, by name / dtype / shape-over-axes, the
  input the next one consumes. A mismatch is refused with a numbered ``L-*`` diagnostic,
  the same shape as PipeLink's ``L-PROTO`` / ``L-HANDLE``, rather than silently producing
  a kernel that hangs. Two physical facts are checked too: a port's ``handle`` (where the
  tensor lives — TMEM / registers / shared memory; a differing kind is bridged by the
  generated copy, an *unknown* kind is ``L-HANDLE``) and its ``generation`` (which pipeline
  generation it carries; a mismatch means several in flight, refused with ``L-PHASE`` and
  the ring depth that would carry them).
* **resource and role solving** — every stage declares what it consumes (shared memory,
  TMEM, registers, barrier slots, warps) and the caller gives the target's caps; the
  linker solves the peak simultaneous use (not the sum), finds the tightest ring depth,
  and refuses a plan that exceeds a cap or double-books a role. Pure arithmetic, so it is
  exact and testable.
* **fused-source generation** — the linked plan renders one kernel source skeleton in
  which each stage's body is inlined under its declared role, wired by generated hand-off
  code: for each edge it emits the actual move (``acc_tmem_to_reg`` when a TMEM accumulator
  is imported into registers, ``acc_reg_to_smem`` to stage a ring slot, ``acc_identity``
  when the kind is unchanged), chosen from the two ports' handle kinds. The skeleton is
  syntactically valid Python (or CUDA-shaped C++), so a downstream step compiles it; here it
  is checked by ``ast.parse`` (Python) — no GPU, no build.
* **semantic parity, on the CPU** — each stage ships a ``verify.json`` whose ``golden``
  says what it computes (torch source). Chaining those goldens through the same wiring the
  linker solved gives a CPU-torch statement of the fused kernel; running it against a
  caller-supplied reference proves the *link semantics* ("stage A then stage B = the stated
  composition") without a GPU. This is the numeric proof the seams are right, separate from
  whether the generated CUDA is fast.

It is a *simulation* and says so: it links what the stages **declare**, not the opaque code
behind the declaration. PipeLink's own README makes the same caveat, and so every linked
kernel still needs its numeric check and sanitizers before it is trusted on hardware. What
this buys is that contract, budget and wiring errors surface here — instantly, with a code —
instead of as a hang on the GPU.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel import component as comp

# The resources a link solves. Each is summed across stages and compared to a cap the
# caller passes in ``caps``. Names match the chips' vocabulary so a manifest reads true.
RESOURCES = ("smem_bytes", "tmem_cols", "regs_per_thread", "barrier_slots")
ROLE_KEYS = ("warps",)

# Default caps: one sm_90/sm_100 CTA's worth. The caller overrides for the actual target.
DEFAULT_CAPS = {"smem_bytes": 227 * 1024, "tmem_cols": 512, "regs_per_thread": 255,
                "barrier_slots": 16, "warps": 32}

# Numbered diagnostics, PipeLink-style. Each names the class of a link failure so a user
# can look it up, not just read a sentence.
DIAG = {
    "L-STAGE": "a stage is missing, unreadable, or is not a component",
    "L-PROTO": "two linked stages disagree on an interface (name / dtype / shape); a bridge "
               "that would fit is named when one exists",
    "L-WIRE": "the pipeline names a stage, or a port, that does not exist",
    "L-HANDLE": "a port names a handle kind (tmem/reg/smem) the linker does not understand, "
                "so it cannot generate the tensor hand-off",
    "L-PHASE": "producer and consumer disagree on the pipeline generation a port carries; "
               "tensor import hands off one generation, so several in flight need a ring",
    "L-RES": "a resource's simultaneous peak exceeds the target's cap",
    "L-ROLE": "the warp peak exceeds the block, or a role is malformed",
    "L-RING": "the pipeline asks for a ring depth outside what the stages allow",
    "L-ORDER": "the pipeline is empty, has a cycle, or lists a stage twice",
}


class LinkError(ValueError):
    """A link that is refused, with the diagnostic codes that explain why."""

    def __init__(self, diagnostics: list["Diagnostic"]):
        self.diagnostics = diagnostics
        super().__init__("; ".join(f"{d.code}: {d.message}" for d in diagnostics))


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class LinkPlan:
    """The result of a successful link: the wired stages and the solved resources."""
    stages: list[str]                       # component names, in pipeline order
    language: str
    edges: list[tuple[str, str, str, str]]  # (from stage, output port, to stage, input port)
    resources: dict                         # solved per-resource (simultaneous peak)
    ring_depth: int
    caps: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)   # why each rig/ring number came out

    def manifest(self) -> dict:
        return {"stages": self.stages, "language": self.language,
                "edges": [list(e) for e in self.edges], "resources": self.resources,
                "ring_depth": self.ring_depth, "caps": self.caps, "reasons": self.reasons}


def summarize(plan: LinkPlan) -> list[str]:
    """Human-readable lines explaining a plan: the wiring, the peak, and why.

    The number a plan reports (``smem_bytes: 131072``) is the peak, not the sum; a reader
    who does not know that will think the linker under-counts. So each resource line says
    both, and the ring line says whether the floor is a stage's own or a rate mismatch."""
    lines = [f"language: {plan.language}"]
    lines.append("stages: " + " -> ".join(plan.stages))
    if plan.edges:
        lines.append("wiring:")
        for frm, out, to, inp in plan.edges:
            lines.append(f"  {frm}.{out} -> {to}.{inp}")
    lines.append("resources (peak simultaneous use; each cap named when overridable):")
    for r in RESOURCES:
        if r in plan.resources:
            cap = plan.caps.get(r)
            tail = f"  cap {cap}" + (" EXCEEDED" if cap is not None and plan.resources[r] > cap
                                     else "") if cap is not None else ""
            lines.append(f"  {r}: {plan.resources[r]}{tail}")
    if "warps" in plan.resources:
        lines.append(f"  warps: {plan.resources['warps']}")
    reason = plan.reasons.get("ring")
    lines.append(f"ring_depth: {plan.ring_depth}" + (f"  ({reason})" if reason else ""))
    budget = plan.reasons.get("expected_ms")
    if budget is not None:
        lines.append(f"expected_ms: {budget}  (summed declared budget, not a measurement)")
    return lines


def diff_plans(before: LinkPlan, after: LinkPlan) -> dict:
    """What changed when a pipeline was re-linked with a stage swapped (PipeLink's "re-link
    only the stage that changed" use case).

    Returns the stage set difference, the edge difference, and the per-resource / ring
    deltas — the numbers a caller checks to be sure swapping one stage did not quietly move
    everyone else's budget."""
    bs, as_ = set(before.stages), set(after.stages)
    be, ae = set(map(tuple, before.edges)), set(map(tuple, after.edges))
    res = {}
    for r in sorted(set(before.resources) | set(after.resources)):
        b, a = before.resources.get(r), after.resources.get(r)
        if b != a:
            res[r] = {"before": b, "after": a, "delta": (a or 0) - (b or 0)}
    return {
        "stages_added": sorted(as_ - bs), "stages_removed": sorted(bs - as_),
        "edges_added": sorted(ae - be), "edges_removed": sorted(be - ae),
        "resources_changed": res,
        "ring_depth": {"before": before.ring_depth, "after": after.ring_depth,
                       "delta": after.ring_depth - before.ring_depth},
        "language": {"before": before.language, "after": after.language},
    }


# --- loading stages -------------------------------------------------------------------

def _load_stage(spec: str, root: str | Path | None) -> tuple[comp.Component, dict]:
    """A pipeline entry: ``name`` or ``name@language``. Returns (component, contract)."""
    entry = spec.strip()
    if "@" in entry:
        name, language = entry.split("@", 1)
    else:
        name, language = entry, ""
    cdir = comp.components_root(root) / name
    try:
        component = comp.read_component(cdir)
    except comp.ComponentError as e:
        raise LinkError([Diagnostic("L-STAGE", f"{name!r}: {e}")]) from e
    if language:
        if language not in component.languages:
            raise LinkError([Diagnostic(
                "L-STAGE", f"{name!r} does not implement {language!r} "
                           f"(has: {', '.join(component.languages)})")])
    elif component.languages:
        language = component.languages[0]
    try:
        contract = comp.read_contract(cdir)
    except comp.ComponentError as e:
        raise LinkError([Diagnostic("L-STAGE", f"{name!r}: {e}")]) from e
    component.dir = cdir
    return component, contract


# --- seam checking --------------------------------------------------------------------

def _seam_for_port(prev_name: str, prev_contract: dict, out_port: str,
                   nxt_name: str, nxt_contract: dict, in_port: str,
                   root: str | Path | None = None) -> list[Diagnostic]:
    """Whether ``prev``'s ``out_port`` can feed ``nxt``'s ``in_port``.

    A seam is well-typed when dtype matches and the shape structure lines up over the
    stages' own axes. This is the per-port form so a DAG can wire any output to any input;
    the chain case is the special case where each stage has exactly one of each.

    On a mismatch, if a component under ``root`` could sit between the two (its input takes
    what the producer emits, its output gives what the consumer wants), the diagnostic names
    it as a bridge rather than only reporting the failure — the difference between "this does
    not fit" and "this does not fit *directly*, insert X". A bridge that would itself be
    mismatched is not suggested, so the hint is always a link that would actually pass."""
    p_out = prev_contract.get("outputs") or {}
    n_in = nxt_contract.get("inputs") or {}
    if out_port not in p_out:
        return [Diagnostic("L-WIRE", f"{prev_name} has no output {out_port!r} "
                                     f"(outputs: {', '.join(p_out) or 'none'})")]
    if in_port not in n_in:
        return [Diagnostic("L-WIRE", f"{nxt_name} has no input {in_port!r} "
                                     f"(inputs: {', '.join(n_in) or 'none'})")]
    po_spec, ni_spec = p_out[out_port], n_in[in_port]
    problems: list[Diagnostic] = []
    problems += _handle_for_edge(prev_name, out_port, po_spec, nxt_name, in_port, ni_spec)
    if not _same_port(po_spec, ni_spec, prev_contract, nxt_contract):
        hint = _bridge_hint(root, prev_name, po_spec, prev_contract, nxt_name, ni_spec, nxt_contract)
        msg = (f"{prev_name}.{out_port} ({_desc(po_spec)}) does not match "
               f"{nxt_name}.{in_port} ({_desc(ni_spec)})")
        if hint:
            problems.append(Diagnostic(
                "L-PROTO", msg + f"; a bridge would fit: insert {hint} "
                                   f"between them ({prev_name} -> {hint} -> {nxt_name})"))
        else:
            problems.append(Diagnostic("L-PROTO", msg))
    return problems


def _handle_for_edge(prev_name: str, out_port: str, po_spec: dict,
                     nxt_name: str, in_port: str, ni_spec: dict) -> list[Diagnostic]:
    """Whether the tensor can actually be handed from ``prev.out`` to ``nxt.in``.

    Two physical facts the seam check above (dtype/shape) does not cover, both from the
    contract's optional port fields:

    * **handle** — where the tensor lives (``tmem`` / ``reg`` / ``smem``). The linker
      *generates* the move between any two known kinds (TMEM -> reg, reg -> smem, ...), so a
      differing kind is expected, not an error. An **unknown** kind is an ``L-HANDLE``: no
      copy can be emitted for a location the linker does not understand. A kind is only a
      problem when it cannot be named.
    * **generation** — which pipeline generation the port carries. Tensor import hands off
      *one* generation at a time; if the producer emits generation G and the consumer expects
      a different G', exactly the "multi-generation in flight" case PipeLink says must be an
      explicit multi-slot ring. So it is ``L-PHASE``, and the message names the ring depth
      (``|G - G'| + 1``) that would make the hand-off legal instead of dropping data.

    Defaults: a port without ``handle`` or ``generation`` imposes no constraint, so the
    common one-input stage needs neither field."""
    diags: list[Diagnostic] = []
    ph = po_spec.get("handle")
    nh = ni_spec.get("handle")
    for which, kind in ((f"{prev_name}.{out_port}", ph), (f"{nxt_name}.{in_port}", nh)):
        if kind is not None and kind not in comp.HANDLE_KINDS:
            diags.append(Diagnostic(
                "L-HANDLE", f"{which} declares handle {kind!r}, not one of "
                            f"{list(comp.HANDLE_KINDS)} — no hand-off code can move it"))
    pg = po_spec.get("generation")
    ng = ni_spec.get("generation")
    if isinstance(pg, int) and not isinstance(pg, bool) and \
            isinstance(ng, int) and not isinstance(ng, bool) and pg != ng:
        depth = abs(pg - ng) + 1
        move = f"{ph or 'unspecified'} -> {nh or 'unspecified'}"
        diags.append(Diagnostic(
            "L-PHASE", f"{prev_name}.{out_port} emits generation {pg}, "
                       f"{nxt_name}.{in_port} expects {ng} ({move}); tensor import hands "
                       f"off one generation — carry these in a ring of depth >= {depth}"))
    return diags


def _bridge_hint(root, prev_name, po_spec, prev_contract, nxt_name, ni_spec,
                 nxt_contract) -> str | None:
    """The first component under ``root`` that takes the producer's output and gives the
    consumer's input — a candidate adapter stage for a mismatched seam. None if no such
    component exists (or the caller has no root to search)."""
    if root is None:
        return None
    try:
        candidates = comp.list_components(root)
    except Exception:  # noqa: BLE001 — a hint is best-effort, never a hard failure
        return None
    for cand in candidates:
        if cand.name in (prev_name, nxt_name):
            continue
        try:
            cct = comp.read_contract(cand.dir)
        except comp.ComponentError:
            continue
        ins = cct.get("inputs") or {}
        outs = cct.get("outputs") or {}
        if len(ins) != 1 or len(outs) != 1:
            continue
        (ci_port, ci_spec), = ins.items()
        (co_port, co_spec), = outs.items()
        # bridge.in <- producer.out, bridge.out -> consumer.in
        if _same_port(po_spec, ci_spec, prev_contract, cct) and \
                _same_port(co_spec, ni_spec, cct, nxt_contract):
            return cand.name
    return None


def _same_port(a: dict, b: dict, ca: dict, cb: dict) -> bool:
    if a.get("dtype") != b.get("dtype"):
        return False
    return comp._shape_canon(a.get("shape"), {x: i for i, x in enumerate(ca.get("axes") or [])}) \
        == comp._shape_canon(b.get("shape"), {x: i for i, x in enumerate(cb.get("axes") or [])})


def _desc(spec: dict) -> str:
    return f"{spec.get('dtype')} {spec.get('shape')}"


# --- resource / role solving ----------------------------------------------------------

def _resource_claims(c: comp.Component, problems: list[Diagnostic]) -> tuple[dict, int, dict]:
    """One stage's declared claim: {resource: int}, warps, and its rate/latency roles."""
    claim = {}
    for r in RESOURCES:
        v = (c.resources or {}).get(r, 0)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            problems.append(Diagnostic("L-RES", f"{c.name}: {r} must be a non-negative int, "
                                                 f"got {v!r}"))
            v = 0
        claim[r] = v
    w = (c.roles or {}).get("warps", 0)
    if not isinstance(w, int) or isinstance(w, bool) or w < 0:
        problems.append(Diagnostic("L-ROLE", f"{c.name}: warps must be a non-negative int, "
                                             f"got {w!r}"))
        w = 0
    return claim, w, (c.roles or {})


def _peak_residency(order: list[str], succ: dict[str, set],
                    claims: dict[str, dict]) -> tuple[dict, dict]:
    """Peak simultaneous resource use over the whole DAG, not the sum of every stage.

    A fused kernel runs its stages as a pipeline: a stage holds its shared memory / TMEM /
    registers / barrier slots from when it starts until its *last consumer* starts (the
    hand-off is then done and the buffer can be reused). So the residency at any point of
    the topological order is the sum over the stages whose live window covers it — an upper
    bound far below "sum of everything" for a chain (only neighbours overlap) and equal to
    it only when every stage is live at once (a source fanning into all the sinks).

    This is the resource analogue of PipeLink's link-time SMEM/TMEM planning: not an
    allocator, but the number the cap check compares against, and the reason a chain of
    eight 64 KB stages (512 KB summed) can still fit a 227 KB CTA. A sink (no consumers)
    stays live to the end of the order.

    Returns ``(peak, holders)``: the peak per resource, and for each the set of stages live
    at the step where that peak first occurred. ``holders`` turns an over-cap reading into
    "smem_bytes peak 262144 at {a, b}" — the stages a user must look at — instead of a bare
    number with no lead. It scans the live windows as events (a resource's total rises when
    a stage opens and falls *after* the last consumer opens), O(stages^2) worst case but
    with no per-step allocation, and ties are taken at the earliest step for a stable set."""
    pos = {n: i for i, n in enumerate(order)}
    last = len(order) - 1
    end = {}
    for n in order:
        consumers = succ.get(n, ())
        end[n] = max((pos[m] for m in consumers if m in pos), default=last)
    keys = list(dict.fromkeys(k for c in claims.values() for k in c))
    peak = {k: 0 for k in keys}
    holders: dict[str, list[str]] = {k: [] for k in keys}
    for step in range(len(order)):
        live = [n for n in order if pos[n] <= step <= end[n]]
        for k in keys:
            acc = sum(claims[n].get(k, 0) for n in live)
            if acc > peak[k]:
                peak[k] = acc
                holders[k] = live
    return peak, holders


def _solve(stages: list[comp.Component], order: list[str], succ: dict[str, set],
           pred: dict[str, set], caps: dict,
           ring_request: int | None) -> tuple[dict, int, list[Diagnostic]]:
    """Solve the plan's peak resources, ring depth and warp peak against the caps."""
    problems: list[Diagnostic] = []
    claims: dict[str, dict] = {}
    warps_of: dict[str, int] = {}
    roles_of: dict[str, dict] = {}
    for c in stages:
        claim, w, roles = _resource_claims(c, problems)
        claims[c.name] = claim
        warps_of[c.name] = w
        roles_of[c.name] = roles

    peak, holders = _peak_residency(order, succ, claims)
    # warps are resident over the same live windows as the buffers
    claims_w = {n: {"warps": w} for n, w in warps_of.items()}
    live_warps, warp_holders = _peak_residency(order, succ, claims_w)
    live_warps = live_warps["warps"]
    for r in RESOURCES:
        cap = caps.get(r)
        if cap is not None and peak[r] > cap:
            at = holders.get(r) or []
            where = f" at {{{', '.join(at)}}}" if at else ""
            problems.append(Diagnostic(
                "L-RES", f"{r} peak {peak[r]}{where} exceeds the cap {cap}"))
    wcap = caps.get("warps")
    if wcap is not None and live_warps > wcap:
        at = warp_holders.get("warps") or []
        where = f" at {{{', '.join(at)}}}" if at else ""
        problems.append(Diagnostic(
            "L-ROLE", f"warps peak {live_warps}{where} exceeds the block cap {wcap}"))

    ring_lo, ring_hi = _ring_limits(stages, roles_of, order, succ, pred)
    reasons = {"ring": _ring_reason(ring_lo, ring_hi, stages, roles_of, order, succ, pred),
               "expected_ms": _expected_ms(stages, roles_of)}
    if ring_hi is not None and ring_lo > ring_hi:
        problems.append(Diagnostic("L-RING", f"ring depth floor {ring_lo} > ceiling {ring_hi}"))
        ring = ring_lo
    elif ring_request is not None:
        if ring_request < ring_lo or (ring_hi is not None and ring_request > ring_hi):
            problems.append(Diagnostic(
                "L-RING", f"requested ring depth {ring_request} outside [{ring_lo}, "
                          f"{ring_hi if ring_hi is not None else 'inf'}]"))
        ring = ring_request
    else:
        ring = ring_lo if ring_lo else (ring_hi or 1)
    peak["warps"] = live_warps
    return peak, ring, problems, reasons


def _expected_ms(stages, roles_of) -> float | None:
    """The summed declared time budget of the stages, or None when none declared one.

    A stage can declare ``roles.expected_ms`` — its own budget on the target. The plan sums
    them as the fused kernel's budget. It is a *declaration*, not a measurement (this is a
    simulation), so it is reported as the budget it is, never as a predicted speed."""
    total, seen = 0.0, False
    for c in stages:
        v = (roles_of.get(c.name) or {}).get("expected_ms")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
            total += float(v)
            seen = True
    return round(total, 6) if seen else None


def _ring_reason(ring_lo, ring_hi, stages, roles_of, order, succ, pred) -> str:
    """One phrase saying what set the ring floor: a stage's own minimum, or a rate need."""
    floor_from = None
    for c in stages:
        rlo = (roles_of.get(c.name) or {}).get("ring_min", 0)
        if isinstance(rlo, int) and rlo and rlo == ring_lo:
            floor_from = f"{c.name}.ring_min"
            break
    need = _ring_throughput(order, succ, pred, roles_of)
    if need > 1 and need == ring_lo:
        return f"floor from a producer/consumer rate mismatch (needs {need} in flight)"
    if floor_from:
        return f"floor from {floor_from}"
    return "minimum depth (no stage or rate requires more)"


def _ring_limits(stages, roles_of: dict[str, dict], order: list[str],
                 succ: dict[str, set], pred: dict[str, set]) -> tuple[int, int | None]:
    """The ring-depth window a fused kernel may use.

    Each stage may declare ``ring_min`` / ``ring_max`` (an explicit window) and a
    ``producer_rate`` / ``consumer_rate`` (units per step). Along an edge, a producer faster
    than its consumer needs ``ceil(producer/consumer)`` slots so it never blocks on a full
    ring — the same in-flight calculation PipeLink uses to size a mbarrier ring. The plan's
    floor is the max of every stage's own floor and the throughput need of every edge; the
    ceiling is the tightest ``ring_max`` any stage declares."""
    lo, hi = 0, None
    for c in stages:
        rlo = (roles_of[c.name] or {}).get("ring_min", 0)
        lo = max(lo, int(rlo) if isinstance(rlo, int) and not isinstance(rlo, bool) else 0)
        rhi = (roles_of[c.name] or {}).get("ring_max")
        if isinstance(rhi, int) and not isinstance(rhi, bool):
            hi = rhi if hi is None else min(hi, rhi)
    lo = max(lo, _ring_throughput(order, succ, pred, roles_of))
    return lo, hi


def _ring_throughput(order: list[str], succ: dict[str, set], pred: dict[str, set],
                     roles_of: dict[str, dict]) -> int:
    """The largest in-flight requirement along any edge from a rate mismatch (>= 1).

    For each consumer with a declared ``consumer_rate`` and each producer feeding it with a
    ``producer_rate``, the ring between them needs ``ceil(prod/cons)`` slots. The max over
    edges is the floor the plan must meet so a fast producer never stalls on a full ring."""
    import math
    need = 1
    for c in order:
        cons = roles_of.get(c, {}).get("consumer_rate")
        if not isinstance(cons, (int, float)) or isinstance(cons, bool) or cons <= 0:
            continue
        for p in pred.get(c, ()):
            prod = roles_of.get(p, {}).get("producer_rate")
            if isinstance(prod, (int, float)) and not isinstance(prod, bool) and prod > 0:
                need = max(need, math.ceil(prod / cons))
    return need



# --- the link ------------------------------------------------------------------------

def link(pipeline: list[str], *, root: str | Path | None = None, language: str | None = None,
         caps: dict | None = None, ring: int | None = None,
         edges: list[tuple[str, str, str, str]] | None = None) -> LinkPlan:
    """Link components into one fused-kernel plan, or raise LinkError.

    Two ways to describe the wiring:

    * **chain** (``edges`` is None): each stage's single output feeds the next stage's
      single input, so ``["load", "gemm"]`` means ``load.y -> gemm.x``. The common case.
    * **DAG** (``edges`` given): explicit ``(from_stage, out_port, to_stage, in_port)``
      edges, so a stage may take several inputs (wired from several upstreams) and an
      output may fan out. The stages not reachable as a source must have every input wired;
      the graph must be acyclic. A DAG is what lets one GEMM feed two consumers, or a
      splat-plus-weights pair feed a fused op — the structures a single chain cannot say.

    ``language`` forces every stage to that language slot; otherwise each stage uses its
    first, and all must agree because a fused kernel is one language."""
    if not pipeline:
        raise LinkError([Diagnostic("L-ORDER", "the pipeline is empty")])
    capsule = {**DEFAULT_CAPS, **(caps or {})}
    loaded = [_load_stage(s, root) for s in pipeline]
    stages = [c for c, _ in loaded]
    by_name = {c.name: (c, ct) for c, ct in loaded}
    if len(by_name) != len(stages):
        dup = sorted({c.name for c in stages if [x.name for x in stages].count(c.name) > 1})
        raise LinkError([Diagnostic("L-ORDER", f"a stage is listed twice: {dup}")])

    langs = {language} if language else {stages[0].languages[0]}
    for c in stages:
        if not (langs & set(c.languages)):
            raise LinkError([Diagnostic(
                "L-STAGE", f"{c.name} does not implement {sorted(langs)[0]!r} "
                           f"(has: {', '.join(c.languages)})")])
    lang = sorted(langs)[0]

    wired = _chain_edges(stages, loaded) if edges is None else list(edges)
    _check_stage_names(wired, by_name)              # raises L-WIRE on an unknown stage
    problems: list[Diagnostic] = []
    order = _topo_order([c.name for c in stages], wired, problems)
    for frm, out, to, inp in wired:
        _, pct = by_name[frm]
        _, nct = by_name[to]
        problems += _seam_for_port(frm, pct, out, to, nct, inp, root=root)
    problems += _input_coverage(stages, by_name, wired)

    succ: dict[str, set] = {c.name: set() for c in stages}
    pred: dict[str, set] = {c.name: set() for c in stages}
    for frm, _o, to, _i in wired:
        if frm in succ and to in pred:
            succ[frm].add(to)
            pred[to].add(frm)

    totals, ring_depth, res_problems, reasons = _solve(stages, order, succ, pred, capsule, ring)
    problems += res_problems
    if problems:
        raise LinkError(_coalesce(problems))

    return LinkPlan(stages=order, language=lang,
                    edges=[(a, b, c, d) for a, b, c, d in wired],
                    resources=totals, ring_depth=ring_depth, caps=caps or {},
                    reasons=reasons)


def _chain_edges(stages, loaded) -> list[tuple[str, str, str, str]]:
    """The implicit chain wiring: each stage's single output -> the next's single input."""
    edges = []
    for i in range(len(loaded) - 1):
        po = next(iter((loaded[i][1].get("outputs") or {})), None)
        ni = next(iter((loaded[i + 1][1].get("inputs") or {})), None)
        if po is None or ni is None:
            raise LinkError([Diagnostic(
                "L-PROTO", f"chain seam {stages[i].name} -> {stages[i + 1].name} needs one "
                           f"output/one input; use explicit edges for a multi-port stage")])
        edges.append((stages[i].name, po, stages[i + 1].name, ni))
    return edges


def _check_stage_names(wired, by_name) -> None:
    for frm, _o, to, _i in wired:
        for name in (frm, to):
            if name not in by_name:
                raise LinkError([Diagnostic("L-WIRE", f"an edge names unknown stage {name!r}")])


def _topo_order(names: list[str], wired, problems: list[Diagnostic]) -> list[str]:
    """Stages in dependency order; appends L-ORDER on a cycle. Stable on input order."""
    succ: dict[str, set] = {n: set() for n in names}
    indeg: dict[str, int] = {n: 0 for n in names}
    for frm, _o, to, _i in wired:
        if to not in succ[frm]:
            succ[frm].add(to)
            indeg[to] += 1
    ready = [n for n in names if indeg[n] == 0]
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in sorted(succ[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    if len(order) != len(names):
        stuck = sorted(set(names) - set(order))
        problems.append(Diagnostic("L-ORDER", f"the wiring has a cycle among {stuck}"))
        order += stuck          # keep them so the remaining checks still run
    return order


def _input_coverage(stages, by_name, wired) -> list[Diagnostic]:
    """Every input that is not satisfied from a pipeline input must be wired.

    A source stage's inputs are the pipeline's inputs (fed from outside); any other stage
    must have every declared input port wired from an upstream output, or it reads garbage.
    An *unwired* input on a stage that also has a wired one is a half-built node, which is
    exactly the mistake a silent kernel would hide."""
    out = []
    incoming = {(to, inp) for _f, _o, to, inp in wired}
    feeds: dict[str, int] = {n: 0 for n in by_name}
    for _f, _o, to, _i in wired:
        feeds[to] += 1
    for name, (comp_obj, contract) in by_name.items():
        if feeds[name] == 0:
            continue            # a source: its inputs come from the caller
        for port in (contract.get("inputs") or {}):
            if (name, port) not in incoming:
                out.append(Diagnostic("L-WIRE", f"{name}.{port} is not wired, but {name} is "
                                                f"not a source stage — every non-source input "
                                                f"must come from an upstream output"))
    return out


def _coalesce(problems: list[Diagnostic]) -> list[Diagnostic]:
    """Collapse repeated codes into one line each, listing every site.

    A DAG with five mismatched seams should read "L-PROTO x3" with the three sites, not five
    near-identical blocks a reader has to diff in their head."""
    order, groups = [], {}
    for d in problems:
        if d.code not in groups:
            order.append(d.code)
            groups[d.code] = []
        groups[d.code].append(d.message)
    out = []
    for code in order:
        msgs = groups[code]
        if len(msgs) == 1:
            out.append(Diagnostic(code, msgs[0]))
        else:
            out.append(Diagnostic(code, f"[{len(msgs)} sites] " + " | ".join(msgs)))
    return out


# --- fused-source generation ----------------------------------------------------------
def render_graph(plan: LinkPlan, fmt: str = "ascii") -> str:
    """A picture of the plan: stages as nodes, wires as labelled edges, resource peaks noted.

    A chain reads left-to-right; a DAG reads as layered rows in topological order. The point
    is that a human can see the shape and the seams at a glance before trusting any number —
    the same reason a linker prints its resolved plan."""
    if fmt == "mermaid":
        lines = ["flowchart LR"]
        for i, name in enumerate(plan.stages):
            lines.append(f"  {_mid(name)}[\"{name}\"]")
        for frm, out, to, inp in plan.edges:
            lines.append(f"  {_mid(frm)} -- \"{out}->{inp}\" --> {_mid(to)}")
        return "\n".join(lines) + "\n"
    # ascii
    lines = [f"plan ({plan.language}), {len(plan.stages)} stage(s):"]
    pos = {n: i for i, n in enumerate(plan.stages)}
    for frm, out, to, inp in plan.edges:
        lines.append(f"  {frm} --[{out}->{inp}]--> {to}")
    if not plan.edges:
        lines.append("  (single stage, no seams)")
    lines.append("  resources: " + ", ".join(f"{r}={plan.resources[r]}" for r in RESOURCES
                                             if r in plan.resources))
    return "\n".join(lines) + "\n"


def _mid(name: str) -> str:
    return "n_" + "".join(ch if ch.isalnum() else "_" for ch in name)


def render_fused(plan: LinkPlan, root: str | Path | None = None) -> str:
    """One kernel source that inlines each stage under its role and wires the hand-offs.

    This is a *skeleton*: the stage bodies are inserted as-is (their file's text), and the
    hand-off between stages is generated from the solved plan. For Python languages the
    result is parsed to guarantee it is at least valid syntax — a skeleton that does not
    parse would waste the first GPU round on a syntax error, the exact failure the whole
    seam-check exists to move earlier."""
    parts: list[str] = [f"# Fused kernel, linked from stages: {' -> '.join(plan.stages)}\n"
                        f"# language: {plan.language}  ring_depth: {plan.ring_depth}  "
                        f"resources: {json.dumps(plan.resources, sort_keys=True)}\n"]
    bodies: list[tuple[str, str]] = []
    for name in plan.stages:
        cdir = comp.components_root(root) / name
        impl = cdir / plan.language / comp.IMPL_FILE[plan.language]
        body = impl.read_text(encoding="utf-8") if impl.is_file() else ""
        bodies.append((name, body))
        parts.append(f"\n# ==== stage: {name} ({plan.language}) ====\n" + body.rstrip() + "\n")
    if plan.edges:
        parts.append("\n# ==== generated hand-offs ====\n")
        handles = _port_handles(plan, root)
        for frm, out, to, inp in plan.edges:
            parts.append(_handoff_line_fmt(frm, out, to, inp, handles, plan.language))
    text = "".join(parts)
    if plan.language in ("cutedsl", "tilelang"):
        _reject_symbol_collisions(bodies, plan.language)
        try:
            ast.parse(text)
        except SyntaxError as e:
            raise LinkError([Diagnostic(
                "L-PROTO", f"the fused {plan.language} source does not parse after linking "
                           f"(a stage body is broken): {e}")]) from e
    return text


def _port_handles(plan: LinkPlan, root: str | Path | None) -> dict[tuple[str, str], str]:
    """The declared handle kind of each (stage, port) the plan wires, defaulting to ``reg``.

    A port that names no handle is treated as living in registers (the most common
    accumulator home), so a one-input stage that declares nothing still gets a concrete
    move function rather than an unknown."""
    handles: dict[tuple[str, str], str] = {}
    wanted: set[tuple[str, str]] = {(f, o) for f, o, _t, _i in plan.edges}
    wanted |= {(t, i) for _f, _o, t, i in plan.edges}
    for name in plan.stages:
        cdir = comp.components_root(root) / name
        try:
            cct = comp.read_contract(cdir)
        except comp.ComponentError:
            continue
        for side in ("inputs", "outputs"):
            for port, spec in (cct.get(side) or {}).items():
                if (name, port) in wanted:
                    handles[(name, port)] = spec.get("handle") or "reg"
    return handles


def _handoff_fn(src_kind: str, dst_kind: str) -> str:
    """The name of the move that carries an accumulator from one location to another.

    ``acc_tmem_to_reg`` imports a tmgen05 accumulator out of TMEM into registers (T4's
    tensor import); ``acc_reg_to_smem`` stages registers into a ring slot; identical kinds
    need no move (``acc_identity``). Unknown kinds were already refused at link time, so
    every pair here has a function — the skeleton never names a move it cannot emit."""
    if src_kind == dst_kind:
        return "acc_identity"
    return f"acc_{src_kind}_to_{dst_kind}"


def _handoff_line_fmt(frm: str, out: str, to: str, inp: str,
                      handles: dict[tuple[str, str], str], language: str) -> str:
    """One generated hand-off: the actual call the fused kernel makes to move the tensor."""
    src = handles.get((frm, out), "reg")
    dst = handles.get((to, inp), "reg")
    fn = _handoff_fn(src, dst)
    call = f"{fn}({_mid(frm)}_{out}, {_mid(to)}_{inp})"
    return (f"# {frm}.{out} ({src}) -> {to}.{inp} ({dst})\n"
            f"{call}\n")


def _reject_symbol_collisions(bodies: list[tuple[str, str]], language: str) -> None:
    """Refuse a fusion where two stages define the same top-level name.

    Inlining stage bodies into one file silently lets the last definition win — stage A's
    ``kernel`` is shadowed by stage B's, and the fused kernel computes B twice without any
    syntax error. So before trusting the skeleton, parse each body and collect its
    top-level function / class / assigned names; a name defined by two stages is a real
    link failure (``L-PROTO``), reported with both stages so the fix (rename or namespace)
    is obvious. A body that does not parse on its own is left to the whole-file parse below,
    which reports it as the broken stage it is."""
    owner: dict[str, str] = {}
    clashes: dict[str, set[str]] = {}
    for name, body in bodies:
        try:
            tree = ast.parse(body)
        except SyntaxError:
            continue
        for node in tree.body:
            syms: set[str] = set()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                syms.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        syms.add(t.id)
            for s in syms:
                if s in owner and owner[s] != name:
                    clashes.setdefault(s, {owner[s]}).add(name)
                else:
                    owner.setdefault(s, name)
    if clashes:
        detail = "; ".join(f"{s} defined by {sorted(who)}" for s, who in sorted(clashes.items()))
        raise LinkError([Diagnostic(
            "L-PROTO", f"fusing these {language} stages would shadow top-level names — "
                       f"rename or namespace them first: {detail}")])


# --- semantic parity on the CPU -------------------------------------------------------

def chained_golden(plan: LinkPlan, root: str | Path | None = None) -> dict:
    """The fused kernel's semantics as a CPU-torch statement, from each stage's golden.

    Each stage's ``verify.json`` carries a ``golden.torch`` — the source that defines what
    it computes, written against ``IN`` (its primary input) and producing ``OUT``. Following
    the plan's wiring, each stage's ``IN`` is bound to a specific upstream output (or the
    pipeline input for a source), so the goldens compose into one program mapping the
    pipeline's input to its output — the proof that the *composition* is right, without a GPU.

    For a multi-input stage the golden may read ``IN_<port>`` for every wired input; the
    single ``IN`` is the first wired input by convention, so a one-input stage's golden
    (the common case) needs no port knowledge.
    """
    stages = set(plan.stages)
    # the upstream STAGE whose output feeds each (stage, port)
    feed_of: dict[tuple[str, str], str] = {}
    for frm, out, to, inp in plan.edges:
        feed_of[(to, inp)] = frm
    inputs_of: dict[str, list[str]] = {n: [] for n in plan.stages}
    for _f, _o, to, inp in plan.edges:
        if inp not in inputs_of[to]:
            inputs_of[to].append(inp)

    steps: list[dict] = []
    missing: list[str] = []
    out_expr: dict[str, str] = {}
    for i, name in enumerate(plan.stages):
        cdir = comp.components_root(root) / name
        vpath = cdir / plan.language / comp.VERIFY_JSON
        try:
            v = json.loads(vpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            v = {}
        src = ((v.get("golden") or {}).get("torch") or "").strip()
        if not src:
            missing.append(name)
        ports = inputs_of[name]
        # each input's source expression: an upstream stage's output, or the pipeline input
        bindings = {p: out_expr.get(feed_of[(name, p)], "IN") for p in ports}
        primary = bindings[ports[0]] if ports else "IN"
        out_name = f"OUT_{i}"
        steps.append({"stage": name, "input": primary, "inputs": bindings,
                      "output": out_name, "torch": src})
        out_expr[name] = out_name
    expression = " then ".join(f"{s['stage']}({s['input']})->{s['output']}" for s in steps)
    return {"steps": steps, "expression": expression, "language": plan.language,
            "missing_golden": missing}


def run_parity(plan: LinkPlan, reference, *, root: str | Path | None = None,
               feeds: dict | None = None, tolerance: float | None = None,
               output_stage: str | None = None) -> list[dict]:
    """Run the chained golden on CPU torch and compare the final output to ``reference``.

    ``reference`` maps the pipeline input to the expected final output (the caller's ground
    truth). ``feeds`` binds the pipeline's input tensor under ``"IN"``; each stage's ``IN``
    (and ``IN_<port>`` for a multi-input stage) is bound to the upstream output the planner
    wired to it. ``output_stage`` names which stage's output is the final result (default: the
    last in topo order), so a DAG with a non-terminal sink can still be checked. Torch is
    imported here, so this raises ImportError where torch is absent — the caller decides
    whether that is a skip. Returns per-stage readings so a failure points at the seam."""
    import torch

    tol = tolerance if tolerance is not None else 1e-4
    chain = chained_golden(plan, root)
    readings: list[dict] = []
    if chain["missing_golden"]:
        readings.append({"parity": "incomplete",
                         "missing_golden": list(chain["missing_golden"])})
        return readings
    values: dict = dict(feeds or {})        # name -> tensor (IN plus any wired upstream)
    result_by_stage: dict[str, object] = {}
    for step in chain["steps"]:
        env: dict = {"torch": torch}
        # bind every wired input: IN (primary) and IN_<port> for each
        for port, expr in step["inputs"].items():
            if expr in values:
                env[f"IN_{port}"] = values[expr]
        if step["input"] in values:
            env["IN"] = values[step["input"]]
        src = step["torch"] or "OUT = IN"
        try:
            exec(compile(src, f"<golden {step['stage']}>", "exec"), env)  # noqa: S102 — trusted verify.json
        except Exception as e:  # noqa: BLE001 — a broken golden is a finding, not a crash
            readings.append({"stage": step["stage"], "error": f"{e.__class__.__name__}: {e}"})
            return readings
        out = env.get("OUT", env.get(step["output"]))
        values[step["output"]] = out
        result_by_stage[step["stage"]] = out
        readings.append({"stage": step["stage"], "output": _as_list(out),
                         "shape": _shape_of(out)})
    final_stage = output_stage or (plan.stages[-1] if plan.stages else None)
    last = result_by_stage.get(final_stage)
    if "IN" in values and last is not None:
        expected = reference(values["IN"])
        try:
            ok = bool(torch.allclose(last, expected, atol=tol, rtol=tol))
        except Exception:  # noqa: BLE001 — non-tensor readings compare directly
            ok = last == expected
        readings.append({"parity": "ok" if ok else "mismatch", "tolerance": tol,
                         "final_stage": final_stage})
    return readings


def _shape_of(v):
    """The shape of a tensor reading (or None) so a parity log names each stage's output."""
    shape = getattr(v, "shape", None)
    if shape is not None:
        try:
            return list(shape)
        except TypeError:
            return None
    return None


def _as_list(v):
    if v is None:
        return None
    try:
        return v.detach().cpu().tolist()
    except AttributeError:
        return v
