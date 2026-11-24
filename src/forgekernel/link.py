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
  a kernel that hangs.
* **resource and role solving** — every stage declares what it consumes (shared memory,
  TMEM, registers, barrier slots, warps) and the caller gives the target's caps; the
  linker sums the per-stage claims, finds the tightest ring depth, and refuses a plan that
  exceeds a cap or double-books a role. Pure arithmetic, so it is exact and testable.
* **fused-source generation** — the linked plan renders one kernel source skeleton in
  which each stage's body is inlined under its declared role, wired by generated hand-off
  code. The skeleton is syntactically valid Python (or CUDA-shaped C++), so a downstream
  step compiles it; here it is checked by ``ast.parse`` (Python) — no GPU, no build.
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
    "L-PROTO": "two linked stages disagree on an interface (name / dtype / shape)",
    "L-WIRE": "the pipeline names a stage, or a port, that does not exist",
    "L-RES": "the summed per-stage resource claim exceeds the target's cap",
    "L-ROLE": "the summed warp roles exceed the block, or a role is double-booked",
    "L-RING": "the pipeline asks for a ring depth outside what the stages allow",
    "L-ORDER": "the pipeline is empty or does not form a single chain",
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
    resources: dict                         # solved per-resource totals
    ring_depth: int
    caps: dict = field(default_factory=dict)

    def manifest(self) -> dict:
        return {"stages": self.stages, "language": self.language,
                "edges": [list(e) for e in self.edges], "resources": self.resources,
                "ring_depth": self.ring_depth, "caps": self.caps}


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
                   nxt_name: str, nxt_contract: dict, in_port: str) -> list[Diagnostic]:
    """Whether ``prev``'s ``out_port`` can feed ``nxt``'s ``in_port``.

    A seam is well-typed when dtype matches and the shape structure lines up over the
    stages' own axes. This is the per-port form so a DAG can wire any output to any input;
    the chain case is the special case where each stage has exactly one of each."""
    p_out = prev_contract.get("outputs") or {}
    n_in = nxt_contract.get("inputs") or {}
    if out_port not in p_out:
        return [Diagnostic("L-WIRE", f"{prev_name} has no output {out_port!r} "
                                     f"(outputs: {', '.join(p_out) or 'none'})")]
    if in_port not in n_in:
        return [Diagnostic("L-WIRE", f"{nxt_name} has no input {in_port!r} "
                                     f"(inputs: {', '.join(n_in) or 'none'})")]
    po_spec, ni_spec = p_out[out_port], n_in[in_port]
    if _same_port(po_spec, ni_spec, prev_contract, nxt_contract):
        return []
    return [Diagnostic(
        "L-PROTO", f"{prev_name}.{out_port} ({_desc(po_spec)}) does not match "
                   f"{nxt_name}.{in_port} ({_desc(ni_spec)})")]


def _same_port(a: dict, b: dict, ca: dict, cb: dict) -> bool:
    if a.get("dtype") != b.get("dtype"):
        return False
    return comp._shape_canon(a.get("shape"), {x: i for i, x in enumerate(ca.get("axes") or [])}) \
        == comp._shape_canon(b.get("shape"), {x: i for i, x in enumerate(cb.get("axes") or [])})


def _desc(spec: dict) -> str:
    return f"{spec.get('dtype')} {spec.get('shape')}"


# --- resource / role solving ----------------------------------------------------------

def _solve(stages: list[comp.Component], caps: dict, ring_request: int | None) -> tuple[dict, int, list[Diagnostic]]:
    problems: list[Diagnostic] = []
    totals = {r: 0 for r in RESOURCES}
    warps = 0
    ring_lo, ring_hi = 0, None
    for c in stages:
        for r in RESOURCES:
            v = (c.resources or {}).get(r, 0)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                problems.append(Diagnostic("L-RES", f"{c.name}: {r} must be a non-negative int, "
                                                     f"got {v!r}"))
                continue
            totals[r] += v
        w = (c.roles or {}).get("warps", 0)
        if not isinstance(w, int) or isinstance(w, bool) or w < 0:
            problems.append(Diagnostic("L-ROLE", f"{c.name}: warps must be a non-negative int, "
                                                 f"got {w!r}"))
        else:
            warps += w
        rlo = (c.roles or {}).get("ring_min", 0)
        rhi = (c.roles or {}).get("ring_max")
        ring_lo = max(ring_lo, int(rlo) if isinstance(rlo, int) else 0)
        if isinstance(rhi, int):
            ring_hi = rhi if ring_hi is None else min(ring_hi, rhi)

    for r in RESOURCES:
        cap = caps.get(r)
        if cap is not None and totals[r] > cap:
            problems.append(Diagnostic("L-RES", f"{r} {totals[r]} exceeds the cap {cap}"))
    wcap = caps.get("warps")
    if wcap is not None and warps > wcap:
        problems.append(Diagnostic("L-ROLE", f"warps {warps} exceeds the block cap {wcap}"))

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
    totals["warps"] = warps
    return totals, ring, problems


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
        problems += _seam_for_port(frm, pct, out, to, nct, inp)
    problems += _input_coverage(stages, by_name, wired)

    totals, ring_depth, res_problems = _solve(stages, capsule, ring)
    problems += res_problems
    if problems:
        raise LinkError(_coalesce(problems))

    return LinkPlan(stages=order, language=lang,
                    edges=[(a, b, c, d) for a, b, c, d in wired],
                    resources=totals, ring_depth=ring_depth, caps=caps or {})


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
    for name in plan.stages:
        cdir = comp.components_root(root) / name
        impl = cdir / plan.language / comp.IMPL_FILE[plan.language]
        body = impl.read_text(encoding="utf-8") if impl.is_file() else ""
        parts.append(f"\n# ==== stage: {name} ({plan.language}) ====\n" + body.rstrip() + "\n")
    if plan.edges:
        parts.append("\n# ==== generated hand-offs ====\n")
        for frm, out, to, inp in plan.edges:
            parts.append(f"# {frm}.{out} -> {to}.{inp}\n")
    text = "".join(parts)
    if plan.language in ("cutedsl", "tilelang"):
        try:
            ast.parse(text)
        except SyntaxError as e:
            raise LinkError([Diagnostic(
                "L-PROTO", f"the fused {plan.language} source does not parse after linking "
                           f"(a stage body is broken): {e}")]) from e
    return text


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
    out_expr: dict[str, str] = {}
    for i, name in enumerate(plan.stages):
        cdir = comp.components_root(root) / name
        vpath = cdir / plan.language / comp.VERIFY_JSON
        try:
            v = json.loads(vpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            v = {}
        src = ((v.get("golden") or {}).get("torch") or "").strip()
        ports = inputs_of[name]
        # each input's source expression: an upstream stage's output, or the pipeline input
        bindings = {p: out_expr.get(feed_of[(name, p)], "IN") for p in ports}
        primary = bindings[ports[0]] if ports else "IN"
        out_name = f"OUT_{i}"
        steps.append({"stage": name, "input": primary, "inputs": bindings,
                      "output": out_name, "torch": src})
        out_expr[name] = out_name
    expression = " then ".join(f"{s['stage']}({s['input']})->{s['output']}" for s in steps)
    return {"steps": steps, "expression": expression, "language": plan.language}


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
    values: dict = dict(feeds or {})        # name -> tensor (IN plus any wired upstream)
    readings: list[dict] = []
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
        readings.append({"stage": step["stage"], "output": _as_list(out)})
    final_stage = output_stage or (plan.stages[-1] if plan.stages else None)
    last = result_by_stage.get(final_stage)
    if "IN" in values and last is not None:
        expected = reference(values["IN"])
        try:
            ok = bool(torch.allclose(last, expected, atol=tol, rtol=tol))
        except Exception:  # noqa: BLE001 — non-tensor readings compare directly
            ok = last == expected
        readings.append({"parity": "ok" if ok else "mismatch", "tolerance": tol})
    return readings


def _as_list(v):
    if v is None:
        return None
    try:
        return v.detach().cpu().tolist()
    except AttributeError:
        return v
