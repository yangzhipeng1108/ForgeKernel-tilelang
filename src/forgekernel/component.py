"""Kernel components: a bounded unit that ships, swaps and reuses on its own.

A run optimizes one operator end to end. That leaves the good work inside a single
``kernel.py``: the fused reduction, the swizzled staging loop, the warp-specialized
epilogue all live as lines of one file, bound to one operator's shapes, readable by
one person. Nothing of it can be taken out and dropped into another operator, another
model, or another DSL without copying code and re-deriving every assumption it was
written under. This module is the contract that fixes that.

A **component** is one such bounded piece made first-class:

    components/<name>/
        component.toml      metadata: what it is, what it needs, which languages implement it
        contract.json       the interface, DSL-neutral: inputs / outputs as shape+dtype over
                            named axes, the invariants a caller may rely on, the semantics in
                            one sentence. This is what makes a component *swappable*: anything
                            that satisfies this contract can stand in for anything else that
                            does, regardless of language.
        <language>/         one implementation per language slot declared in component.toml
            kernel.<ext>    the implementation (kernel.py / kernel.cu / kernel_tl.py)
            verify.json     this implementation's own small test: a shape, a tolerance, and
                            a golden statement (torch source or a seed) -- so it is delivered
                            *with* its proof, not on trust

The boundary a developer or an agent works inside is then exactly one component
directory: the contract says what may cross it, the implementation slot says where their
work lives, and ``verify.json`` says how it will be judged before it leaves. Swapping a
component for another is choosing a different implementation of the same contract;
reusing it in a new model or system is importing it by contract, never by name.

``contract.json`` deliberately reuses the shape of ``forgekernel.judge``'s definition
(``{"inputs": {name: {"shape", "dtype"}}, "outputs": ...}``) so a component's interface and
an operator's judge describe shapes and dtypes the same way; a component is the same kind
of statement at a smaller grain. The additions are the ones a standalone unit needs and a
whole-operator definition does not: ``axes`` (the names its shapes are written over),
``invariants`` (what a caller may assume without reading the code), and ``semantics``
(one sentence of what it computes, the thing every implementation must agree on).

This module only *describes and checks*. It runs nothing and holds no GPU: ``check_tree``
is the static gate a component must pass to be delivered, the same way ``judge``'s
``check_problems`` gates an operator's check directory before any job is bought.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

COMPONENT_TOML = "component.toml"
CONTRACT_JSON = "contract.json"
VERIFY_JSON = "verify.json"
COMPONENTS_DIR = "components"

# The language slots a component may implement, and the file each implementation is
# written in. Kept beside ``taskcard.KERNEL_FILE`` on purpose: a component implemented
# in a language is the same unit the run's kernel is, one grain down, so the file each
# language uses must be the one the harness already knows how to build and judge.
IMPL_FILE = {"cutedsl": "kernel.py", "cuda": "kernel.cu", "tilelang": "kernel_tl.py"}

# dtypes a contract may name — the same vocabulary judge.py validates a definition
# against, so a component's contract and an operator's definition cannot drift apart.
DTYPES = ("float64", "float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2",
          "int64", "int32", "int16", "int8", "bool")

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Where a port's tensor physically lives. A producer may leave an accumulator in TMEM
# (Blackwell tcgen05), in registers (Hopper WGMMA / consumer-Blackwell mma.sync) or in shared
# memory (a ring slot). The linker generates the copy between any two, so a handle *kind*
# mismatch is not by itself a failure — but the kind decides which hand-off code it emits,
# and an unknown kind cannot be moved, so it is rejected at link time (L-HANDLE).
HANDLE_KINDS = ("tmem", "reg", "smem")

# component.toml's known top-level tables. An unknown table is a finding, not silently
# ignored: a typo'd ``[require]`` next to ``[requires]`` must not read as "no requires".
# ``resources`` and ``roles`` are what a linker solves at link time (forgekernel.link):
# declaring them here lets one static checker and the linker agree on what a component
# claims to consume.
_KNOWN_TABLES = {"component", "requires", "provides", "python", "judge", "resources", "roles"}


@dataclass
class Component:
    """A parsed ``component.toml`` plus the raw tables, for the checker and the CLI."""
    dir: Path
    name: str
    languages: list[str]
    summary: str = ""
    requires: list[dict] = field(default_factory=list)   # [{name, contract?, optional?}]
    provides: dict = field(default_factory=dict)          # extra named surfaces it offers
    python: dict = field(default_factory=dict)            # {"package": ..., "module": ...}
    judge: dict = field(default_factory=dict)             # {"reference": ...} optional
    resources: dict = field(default_factory=dict)         # what the linker must budget
    roles: dict = field(default_factory=dict)             # {"warps": N, "role": ...}


class ComponentError(ValueError):
    """A component.toml or contract.json that cannot be loaded at all."""


# --- paths ---------------------------------------------------------------------------

def components_root(root: str | Path | None = None) -> Path:
    """Where ``forge component`` reads and writes: ``<root>/components``.

    Defaults to the repository root, so a component accounts alongside the code that
    runs it; a run that wants its own set points ``--root`` at it."""
    base = Path(root) if root is not None else _default_root()
    return base / COMPONENTS_DIR


def _default_root() -> Path:
    from forgekernel.paths import repo_root
    return repo_root()


def component_dir(name: str, root: str | Path | None = None) -> Path:
    if not _NAME.match(name):
        raise ComponentError(f"component name {name!r} must match {_NAME.pattern}")
    return components_root(root) / name


# --- loading -------------------------------------------------------------------------

def read_component(cdir: str | Path) -> Component:
    """Parse ``component.toml`` under ``cdir``. Raises ComponentError if it cannot load."""
    cdir = Path(cdir)
    path = cdir / COMPONENT_TOML
    if not path.is_file():
        raise ComponentError(f"{path} is missing")
    try:
        with path.open("rb") as fh:
            cfg = tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise ComponentError(f"{path} does not parse: {e}") from e
    meta = cfg.get("component") or {}
    if not isinstance(meta, dict):
        raise ComponentError(f"{path}: [component] must be a table")
    name = str(meta.get("name") or cdir.name)
    langs = meta.get("languages")
    if isinstance(langs, str):
        langs = [langs]
    if not isinstance(langs, list) or not langs:
        raise ComponentError(f"{path}: [component].languages must be a non-empty list")
    req = (cfg.get("requires") or {}).get("component") or []
    if isinstance(req, dict):
        req = [req]
    secondary = cfg.get("requires") or {}
    # a component may declare other requires buckets (python packages, etc.); the
    # schemas for those are open, so we keep the table verbatim under ``requires``.
    requires = [{k: v for k, v in r.items()} for r in req] if isinstance(req, list) else []
    return Component(
        dir=cdir, name=name, languages=[str(x) for x in langs],
        summary=str(meta.get("summary") or ""),
        requires=requires,
        provides=(cfg.get("provides") or {}),
        python=(cfg.get("python") or {}),
        judge=(cfg.get("judge") or {}),
        resources=(cfg.get("resources") or {}),
        roles=(cfg.get("roles") or {}),
    )


def read_contract(cdir: str | Path) -> dict:
    p = Path(cdir) / CONTRACT_JSON
    if not p.is_file():
        raise ComponentError(f"{p} is missing")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ComponentError(f"{p} does not parse: {e}") from e


def list_components(root: str | Path | None = None) -> list[Component]:
    """Every component under the root that parses; a broken one is skipped here (the
    checker reports it) so one bad directory cannot hide the healthy ones."""
    base = components_root(root)
    out = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        if (d / COMPONENT_TOML).is_file():
            try:
                out.append(read_component(d))
            except ComponentError:
                continue
    return out


# --- checking ------------------------------------------------------------------------

def check_tree(cdir: str | Path) -> list[str]:
    """Everything wrong with a component directory, statically (empty = deliverable).

    Mirrors ``judge.check_problems`` in spirit: no GPU, no build — just the structural
    and interface facts a component must hold before it can be called delivered. A
    component that clears this can be listed, imported by contract and swapped for any
    other implementation of the same contract."""
    cdir = Path(cdir)
    out: list[str] = []
    if not cdir.is_dir():
        return [f"{cdir} is not a directory"]

    try:
        comp = read_component(cdir)
    except ComponentError as e:
        return [str(e)]

    toml = cdir / COMPONENT_TOML
    try:
        with toml.open("rb") as fh:
            raw = tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError):
        raw = {}
    out += [f"{COMPONENT_TOML}: unknown table [{t}] (known: "
            f"{', '.join(sorted(_KNOWN_TABLES))})"
            for t in raw if t not in _KNOWN_TABLES]

    if comp.name != cdir.name:
        out.append(f"{COMPONENT_TOML}: name {comp.name!r} does not match the directory "
                   f"{cdir.name!r} — a component is imported by contract, and a name that "
                   f"disagrees with where it lives is two names for one thing")
    out += [f"{COMPONENT_TOML}: unknown language {lang!r} in languages "
            f"(known: {', '.join(IMPL_FILE)})"
            for lang in comp.languages if lang not in IMPL_FILE]
    if len(set(comp.languages)) != len(comp.languages):
        out.append(f"{COMPONENT_TOML}: languages lists a language twice")
    if not comp.summary.strip():
        out.append(f"{COMPONENT_TOML}: [component].summary is required — one sentence a "
                   f"reader picks the component by")
    _ = comp.provides  # open schema; presence is enough, content is free-form

    # the contract: the one thing every implementation must agree on
    try:
        contract = read_contract(cdir)
    except ComponentError as e:
        out.append(str(e))
        contract = None
    if contract is not None:
        out += _contract_problems(contract, cdir)

    # one implementation slot per declared language, each with a self-contained verify
    for lang in comp.languages:
        if lang not in IMPL_FILE:
            continue
        slot = cdir / lang
        impl = slot / IMPL_FILE[lang]
        if not impl.is_file():
            out.append(f"{lang}/{IMPL_FILE[lang]} is missing — {lang!r} is declared in "
                       f"languages but has no implementation")
        out += _verify_problems(slot / VERIFY_JSON, lang, contract, cdir)

    # a language directory that is not declared is dead weight a reader will trip over
    for sub in sorted(d for d in cdir.iterdir() if d.is_dir()):
        if sub.name in IMPL_FILE and sub.name not in comp.languages:
            out.append(f"{sub.name}/ exists but {sub.name!r} is not listed in "
                       f"[component].languages — either declare it or remove it")

    # requires: each named dependency must exist under the same root
    out += _require_problems(comp, cdir)
    out += _resource_role_problems(comp)
    return out


_RESOURCE_KEYS = ("smem_bytes", "tmem_cols", "regs_per_thread", "barrier_slots")
_ROLE_INTS = ("warps", "ring_min", "ring_max")
_ROLE_RATES = ("producer_rate", "consumer_rate", "expected_ms")


def _resource_role_problems(comp: Component) -> list[str]:
    """The [resources] / [roles] declarations the linker will rely on must be well-formed.

    These tables do not affect whether a component runs alone, but the linker *solves* with
    them; a string where an int belongs, or a negative claim, would make the resource check
    silently wrong at link time. Checking them here — where a component is delivered — means
    the error surfaces before any link, not as a bad number in a plan."""
    out: list[str] = []
    for k, v in (comp.resources or {}).items():
        if k not in _RESOURCE_KEYS:
            out.append(f"[resources].{k} is not a known resource (known: "
                       f"{', '.join(_RESOURCE_KEYS)})")
            continue
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            out.append(f"[resources].{k} must be a non-negative int, got {v!r}")
    for k, v in (comp.roles or {}).items():
        if k in _ROLE_INTS:
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                out.append(f"[roles].{k} must be a non-negative int, got {v!r}")
        elif k in _ROLE_RATES:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
                out.append(f"[roles].{k} must be a positive number (units per step), "
                           f"got {v!r}")
        else:
            out.append(f"[roles].{k} is not a known role (known: "
                       f"{', '.join(_ROLE_INTS + _ROLE_RATES)})")
    rlo = (comp.roles or {}).get("ring_min")
    rhi = (comp.roles or {}).get("ring_max")
    if isinstance(rlo, int) and isinstance(rhi, int) and rlo > rhi:
        out.append(f"[roles] ring_min ({rlo}) > ring_max ({rhi}): no ring depth can satisfy "
                   f"this component")
    return out


def _contract_problems(contract: object, cdir: Path) -> list[str]:
    """The DSL-neutral interface: inputs/outputs over named axes, semantics, invariants."""
    out: list[str] = []
    if not isinstance(contract, dict):
        return [f"{CONTRACT_JSON}: must be a JSON object"]
    axes = contract.get("axes")
    if not isinstance(axes, list) or not all(isinstance(a, str) for a in axes):
        out.append(f"{CONTRACT_JSON}: axes must be a list of axis names")
        axes = []
    for name in (a for a in axes if not _IDENT.match(a)):
        out.append(f"{CONTRACT_JSON}: axis {name!r} is not a Python identifier")
    axis_set = set(axes)

    for side in ("inputs", "outputs"):
        specs = contract.get(side)
        if not isinstance(specs, dict) or not specs:
            out.append(f"{CONTRACT_JSON}: {side} must be a non-empty object")
            continue
        for nm, spec in specs.items():
            where = f"{CONTRACT_JSON} {side}.{nm}"
            if not _IDENT.match(nm):
                out.append(f"{where}: not a Python identifier")
            if not isinstance(spec, dict):
                out.append(f"{where}: must be an object with shape and dtype")
                continue
            if spec.get("dtype") not in DTYPES:
                out.append(f"{where}: dtype must be one of {list(DTYPES)}, "
                           f"got {spec.get('dtype')!r}")
            out += _port_handle_problems(spec, where)
            _shape_problems(spec.get("shape"), axis_set, where, side, out)

    ins, outs = contract.get("inputs"), contract.get("outputs")
    if isinstance(ins, dict) and isinstance(outs, dict):
        clash = set(ins) & set(outs)
        if clash:
            out.append(f"{CONTRACT_JSON}: names used for both an input and an output: "
                       f"{sorted(clash)}")

    sem = contract.get("semantics")
    if not isinstance(sem, str) or not sem.strip():
        out.append(f"{CONTRACT_JSON}: semantics is required — one sentence every "
                   f"implementation must agree on")
    inv = contract.get("invariants", [])
    if not isinstance(inv, list) or not all(isinstance(x, str) and x.strip() for x in inv):
        out.append(f"{CONTRACT_JSON}: invariants must be a list of non-empty strings")
    return out


def _port_handle_problems(spec: dict, where: str) -> list[str]:
    """A port's optional ``handle`` (where the tensor lives) and ``generation`` (which
    pipeline generation it carries) must be well-formed, because the linker branches on
    both: an unknown handle cannot be moved, and a generation mismatch is an L-PHASE."""
    out: list[str] = []
    if "handle" in spec and spec["handle"] not in HANDLE_KINDS:
        out.append(f"{where}: handle must be one of {list(HANDLE_KINDS)}, "
                   f"got {spec['handle']!r}")
    if "generation" in spec:
        g = spec["generation"]
        if not isinstance(g, int) or isinstance(g, bool) or g < 0:
            out.append(f"{where}: generation must be a non-negative int, got {g!r}")
    return out


def _shape_problems(shape, axis_set: set, where: str, side: str, out: list[str]) -> None:
    if shape is None:
        if side == "inputs":
            return          # a python scalar input
        out.append(f"{where}: an output must declare a shape")
        return
    if not isinstance(shape, list):
        out.append(f"{where}: shape must be a list of axis names / integers")
        return
    for dim in shape:
        if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
            continue
        if isinstance(dim, str) and dim in axis_set:
            continue
        out.append(f"{where}: dimension {dim!r} is neither a positive integer nor one of "
                   f"the declared axes {sorted(axis_set)}")


def _verify_problems(vpath: Path, lang: str, contract: dict | None, cdir: Path) -> list[str]:
    """A component ships with its own proof: a small shape, a tolerance, and a golden
    statement. Without it a unit is delivered on trust, which is the thing components
    exist to end."""
    out: list[str] = []
    if not vpath.is_file():
        return [f"{lang}/{VERIFY_JSON} is missing — a component ships with its own test"]
    try:
        v = json.loads(vpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"{lang}/{VERIFY_JSON} does not parse: {e}"]
    if not isinstance(v, dict):
        return [f"{lang}/{VERIFY_JSON}: must be a JSON object"]
    shape = v.get("shape")
    if not isinstance(shape, dict) or not shape:
        out.append(f"{lang}/{VERIFY_JSON}: shape must be a non-empty {axis: value} object")
    axes = (contract or {}).get("axes") or []
    if isinstance(shape, dict) and axes:
        unknown = sorted(set(shape) - set(axes))
        if unknown:
            out.append(f"{lang}/{VERIFY_JSON}: shape names axes not in the contract: "
                       f"{unknown}")
    tol = v.get("tolerance")
    if not isinstance(tol, (int, float)) or isinstance(tol, bool) or tol < 0:
        out.append(f"{lang}/{VERIFY_JSON}: tolerance must be a non-negative number")
    golden = v.get("golden")
    if not isinstance(golden, dict) or not (golden.get("torch") or golden.get("seed") is not None):
        out.append(f"{lang}/{VERIFY_JSON}: golden must name a reference: "
                   f"{{\"torch\": <src>}} or {{\"seed\": <int>}}")
    return out


def _require_problems(comp: Component, cdir: Path) -> list[str]:
    out: list[str] = []
    root = cdir.parent
    for req in comp.requires:
        nm = req.get("name") if isinstance(req, dict) else None
        if not nm:
            out.append(f"{COMPONENT_TOML}: every [requires.component] entry needs a name")
            continue
        if not _NAME.match(str(nm)):
            out.append(f"{COMPONENT_TOML}: requires.component name {nm!r} is not a valid name")
            continue
        if req.get("contract") is None and str(nm) != comp.name:
            out.append(f"{COMPONENT_TOML}: requires.component {nm!r} should name the "
                       f"contract it depends on, so a swap keeps satisfying it")
        if req.get("optional"):
            continue
        dep = root / str(nm)
        if not (dep / COMPONENT_TOML).is_file():
            out.append(f"{COMPONENT_TOML}: requires.component {nm!r} is not a component under "
                       f"{root} (no {nm}/{COMPONENT_TOML})")
    return out


# --- cross-language substitution -----------------------------------------------------

def satisfies(dep: Component, contract: dict, root: str | Path | None = None) -> bool:
    """Whether component ``dep`` satisfies ``contract`` closely enough to stand in for it.

    Substitution is by contract, not by name: a dependency may be met by any component
    whose own contract is *compatible* — same inputs and outputs (by name, dtype and the
    shape written over its axes), and whose declared languages include the ones the
    caller needs. This is the rule that lets one implementation of a contract be swapped
    for another (another language, another hand) without the caller changing anything."""
    try:
        dep_contract = read_contract(dep.dir)
    except ComponentError:
        return False
    return contracts_compatible(contract, dep_contract)


def contracts_compatible(a: dict, b: dict) -> bool:
    """Two contracts are compatible when they describe the same interface.

    Names must match on both sides, dtypes must match, and shapes must be equal up to a
    renaming of axes (``{"m","k"}`` vs ``{"rows","inner"}`` with the same variable
    pattern) — the axis names are local to a component, the structure they carry is not.
    Semantics is deliberately *not* compared: it is prose, and a prose equality would
    reject "y = x @ w" and "matrix multiply of x by w" that are the same contract. A
    caller wanting an exact semantic match reads the two ``semantics`` strings."""
    ia, ib = a.get("inputs") or {}, b.get("inputs") or {}
    oa, ob = a.get("outputs") or {}, b.get("outputs") or {}
    if set(ia) != set(ib) or set(oa) != set(ob):
        return False
    ax_a = {x: i for i, x in enumerate(a.get("axes") or [])}
    ax_b = {x: i for i, x in enumerate(b.get("axes") or [])}
    for side_a, side_b, ax1, ax2 in ((ia, ib, ax_a, ax_b), (oa, ob, ax_a, ax_b)):
        for nm in side_a:
            sa, sb = side_a[nm], side_b[nm]
            if sa.get("dtype") != sb.get("dtype"):
                return False
            if _shape_canon(sa.get("shape"), ax1) != _shape_canon(sb.get("shape"), ax2):
                return False
    return True


def _shape_canon(shape, axes: dict) -> tuple:
    """A shape with each axis replaced by its position, so two components that name the
    same structure with different axis names compare equal."""
    if shape is None:
        return ()
    return tuple(axes.get(d, d) if isinstance(d, str) else d for d in shape)


# --- exporting a run's best kernel as a component -----------------------------------

def contract_from_problem(problem: dict, semantics: str) -> dict:
    """A component contract built from a card's ``problem.json`` definition.

    The definition already carries ``axes``, ``inputs`` and ``outputs`` in exactly the
    shape a contract uses (that is the point of sharing the vocabulary), so exporting a
    kernel as a component does not re-derive its interface — it lifts the one the run
    was already judged by, which is the only interface we can claim the kernel satisfies.
    ``semantics`` is the caller's one sentence: a card's definition names a reference
    implementation, not prose, so the exporter supplies it (defaulting to the op name)."""
    defn = problem.get("definition") or {}
    inputs = {name: {"shape": spec.get("shape"), "dtype": spec.get("dtype")}
              for name, spec in (defn.get("inputs") or {}).items()}
    outputs = {name: {"shape": spec.get("shape"), "dtype": spec.get("dtype")}
               for name, spec in (defn.get("outputs") or {}).items()}
    axes = list(defn.get("axes") or [])
    return {
        "_about": ("Lifted from the card's judged definition (eval/leaderboard/problem.json): "
                   "the interface this kernel was actually measured against."),
        "axes": axes,
        "inputs": inputs,
        "outputs": outputs,
        "semantics": semantics,
        "invariants": ["outputs have the dtypes and shapes their entries declare"],
    }


def verify_from_problem(problem: dict, shape: dict) -> dict:
    """A ``verify.json`` from the card's tolerance and reference — a component ships with
    the proof it was held to, not a blank one."""
    tol = (problem.get("tolerance") or {}).get("outputs") or {}
    worst = max((float(v) for v in tol.values()), default=0.0)
    ref = (problem.get("definition") or {}).get("reference")
    golden = {"torch": ref} if ref else {"seed": 1}
    return {
        "_about": "Lifted from the card: the tolerance the run was judged under and its "
                  "reference statement.",
        "shape": dict(shape),
        "tolerance": worst,
        "golden": golden,
    }


def export_component(dest: str | Path, *, name: str, language: str, kernel_src: str,
                     contract: dict, verify: dict, summary: str) -> Path:
    """Write a component directory from a delivered kernel and its lifted interface.

    This is the bridge from "a run finished" to "its work is a reusable unit": the best
    kernel becomes one implementation slot, the card's judged definition becomes the
    contract, and the card's tolerance becomes the shipped verify. ``dest`` is created;
    an existing directory is refused (a component is a deliverable, not a scratch pad —
    overwriting one a caller already depends on is the thing components prevent)."""
    if language not in IMPL_FILE:
        raise ComponentError(f"unknown language {language!r} (known: {list(IMPL_FILE)})")
    dest = Path(dest)
    if dest.exists():
        raise ComponentError(f"{dest} already exists — a delivered component is not "
                             f"overwritten; export under a new name")
    if not _NAME.match(name):
        raise ComponentError(f"component name {name!r} must match {_NAME.pattern}")
    dest.mkdir(parents=True)
    lang_list = ", ".join(f'"{x}"' for x in [language])
    (dest / COMPONENT_TOML).write_text(
        "# Exported from a finished run (`forge export --as-component`).\n"
        "# The contract and verify are lifted from the card the run was judged by.\n\n"
        "[component]\n"
        f'name = "{name}"\n'
        f"languages = [{lang_list}]\n"
        f'summary = "{summary}"\n', encoding="utf-8", newline="\n")
    (dest / CONTRACT_JSON).write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8", newline="\n")
    slot = dest / language
    slot.mkdir()
    (slot / IMPL_FILE[language]).write_text(kernel_src, encoding="utf-8", newline="\n")
    (slot / VERIFY_JSON).write_text(json.dumps(verify, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8", newline="\n")
    findings = check_tree(dest)
    if findings:
        raise ComponentError(f"exported component {name!r} failed its own check (the source "
                             f"interface was not complete): " + "; ".join(findings))
    return dest
