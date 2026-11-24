"""forge component — create, list and check kernel components (forgekernel.component).

    forge component new <name> [--language cutedsl,cuda,tilelang] [--summary "..."]
    forge component list [--root DIR]
    forge component check [<name> | --all] [--root DIR]
    forge component link <a> <b> ... [--emit DIR] [--cap KEY=INT] [--ring N]

A component is a bounded unit that ships, swaps and reuses on its own: the interface in
``contract.json``, one implementation per language slot, and each implementation's own
``verify.json``. ``check`` is the static gate it must pass to be called deliverable —
the same role ``judge.check_problems`` plays for an operator's check directory, one
grain down. Nothing here runs a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forgekernel import component as comp

_STUB_SUMMARY = "one sentence: what this component computes, independent of any operator"


def _toml_text(name: str, languages: list[str], summary: str) -> str:
    langs = ", ".join(f'"{x}"' for x in languages)
    return (
        "# A kernel component (forgekernel.component): a bounded unit that ships, swaps and\n"
        "# reuses on its own. Fill in [component], the contract, and one implementation per\n"
        "# language. `forge component check` must pass before it is delivered.\n\n"
        "[component]\n"
        f'name = "{name}"\n'
        f'languages = [{langs}]\n'
        f'summary = "{summary}"\n'
        "\n# What this component needs from others, by contract (not by package path), so any\n"
        "# implementation satisfying the contract can fill the slot.\n"
        "# [[requires.component]]\n"
        '# name = "rope"\n'
        '# contract = "rope/contract.json"\n'
        '# optional = false\n'
        "\n# Extra named surfaces it offers a caller (free-form; open schema).\n"
        "# [provides]\n"
        '# tile = "the staging tile it is built around"\n'
        "\n# Optional: the Python package a same-language caller imports it through.\n"
        '# [python]\n'
        '# package = "kernels._components"\n'
        '# module = "attention.flash_decode"\n'
    )


def _contract_text(name: str, languages: list[str]) -> str:
    doc = {
        "_about": (
            "The DSL-neutral interface every implementation must satisfy. Same shape as a "
            "task's judge definition (inputs/outputs as shape+dtype), plus axes, invariants "
            "and one sentence of semantics. A component that satisfies this contract can be "
            "swapped for any other that does, in any language."),
        "axes": ["m", "n"],
        "inputs": {
            "x": {"shape": ["m", "n"], "dtype": "float32"},
        },
        "outputs": {
            "y": {"shape": ["m", "n"], "dtype": "float32"},
        },
        "semantics": "y = x (replace with the one sentence that defines this component)",
        "invariants": ["x and y have the same shape and dtype"],
    }
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def _verify_text(lang: str) -> str:
    doc = {
        "_about": (
            "This implementation's own test: a small concrete shape, a tolerance, and a "
            "golden statement (torch source) or a seed. It travels with the implementation so "
            "the unit is delivered with its proof, not on trust."),
        "shape": {"m": 4, "n": 8},
        "tolerance": 0.0,
        "golden": {"torch": "y = x"},
    }
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def _impl_stub(lang: str, name: str) -> str:
    fname = comp.IMPL_FILE[lang]
    if lang == "cuda":
        return (f"// {name} — CUDA implementation of {comp.CONTRACT_JSON}'s contract.\n"
                f"// Replace the body; keep the interface the contract names.\n")
    if lang == "tilelang":
        return (f"# {name} — TileLang implementation of {comp.CONTRACT_JSON}'s contract.\n"
                f"# Replace the body; keep the interface the contract names.\n"
                f"import tilelang\n")
    return (f'"""\'{name}\' — CuTe DSL implementation of {comp.CONTRACT_JSON}\'s contract.\n\n'
            f'Replace the body; keep the interface the contract names."""\n')


def cmd_new(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="forge component new")
    p.add_argument("name")
    p.add_argument("--language", default="cutedsl",
                   help="comma-separated language slots to scaffold (default: cutedsl)")
    p.add_argument("--summary", default=_STUB_SUMMARY)
    p.add_argument("--root", default=None, help="project root (default: the repository)")
    p.add_argument("--from-run", default=None, metavar="RUN",
                   help="scaffold from a finished run's best kernel: contract + verify are "
                        "lifted from the card it was judged by (see forge export)")
    a = p.parse_args(argv)
    if a.from_run:
        return _new_from_run(a)
    languages = [x.strip() for x in a.language.split(",") if x.strip()]
    bad = [x for x in languages if x not in comp.IMPL_FILE]
    if bad:
        sys.stderr.write(f"[component] error: unknown language(s) {bad}; "
                         f"known: {list(comp.IMPL_FILE)}\n")
        return 2
    try:
        cdir = comp.component_dir(a.name, a.root)
    except comp.ComponentError as e:
        sys.stderr.write(f"[component] error: {e}\n")
        return 2
    if cdir.exists():
        sys.stderr.write(f"[component] error: {cdir} already exists\n")
        return 2
    (cdir).mkdir(parents=True)
    (cdir / comp.COMPONENT_TOML).write_text(_toml_text(a.name, languages, a.summary),
                                            encoding="utf-8", newline="\n")
    (cdir / comp.CONTRACT_JSON).write_text(_contract_text(a.name, languages),
                                           encoding="utf-8", newline="\n")
    for lang in languages:
        slot = cdir / lang
        slot.mkdir()
        (slot / comp.IMPL_FILE[lang]).write_text(_impl_stub(lang, a.name),
                                                 encoding="utf-8", newline="\n")
        (slot / comp.VERIFY_JSON).write_text(_verify_text(lang), encoding="utf-8", newline="\n")
    print(f"[component] created {cdir}")
    print(f"[component] next: edit {comp.COMPONENT_TOML} + {comp.CONTRACT_JSON} + each "
          f"implementation, then `forge component check {a.name}`")
    return 0


def _new_from_run(a) -> int:
    """Scaffold a component from a finished run — the same delivery ``forge export
    --as-component`` makes, reachable from the component command and rooting at --root."""
    import json as _json
    from forgekernel import launch, taskcard
    try:
        run = launch.open_run(a.from_run)
        task = launch.load_task(run.task_dir, name=run.task_name)
    except Exception as e:  # noqa: BLE001 — RunError/TaskError share a base we cannot name here
        sys.stderr.write(f"[component] error: cannot open run {a.from_run!r}: {e}\n")
        return 2
    best = launch.best_of(run, task)
    if best is None or not best.get("kernel"):
        sys.stderr.write("[component] error: the run has no confirmed best kernel to export\n")
        return 2
    card = run.operators_dir / task.name
    problem_path = card / taskcard.PROBLEM
    try:
        problem = _json.loads(problem_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.stderr.write(f"[component] error: cannot read the card's definition "
                         f"({problem_path}): {e}\n")
        return 2
    desc = (getattr(task, "description", None) or "").strip()
    first = next((ln.strip() for ln in desc.splitlines() if ln.strip()), "") or \
        f"{task.name} (exported from a finished run)"
    contract = comp.contract_from_problem(problem, first)
    shape = dict(task.shapes[0].axes) if getattr(task, "shapes", None) else {}
    verify = comp.verify_from_problem(problem, shape)
    try:
        dest = comp.export_component(comp.component_dir(a.name, a.root), name=a.name,
                                     language=task.language, kernel_src=best["kernel"],
                                     contract=contract, verify=verify, summary=first)
    except comp.ComponentError as e:
        sys.stderr.write(f"[component] error: {e}\n")
        return 2
    print(f"[component] created {dest} from {a.from_run} (best {best['sha'][:12]})")
    return 0


def cmd_list(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="forge component list")
    p.add_argument("--root", default=None)
    a = p.parse_args(argv)
    comps = comp.list_components(a.root)
    if not comps:
        print(f"(no components under {comp.components_root(a.root)})")
        return 0
    for c in comps:
        langs = ",".join(c.languages)
        print(f"{c.name:32} [{langs}]  {c.summary}")
    return 0


def cmd_link(argv: list[str]) -> int:
    """Link components into one fused-kernel plan — as a chain, or a DAG via --edge.

    Pure CPU: it checks seams, solves resources/roles against caps, and (with --emit) writes
    the fused-source skeleton and the plan manifest. It does not compile or run a GPU."""
    from forgekernel import link as linkmod
    p = argparse.ArgumentParser(prog="forge component link")
    p.add_argument("pipeline", nargs="+",
                   help="components in link order; each may name its language as name@lang")
    p.add_argument("--root", default=None)
    p.add_argument("--language", default=None, help="force one language for every stage")
    p.add_argument("--cap", action="append", default=[], metavar="KEY=INT",
                   help=f"override a resource cap ({', '.join(linkmod.RESOURCES)} or warps)")
    p.add_argument("--ring", type=int, default=None, help="requested ring depth")
    p.add_argument("--edge", action="append", default=[], metavar="FROM.OUT=TO.IN",
                   help="wire an explicit seam (repeatable); with any --edge the pipeline is "
                        "treated as a DAG instead of a chain")
    p.add_argument("--emit", default=None, metavar="DIR",
                   help="write the fused source and link-plan.json into DIR")
    p.add_argument("--format", choices=["text", "json"], default="text",
                   help="output format (json is CI-friendly)")
    a = p.parse_args(argv)
    caps = _parse_caps(a.cap)
    if caps is None:
        return 2
    wiring = _parse_edges(a.edge)
    if wiring is _BAD:
        return 2
    try:
        plan = linkmod.link(a.pipeline, root=a.root, language=a.language, caps=caps or None,
                            ring=a.ring, edges=wiring)
    except linkmod.LinkError as e:
        if a.format == "json":
            print(json.dumps({"ok": False,
                              "diagnostics": [{"code": d.code, "message": d.message}
                                              for d in e.diagnostics]}, indent=2))
        else:
            sys.stderr.write(f"[component] link refused ({len(e.diagnostics)} diagnostic(s)):\n")
            for d in e.diagnostics:
                sys.stderr.write(f"  {d}\n")
        return 1
    if a.format == "json":
        print(json.dumps({"ok": True, **plan.manifest()}, indent=2))
    else:
        for line in linkmod.summarize(plan):
            print(f"[component] {line}")
    if a.emit:
        out = Path(a.emit)
        try:
            src = linkmod.render_fused(plan, a.root)
        except linkmod.LinkError as e:
            # fusing can still refuse (e.g. two stages shadow a top-level name); report it
            # the same way a refused link is reported rather than dumping a traceback.
            if a.format == "json":
                print(json.dumps({"ok": False,
                                  "diagnostics": [{"code": d.code, "message": d.message}
                                                  for d in e.diagnostics]}, indent=2))
            else:
                sys.stderr.write(f"[component] fusion refused "
                                 f"({len(e.diagnostics)} diagnostic(s)):\n")
                for d in e.diagnostics:
                    sys.stderr.write(f"  {d}\n")
            return 1
        out.mkdir(parents=True, exist_ok=True)
        ext = comp.IMPL_FILE.get(plan.language, "kernel.txt").split(".")[-1]
        (out / f"fused_kernel.{ext}").write_text(src, encoding="utf-8", newline="\n")
        (out / "link-plan.json").write_text(
            json.dumps(plan.manifest(), indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"[component] wrote {out / ('fused_kernel.' + ext)} and {out / 'link-plan.json'}")
    return 0


_BAD = object()


def _parse_caps(specs: list[str]) -> dict | None:
    caps: dict = {}
    for spec in specs:
        if "=" not in spec:
            sys.stderr.write(f"[component] error: --cap expects KEY=INT, got {spec!r}\n")
            return None
        k, v = spec.split("=", 1)
        try:
            caps[k.strip()] = int(v)
        except ValueError:
            sys.stderr.write(f"[component] error: --cap {spec!r}: value is not an int\n")
            return None
    return caps


def _parse_edges(specs: list[str]):
    """Parse --edge FROM.OUT=TO.IN specs into tuples; returns _BAD on a malformed one."""
    wiring = None
    for spec in specs:
        if "=" not in spec or "." not in spec:
            sys.stderr.write(f"[component] error: --edge expects FROM.OUT=TO.IN, got {spec!r}\n")
            return _BAD
        lhs, rhs = spec.split("=", 1)
        try:
            frm, out = lhs.rsplit(".", 1)
            to, inp = rhs.rsplit(".", 1)
        except ValueError:
            sys.stderr.write(f"[component] error: --edge {spec!r}: both sides need NAME.PORT\n")
            return _BAD
        wiring = wiring or []
        wiring.append((frm.strip(), out.strip(), to.strip(), inp.strip()))
    return wiring


def cmd_check(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="forge component check")
    p.add_argument("name", nargs="?", default=None)
    p.add_argument("--all", action="store_true", help="check every component under the root")
    p.add_argument("--root", default=None)
    a = p.parse_args(argv)
    if not a.name and not a.all:
        sys.stderr.write("[component] error: give a component name or --all\n")
        return 2
    if a.all:
        dirs = [c.dir for c in comp.list_components(a.root)]
        broken = [d for d in comp.components_root(a.root).glob(f"*/{comp.COMPONENT_TOML}")
                  if d.parent not in dirs]
        dirs += [d.parent for d in broken]
        if not dirs:
            print(f"(no components under {comp.components_root(a.root)})")
            return 0
    else:
        dirs = [comp.component_dir(a.name, a.root)]
    failed = 0
    for d in dirs:
        findings = comp.check_tree(d)
        if findings:
            failed += 1
            print(f"[component] {d.name}: NOT deliverable")
            for f in findings:
                print(f"  - {f}")
        else:
            print(f"[component] {d.name}: ok")
    return 1 if failed else 0


def cmd_plan(argv: list[str]) -> int:
    """Print the solved link plan (wiring, resource peaks, ring) without emitting sources.

    A dry run of ``link``: same seam checks and solving, human summary plus the per-resource
    reasoning, optionally an ASCII/Mermaid graph. Nothing is written."""
    from forgekernel import link as linkmod
    p = argparse.ArgumentParser(prog="forge component plan")
    p.add_argument("pipeline", nargs="+")
    p.add_argument("--root", default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--cap", action="append", default=[], metavar="KEY=INT")
    p.add_argument("--ring", type=int, default=None)
    p.add_argument("--edge", action="append", default=[], metavar="FROM.OUT=TO.IN")
    p.add_argument("--graph", choices=["ascii", "mermaid"], default=None,
                   help="also print a picture of the plan")
    p.add_argument("--format", choices=["text", "json"], default="text")
    a = p.parse_args(argv)
    caps = _parse_caps(a.cap)
    if caps is None:
        return 2
    wiring = _parse_edges(a.edge)
    if wiring is _BAD:
        return 2
    try:
        plan = linkmod.link(a.pipeline, root=a.root, language=a.language, caps=caps or None,
                            ring=a.ring, edges=wiring)
    except linkmod.LinkError as e:
        if a.format == "json":
            print(json.dumps({"ok": False, "diagnostics": [
                {"code": d.code, "message": d.message} for d in e.diagnostics]}, indent=2))
        else:
            sys.stderr.write(f"[component] plan refused ({len(e.diagnostics)} diagnostic(s)):\n")
            for d in e.diagnostics:
                sys.stderr.write(f"  {d}\n")
        return 1
    if a.format == "json":
        print(json.dumps({"ok": True, **plan.manifest()}, indent=2))
        return 0
    for line in linkmod.summarize(plan):
        print(f"[component] {line}")
    if a.graph:
        print(linkmod.render_graph(plan, a.graph), end="")
    return 0


def cmd_diff(argv: list[str]) -> int:
    """Re-link two pipelines and report what changed (e.g. after swapping one stage)."""
    from forgekernel import link as linkmod
    p = argparse.ArgumentParser(prog="forge component diff")
    p.add_argument("before", nargs="+", help="the original pipeline (up to --then)")
    p.add_argument("--then", nargs="+", required=True, dest="after",
                   help="the new pipeline")
    p.add_argument("--root", default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--format", choices=["text", "json"], default="text")
    a = p.parse_args(argv)
    try:
        before = linkmod.link(a.before, root=a.root, language=a.language)
        after = linkmod.link(a.after, root=a.root, language=a.language)
    except linkmod.LinkError as e:
        sys.stderr.write(f"[component] diff: a pipeline does not link:\n")
        for d in e.diagnostics:
            sys.stderr.write(f"  {d}\n")
        return 1
    d = linkmod.diff_plans(before, after)
    if a.format == "json":
        print(json.dumps(d, indent=2))
        return 0
    def _fmt(items):
        return ", ".join(items) if items else "(none)"
    print(f"[component] stages added: {_fmt(d['stages_added'])}")
    print(f"[component] stages removed: {_fmt(d['stages_removed'])}")
    print(f"[component] edges added: {_fmt([f'{a}.{b}->{c}.{e}' for a, b, c, e in d['edges_added']])}")
    print(f"[component] edges removed: {_fmt([f'{a}.{b}->{c}.{e}' for a, b, c, e in d['edges_removed']])}")
    for r, ch in d["resources_changed"].items():
        print(f"[component] {r}: {ch['before']} -> {ch['after']} ({ch['delta']:+d})")
    rd = d["ring_depth"]
    print(f"[component] ring_depth: {rd['before']} -> {rd['after']} ({rd['delta']:+d})")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    sub = argv[0]
    table = {"new": cmd_new, "list": cmd_list, "check": cmd_check, "link": cmd_link,
             "plan": cmd_plan, "diff": cmd_diff}
    if sub not in table:
        sys.stderr.write(f"[component] error: unknown subcommand {sub!r}; "
                         f"use one of {list(table)}\n")
        return 2
    return table[sub](argv[1:])
