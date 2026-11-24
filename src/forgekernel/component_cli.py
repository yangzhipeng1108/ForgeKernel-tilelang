"""forge component — create, list and check kernel components (forgekernel.component).

    forge component new <name> [--language cutedsl,cuda,tilelang] [--summary "..."]
    forge component list [--root DIR]
    forge component check [<name> | --all] [--root DIR]

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
    a = p.parse_args(argv)
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


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    sub = argv[0]
    table = {"new": cmd_new, "list": cmd_list, "check": cmd_check}
    if sub not in table:
        sys.stderr.write(f"[component] error: unknown subcommand {sub!r}; "
                         f"use one of {list(table)}\n")
        return 2
    return table[sub](argv[1:])
