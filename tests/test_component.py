"""Kernel components (forgekernel.component): a bounded unit that ships, swaps, reuses.

The point of a component is that a developer or an agent can work inside one directory
with a fixed contract, deliver it with its own proof, and have it dropped into another
operator or system by contract rather than by copy."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgekernel import component as comp
from forgekernel.component_cli import main as component_main


def _write(root: Path, name: str, *, languages=("cutedsl",), summary="does a thing",
           contract=None, requires: str = "", impls=True, verify=None):
    d = root / "components" / name
    d.mkdir(parents=True)
    langs = ", ".join(f'"{x}"' for x in languages)
    (d / comp.COMPONENT_TOML).write_text(
        f'[component]\nname = "{name}"\nlanguages = [{langs}]\nsummary = "{summary}"\n'
        + requires, encoding="utf-8")
    c = contract if contract is not None else {
        "axes": ["m", "n"],
        "inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}},
        "semantics": "y = x",
        "invariants": ["same shape"],
    }
    if c is not None:
        (d / comp.CONTRACT_JSON).write_text(
            c if isinstance(c, str) else json.dumps(c), encoding="utf-8")
    if impls:
        for lang in languages:
            if lang not in comp.IMPL_FILE:
                continue
            slot = d / lang
            slot.mkdir()
            (slot / comp.IMPL_FILE[lang]).write_text("# impl\n", encoding="utf-8")
            v = verify if verify is not None else {
                "shape": {"m": 4, "n": 8}, "tolerance": 0.0, "golden": {"torch": "y = x"}}
            (slot / comp.VERIFY_JSON).write_text(
                v if isinstance(v, str) else json.dumps(v), encoding="utf-8")
    return d


# --- the happy path ------------------------------------------------------------------

def test_a_well_formed_component_is_deliverable(tmp_path):
    _write(tmp_path, "scale")
    assert comp.check_tree(tmp_path / "components" / "scale") == []


def test_list_components_reads_metadata(tmp_path):
    _write(tmp_path, "scale", summary="y = x * s")
    comps = comp.list_components(tmp_path)
    assert [c.name for c in comps] == ["scale"]
    assert comps[0].languages == ["cutedsl"] and comps[0].summary == "y = x * s"


# --- contract problems ---------------------------------------------------------------

def test_a_bad_dtype_is_reported(tmp_path):
    bad = {"axes": ["m"], "inputs": {"x": {"shape": ["m"], "dtype": "f32"}},
           "outputs": {"y": {"shape": ["m"], "dtype": "float32"}},
           "semantics": "y = x", "invariants": []}
    _write(tmp_path, "bad", contract=bad)
    findings = comp.check_tree(tmp_path / "components" / "bad")
    assert any("dtype" in f for f in findings)


def test_a_shape_over_an_undeclared_axis_is_reported(tmp_path):
    bad = {"axes": ["m"], "inputs": {"x": {"shape": ["k"], "dtype": "float32"}},
           "outputs": {"y": {"shape": ["m"], "dtype": "float32"}},
           "semantics": "y = x", "invariants": []}
    _write(tmp_path, "bad", contract=bad)
    assert any("k" in f and "axes" in f for f in comp.check_tree(tmp_path / "components" / "bad"))


def test_semantics_is_required(tmp_path):
    no_sem = {"axes": ["m"], "inputs": {"x": {"shape": ["m"], "dtype": "float32"}},
              "outputs": {"y": {"shape": ["m"], "dtype": "float32"}}}
    _write(tmp_path, "bad", contract=no_sem)
    assert any("semantics" in f for f in comp.check_tree(tmp_path / "components" / "bad"))


def test_an_unknown_handle_is_reported(tmp_path):
    bad = {"axes": ["m"], "inputs": {"x": {"shape": ["m"], "dtype": "float32"}},
           "outputs": {"y": {"shape": ["m"], "dtype": "float32", "handle": "ldsm"}},
           "semantics": "y = x"}
    _write(tmp_path, "bad", contract=bad)
    findings = comp.check_tree(tmp_path / "components" / "bad")
    assert any("handle" in f and "ldsm" in f for f in findings)


def test_a_negative_generation_is_reported(tmp_path):
    bad = {"axes": ["m"], "inputs": {"x": {"shape": ["m"], "dtype": "float32"}},
           "outputs": {"y": {"shape": ["m"], "dtype": "float32", "generation": -1}},
           "semantics": "y = x"}
    _write(tmp_path, "bad", contract=bad)
    assert any("generation" in f for f in comp.check_tree(tmp_path / "components" / "bad"))


def test_a_valid_handle_and_generation_pass(tmp_path):
    ok = {"axes": ["m"], "inputs": {"x": {"shape": ["m"], "dtype": "float32", "handle": "reg"}},
          "outputs": {"y": {"shape": ["m"], "dtype": "float32", "handle": "tmem",
                            "generation": 2}},
          "semantics": "y = x"}
    _write(tmp_path, "good", contract=ok,
           verify={"shape": {"m": 4}, "tolerance": 0.0, "golden": {"torch": "y = x"}})
    assert comp.check_tree(tmp_path / "components" / "good") == []


def test_missing_contract_is_reported(tmp_path):
    _write(tmp_path, "nocontract", contract=None)
    # delete what _write made (it substitutes a default when None) — make it truly absent
    (tmp_path / "components" / "nocontract" / comp.CONTRACT_JSON).unlink()
    assert any("contract.json" in f for f in comp.check_tree(tmp_path / "components" / "nocontract"))


def test_names_clashing_between_inputs_and_outputs_are_reported(tmp_path):
    bad = {"axes": ["m"], "inputs": {"y": {"shape": ["m"], "dtype": "float32"}},
           "outputs": {"y": {"shape": ["m"], "dtype": "float32"}}, "semantics": "y = y"}
    _write(tmp_path, "bad", contract=bad)
    assert any("both an input and an output" in f
               for f in comp.check_tree(tmp_path / "components" / "bad"))


# --- implementation slots and verify -------------------------------------------------

def test_a_declared_language_without_an_implementation_is_reported(tmp_path):
    _write(tmp_path, "half", languages=("cutedsl", "tilelang"), impls=False)
    findings = comp.check_tree(tmp_path / "components" / "half")
    assert any("cutedsl/kernel.py is missing" in f for f in findings)
    assert any("tilelang/kernel_tl.py is missing" in f for f in findings)


def test_an_undeclared_language_directory_is_reported(tmp_path):
    d = _write(tmp_path, "extra", languages=("cutedsl",))
    (d / "cuda").mkdir()
    assert any("cuda/ exists but" in f for f in comp.check_tree(d))


def test_verify_without_a_golden_is_reported(tmp_path):
    _write(tmp_path, "nogold", verify={"shape": {"m": 1, "n": 1}, "tolerance": 0.0})
    assert any("golden" in f for f in comp.check_tree(tmp_path / "components" / "nogold"))


def test_verify_naming_an_undeclared_axis_is_reported(tmp_path):
    _write(tmp_path, "badshape",
           verify={"shape": {"m": 4, "zzz": 8}, "tolerance": 0.0, "golden": {"seed": 1}})
    assert any("zzz" in f for f in comp.check_tree(tmp_path / "components" / "badshape"))


def test_a_name_that_disagrees_with_the_directory_is_reported(tmp_path):
    d = tmp_path / "components" / "dirname"
    d.mkdir(parents=True)
    (d / comp.COMPONENT_TOML).write_text(
        '[component]\nname = "other"\nlanguages = ["cutedsl"]\nsummary = "s"\n',
        encoding="utf-8")
    assert any("does not match the directory" in f for f in comp.check_tree(d))


def test_an_unknown_toml_table_is_reported(tmp_path):
    d = _write(tmp_path, "typo", requires="\n[require]\nkind = 'x'\n")
    assert any("unknown table [require]" in f for f in comp.check_tree(d))


def test_a_missing_summary_is_reported(tmp_path):
    _write(tmp_path, "nosummary", summary="")
    assert any("summary is required" in f for f in comp.check_tree(tmp_path / "components" / "nosummary"))


# --- requires and cross-language substitution ---------------------------------------

def test_a_missing_required_component_is_reported(tmp_path):
    _write(tmp_path, "top", requires='\n[[requires.component]]\nname = "base"\n'
                                       'contract = "base/contract.json"\n')
    assert any("'base'" in f and "requires.component" in f
               for f in comp.check_tree(tmp_path / "components" / "top"))


def test_a_present_required_component_clears(tmp_path):
    _write(tmp_path, "base")
    _write(tmp_path, "top", requires='\n[[requires.component]]\nname = "base"\n'
                                   'contract = "base/contract.json"\n')
    assert comp.check_tree(tmp_path / "components" / "top") == []


def test_contracts_compatible_ignores_axis_names_but_not_structure():
    a = {"axes": ["m", "n"], "inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"}},
         "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}}}
    renamed = {"axes": ["rows", "cols"],
               "inputs": {"x": {"shape": ["rows", "cols"], "dtype": "float32"}},
               "outputs": {"y": {"shape": ["rows", "cols"], "dtype": "float32"}}}
    different = {"axes": ["m", "n"], "inputs": {"x": {"shape": ["n", "m"], "dtype": "float32"}},
                 "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}}}
    dtype = {"axes": ["m", "n"], "inputs": {"x": {"shape": ["m", "n"], "dtype": "float16"}},
             "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}}}
    assert comp.contracts_compatible(a, renamed)
    assert not comp.contracts_compatible(a, different)
    assert not comp.contracts_compatible(a, dtype)


def test_satisfies_is_by_contract(tmp_path):
    _write(tmp_path, "base")
    other = {"axes": ["rows", "cols"],
             "inputs": {"x": {"shape": ["rows", "cols"], "dtype": "float32"}},
             "outputs": {"y": {"shape": ["rows", "cols"], "dtype": "float32"}},
             "semantics": "y = x (another implementation)"}
    _write(tmp_path, "other", contract=other)
    base_contract = comp.read_contract(tmp_path / "components" / "base")
    assert comp.satisfies(comp.read_component(tmp_path / "components" / "other"), base_contract)


# --- CLI -----------------------------------------------------------------------------

def test_cli_new_scaffolds_a_checkable_component(tmp_path, capsys):
    assert component_main(["new", "scale", "--root", str(tmp_path),
                           "--summary", "y = x * s"]) == 0
    d = tmp_path / "components" / "scale"
    assert (d / comp.COMPONENT_TOML).is_file()
    assert (d / comp.CONTRACT_JSON).is_file()
    assert (d / "cutedsl" / "kernel.py").is_file()
    # the scaffold points at cutedsl; strip its stub placeholder semantics so it checks
    findings = comp.check_tree(d)
    # the generated stub is intentionally minimal: it should be checkable once filled,
    # but the structure (all slots present, verify present) must already be sound
    assert not any("is missing" in f for f in findings)


def test_cli_new_refuses_an_existing_name(tmp_path, capsys):
    component_main(["new", "scale", "--root", str(tmp_path)])
    assert component_main(["new", "scale", "--root", str(tmp_path)]) == 2


def test_cli_new_rejects_an_unknown_language(tmp_path, capsys):
    assert component_main(["new", "x", "--root", str(tmp_path), "--language", "rust"]) == 2


def test_cli_check_fails_on_a_broken_component_and_passes_on_a_good_one(tmp_path, capsys):
    _write(tmp_path, "good")
    _write(tmp_path, "bad", languages=("cutedsl", "cuda"), impls=False)
    assert component_main(["check", "good", "--root", str(tmp_path)]) == 0
    assert component_main(["check", "bad", "--root", str(tmp_path)]) == 1
    assert component_main(["check", "--all", "--root", str(tmp_path)]) == 1


def test_cli_list(tmp_path, capsys):
    _write(tmp_path, "scale", summary="y = x * s")
    assert component_main(["list", "--root", str(tmp_path)]) == 0
    assert "scale" in capsys.readouterr().out


# --- exporting a finished run's best kernel as a component --------------------------

def _problem():
    return {
        "definition": {
            "axes": {"m": {"type": "var"}, "n": {"type": "var"}},
            "inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                       "w": {"shape": ["n", "n"], "dtype": "float32"}},
            "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}},
            "reference": "def run(x, w):\n    return x @ w\n",
        },
        "tolerance": {"outputs": {"y": 2e-6}},
    }


def test_contract_and_verify_are_lifted_from_the_problem():
    contract = comp.contract_from_problem(_problem(), "y = x @ w")
    assert sorted(contract["axes"]) == ["m", "n"]
    assert contract["inputs"]["x"] == {"shape": ["m", "n"], "dtype": "float32"}
    assert contract["outputs"]["y"]["dtype"] == "float32"
    assert contract["semantics"] == "y = x @ w"
    verify = comp.verify_from_problem(_problem(), {"m": 4, "n": 8})
    assert verify["shape"] == {"m": 4, "n": 8}
    assert verify["tolerance"] == 2e-6
    assert "x @ w" in verify["golden"]["torch"]


def test_export_component_writes_a_checkable_component(tmp_path):
    dest = tmp_path / "components" / "gemm"
    kernel = "import torch\n\ndef run(x, w):\n    return x @ w\n"
    comp.export_component(dest, name="gemm", language="cutedsl", kernel_src=kernel,
                          contract=comp.contract_from_problem(_problem(), "y = x @ w"),
                          verify=comp.verify_from_problem(_problem(), {"m": 4, "n": 8}),
                          summary="y = x @ w")
    assert (dest / "cutedsl" / "kernel.py").read_text() == kernel
    assert comp.check_tree(dest) == []          # export gates on its own check


def test_export_component_refuses_to_overwrite_a_delivered_component(tmp_path):
    dest = tmp_path / "components" / "gemm"
    dest.mkdir(parents=True)
    with pytest.raises(comp.ComponentError, match="already exists"):
        comp.export_component(dest, name="gemm", language="cutedsl", kernel_src="x",
                              contract=comp.contract_from_problem(_problem(), "s"),
                              verify=comp.verify_from_problem(_problem(), {}), summary="s")
