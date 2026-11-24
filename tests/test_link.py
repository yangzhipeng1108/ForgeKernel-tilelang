"""The simulated PipeLink linker (forgekernel.link): fuse components into one plan.

Without a GPU it checks seams, solves resources/roles against caps, generates a
syntactically valid fused source, and proves link semantics on CPU torch by chaining each
stage's golden through the same wiring."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from forgekernel import component as comp
from forgekernel import link as linkmod
from forgekernel.component_cli import main as component_main


def _stage(root: Path, name: str, *, in_shape=("m",), out_shape=("m",),
           in_dtype="float32", out_dtype="float32", resources=None, roles=None,
           golden="OUT = IN", impl="# impl\n"):
    d = root / "components" / name
    d.mkdir(parents=True)
    (d / comp.COMPONENT_TOML).write_text(
        f'[component]\nname = "{name}"\nlanguages = ["cutedsl"]\nsummary = "{name}"\n'
        + _tbl("resources", resources) + _tbl("roles", roles), encoding="utf-8")
    contract = {
        "axes": list(in_shape) if in_shape == out_shape else sorted(set(in_shape) | set(out_shape)),
        "inputs": {"x": {"shape": list(in_shape), "dtype": in_dtype}},
        "outputs": {"y": {"shape": list(out_shape), "dtype": out_dtype}},
        "semantics": f"{name}",
    }
    (d / comp.CONTRACT_JSON).write_text(json.dumps(contract), encoding="utf-8")
    slot = d / "cutedsl"
    slot.mkdir()
    (slot / "kernel.py").write_text(impl, encoding="utf-8")
    (slot / comp.VERIFY_JSON).write_text(
        json.dumps({"shape": {"m": 4}, "tolerance": 0.0, "golden": {"torch": golden}}),
        encoding="utf-8")
    return d


def _tbl(name, d):
    if not d:
        return ""
    return f"\n[{name}]\n" + "".join(f"{k} = {v}\n" for k, v in d.items())


def _stage_ports(root: Path, name: str, *, ins=("x",), outs=("y",), shape=("m",),
                 dtype="float32", golden="OUT = IN"):
    """A stage with an explicit set of input/output ports, for DAG wiring tests."""
    d = root / "components" / name
    d.mkdir(parents=True)
    (d / comp.COMPONENT_TOML).write_text(
        f'[component]\nname = "{name}"\nlanguages = ["cutedsl"]\nsummary = "{name}"\n',
        encoding="utf-8")
    contract = {
        "axes": list(shape),
        "inputs": {p: {"shape": list(shape), "dtype": dtype} for p in ins},
        "outputs": {p: {"shape": list(shape), "dtype": dtype} for p in outs},
        "semantics": name,
    }
    (d / comp.CONTRACT_JSON).write_text(json.dumps(contract), encoding="utf-8")
    slot = d / "cutedsl"
    slot.mkdir()
    (slot / "kernel.py").write_text("# impl\n", encoding="utf-8")
    (slot / comp.VERIFY_JSON).write_text(
        json.dumps({"shape": {"m": 4}, "tolerance": 0.0, "golden": {"torch": golden}}),
        encoding="utf-8")
    return d


# --- seam checks ---------------------------------------------------------------------

def test_two_matching_stages_link(tmp_path):
    _stage(tmp_path, "load")
    _stage(tmp_path, "gemm")
    plan = linkmod.link(["load", "gemm"], root=tmp_path)
    assert plan.stages == ["load", "gemm"]
    assert plan.edges == [("load", "y", "gemm", "x")]
    assert plan.language == "cutedsl"


def test_a_dtype_mismatch_is_refused_with_L_PROTO(tmp_path):
    _stage(tmp_path, "load", out_dtype="float32")
    _stage(tmp_path, "gemm", in_dtype="float16")
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["load", "gemm"], root=tmp_path)
    assert any(d.code == "L-PROTO" for d in ei.value.diagnostics)


def test_a_shape_mismatch_is_refused_with_L_PROTO(tmp_path):
    _stage(tmp_path, "load", out_shape=("m", "n"))
    _stage(tmp_path, "gemm", in_shape=("m",))
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["load", "gemm"], root=tmp_path)
    assert any(d.code == "L-PROTO" for d in ei.value.diagnostics)


def test_renamed_axes_still_link(tmp_path):
    _stage(tmp_path, "load", out_shape=("rows",))
    _stage(tmp_path, "gemm", in_shape=("rows",))
    assert linkmod.link(["load", "gemm"], root=tmp_path).stages == ["load", "gemm"]


def test_an_unknown_stage_is_refused_with_L_STAGE(tmp_path):
    _stage(tmp_path, "load")
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["load", "ghost"], root=tmp_path)
    assert any(d.code == "L-STAGE" for d in ei.value.diagnostics)


def test_an_empty_pipeline_is_refused_with_L_ORDER(tmp_path):
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link([], root=tmp_path)
    assert any(d.code == "L-ORDER" for d in ei.value.diagnostics)


# --- resource / role solving ---------------------------------------------------------

def test_resources_are_summed_across_stages(tmp_path):
    _stage(tmp_path, "load", resources={"smem_bytes": 32768, "barrier_slots": 2})
    _stage(tmp_path, "gemm", resources={"smem_bytes": 65536, "barrier_slots": 4})
    plan = linkmod.link(["load", "gemm"], root=tmp_path)
    assert plan.resources["smem_bytes"] == 32768 + 65536
    assert plan.resources["barrier_slots"] == 6


def test_a_resource_over_cap_is_refused_with_L_RES(tmp_path):
    _stage(tmp_path, "load", resources={"smem_bytes": 200000})
    _stage(tmp_path, "gemm", resources={"smem_bytes": 200000})
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["load", "gemm"], root=tmp_path)
    diag = [d for d in ei.value.diagnostics if d.code == "L-RES"]
    assert diag
    # the diagnostic names the stages live at the peak, so the fix has a lead
    assert "load" in diag[0].message and "gemm" in diag[0].message


def test_warps_over_the_block_are_refused_with_L_ROLE(tmp_path):
    _stage(tmp_path, "load", roles={"warps": 16})
    _stage(tmp_path, "gemm", roles={"warps": 16})
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["load", "gemm"], root=tmp_path, caps={"warps": 24})
    assert any(d.code == "L-ROLE" for d in ei.value.diagnostics)


def test_ring_depth_is_solved_from_the_stage_floors(tmp_path):
    _stage(tmp_path, "load", roles={"ring_min": 2})
    _stage(tmp_path, "gemm", roles={"ring_min": 3, "ring_max": 5})
    plan = linkmod.link(["load", "gemm"], root=tmp_path)
    assert plan.ring_depth == 3


def test_a_ring_outside_the_window_is_refused(tmp_path):
    _stage(tmp_path, "load", roles={"ring_min": 2, "ring_max": 3})
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["load"], root=tmp_path, ring=5)
    assert any(d.code == "L-RING" for d in ei.value.diagnostics)


# --- fused source + manifest ---------------------------------------------------------

def test_render_fused_embeds_each_stage_and_parses(tmp_path):
    _stage(tmp_path, "load", impl="def load(x):\n    return x * 2\n")
    _stage(tmp_path, "gemm", impl="def gemm(x):\n    return x + 1\n")
    plan = linkmod.link(["load", "gemm"], root=tmp_path)
    src = linkmod.render_fused(plan, root=tmp_path)
    assert "def load(x):" in src and "def gemm(x):" in src
    assert "load.y -> gemm.x" in src
    ast.parse(src)  # the skeleton is valid Python


def test_render_fused_refuses_shadowed_top_level_names(tmp_path):
    # both stages define `step`; inlining would let the second silently win
    _stage(tmp_path, "a", impl="def step(x):\n    return x\n")
    _stage(tmp_path, "b", impl="def step(x):\n    return x + 1\n")
    plan = linkmod.link(["a", "b"], root=tmp_path)
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.render_fused(plan, root=tmp_path)
    assert any(d.code == "L-PROTO" and "step" in d.message for d in ei.value.diagnostics)


def test_render_fused_allows_distinct_names(tmp_path):
    _stage(tmp_path, "a", impl="def step_a(x):\n    return x\n")
    _stage(tmp_path, "b", impl="def step_b(x):\n    return x + 1\n")
    plan = linkmod.link(["a", "b"], root=tmp_path)
    linkmod.render_fused(plan, root=tmp_path)   # no collision -> no raise


def test_manifest_is_json_serializable(tmp_path):
    _stage(tmp_path, "load")
    _stage(tmp_path, "gemm")
    plan = linkmod.link(["load", "gemm"], root=tmp_path)
    m = plan.manifest()
    assert json.loads(json.dumps(m))["stages"] == ["load", "gemm"]


# --- semantic parity on CPU torch ----------------------------------------------------

def test_chained_golden_names_each_step(tmp_path):
    _stage(tmp_path, "load", golden="OUT = IN")
    _stage(tmp_path, "scale", golden="OUT = IN * 3")
    plan = linkmod.link(["load", "scale"], root=tmp_path)
    chain = linkmod.chained_golden(plan, root=tmp_path)
    assert [s["stage"] for s in chain["steps"]] == ["load", "scale"]
    assert chain["steps"][0]["output"] == "OUT_0"
    assert chain["steps"][1]["input"] == "OUT_0"


def test_parity_proves_the_composition_on_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    _stage(tmp_path, "scale", golden="OUT = IN * 2")
    _stage(tmp_path, "shift", golden="OUT = IN + 1")
    plan = linkmod.link(["scale", "shift"], root=tmp_path)
    x = torch.arange(4, dtype=torch.float32)
    readings = linkmod.run_parity(plan, lambda v: v * 2 + 1, root=tmp_path, feeds={"IN": x})
    assert readings[-1]["parity"] == "ok"


def test_parity_reports_a_mismatch(tmp_path):
    torch = pytest.importorskip("torch")
    _stage(tmp_path, "scale", golden="OUT = IN * 2")
    plan = linkmod.link(["scale"], root=tmp_path)
    x = torch.arange(4, dtype=torch.float32)
    readings = linkmod.run_parity(plan, lambda v: v * 3, root=tmp_path, feeds={"IN": x})
    assert readings[-1]["parity"] == "mismatch"


def test_parity_records_each_stage_output_shape(tmp_path):
    torch = pytest.importorskip("torch")
    _stage(tmp_path, "scale", golden="OUT = IN * 2")
    _stage(tmp_path, "shift", golden="OUT = IN + 1")
    plan = linkmod.link(["scale", "shift"], root=tmp_path)
    x = torch.arange(4, dtype=torch.float32)
    readings = linkmod.run_parity(plan, lambda v: v * 2 + 1, root=tmp_path, feeds={"IN": x})
    per_stage = [r for r in readings if "stage" in r]
    assert per_stage[0]["shape"] == [4]
    assert readings[-1]["final_stage"] == "shift"


def test_parity_is_incomplete_without_a_golden(tmp_path):
    pytest.importorskip("torch")
    _stage(tmp_path, "scale", golden="")            # no golden.torch
    plan = linkmod.link(["scale"], root=tmp_path)
    chain = linkmod.chained_golden(plan, root=tmp_path)
    assert chain["missing_golden"] == ["scale"]
    readings = linkmod.run_parity(plan, lambda v: v, root=tmp_path, feeds={"IN": 1})
    assert readings[0]["parity"] == "incomplete"
    assert readings[0]["missing_golden"] == ["scale"]


# --- CLI -----------------------------------------------------------------------------

def test_cli_link_reports_and_emits(tmp_path, capsys):
    _stage(tmp_path, "load", impl="def load(x):\n    return x\n")
    _stage(tmp_path, "gemm", impl="def gemm(x):\n    return x\n")
    out = tmp_path / "build"
    rc = component_main(["link", "load", "gemm", "--root", str(tmp_path), "--emit", str(out)])
    assert rc == 0
    assert (out / "fused_kernel.py").is_file()
    assert (out / "link-plan.json").is_file()
    plan = json.loads((out / "link-plan.json").read_text())
    assert plan["stages"] == ["load", "gemm"]


def test_cli_link_exits_1_on_a_refused_link(tmp_path, capsys):
    _stage(tmp_path, "load", out_dtype="float32")
    _stage(tmp_path, "gemm", in_dtype="float16")
    rc = component_main(["link", "load", "gemm", "--root", str(tmp_path)])
    assert rc == 1
    assert "L-PROTO" in capsys.readouterr().err


def test_cli_link_emit_reports_a_fusion_collision(tmp_path, capsys):
    _stage(tmp_path, "a", impl="def step(x):\n    return x\n")
    _stage(tmp_path, "b", impl="def step(x):\n    return x\n")
    out = tmp_path / "build"
    rc = component_main(["link", "a", "b", "--root", str(tmp_path), "--emit", str(out)])
    assert rc == 1
    assert "L-PROTO" in capsys.readouterr().err
    assert not (out / "fused_kernel.py").is_file()   # nothing half-written


# --- DAG wiring (multi-input, fan-out) ----------------------------------------------

def test_a_diamond_dag_links_in_topo_order(tmp_path):
    # split -> (branch_a, branch_b) -> join, with join taking two wired inputs
    _stage_ports(tmp_path, "split", ins=("x",), outs=("y",))
    _stage_ports(tmp_path, "branch_a", ins=("x",), outs=("y",))
    _stage_ports(tmp_path, "branch_b", ins=("x",), outs=("y",))
    _stage_ports(tmp_path, "join", ins=("p", "q"), outs=("y",))
    edges = [("split", "y", "branch_a", "x"),
             ("split", "y", "branch_b", "x"),
             ("branch_a", "y", "join", "p"),
             ("branch_b", "y", "join", "q")]
    plan = linkmod.link(["split", "branch_a", "branch_b", "join"], root=tmp_path, edges=edges)
    # topo order: split before its consumers, join last
    assert plan.stages.index("split") < plan.stages.index("branch_a")
    assert plan.stages[-1] == "join"
    assert set(plan.edges) == set(edges)


def test_a_multi_input_stage_requires_all_inputs_wired(tmp_path):
    _stage_ports(tmp_path, "a", ins=("x",), outs=("y",))
    _stage_ports(tmp_path, "join", ins=("p", "q"), outs=("y",))
    # only p is wired; q is left dangling on a non-source stage
    edges = [("a", "y", "join", "p")]
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["a", "join"], root=tmp_path, edges=edges)
    assert any(d.code == "L-WIRE" and "join.q" in d.message for d in ei.value.diagnostics)


def test_fanout_from_one_output_is_allowed(tmp_path):
    _stage_ports(tmp_path, "src", ins=("x",), outs=("y",))
    _stage_ports(tmp_path, "c1", ins=("x",), outs=("y",))
    _stage_ports(tmp_path, "c2", ins=("x",), outs=("y",))
    edges = [("src", "y", "c1", "x"), ("src", "y", "c2", "x")]
    plan = linkmod.link(["src", "c1", "c2"], root=tmp_path, edges=edges)
    assert len([e for e in plan.edges if e[0] == "src"]) == 2


def test_a_cycle_is_refused_with_L_ORDER(tmp_path):
    _stage_ports(tmp_path, "a", ins=("x",), outs=("y",))
    _stage_ports(tmp_path, "b", ins=("x",), outs=("y",))
    edges = [("a", "y", "b", "x"), ("b", "y", "a", "x")]
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["a", "b"], root=tmp_path, edges=edges)
    assert any(d.code == "L-ORDER" and "cycle" in d.message for d in ei.value.diagnostics)


def test_an_edge_naming_an_unknown_stage_is_L_WIRE(tmp_path):
    _stage_ports(tmp_path, "a", ins=("x",), outs=("y",))
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["a"], root=tmp_path, edges=[("a", "y", "ghost", "x")])
    assert any(d.code == "L-WIRE" for d in ei.value.diagnostics)


def test_an_edge_naming_an_unknown_port_is_L_WIRE(tmp_path):
    _stage_ports(tmp_path, "a", ins=("x",), outs=("y",))
    _stage_ports(tmp_path, "b", ins=("x",), outs=("y",))
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["a", "b"], root=tmp_path, edges=[("a", "nope", "b", "x")])
    assert any(d.code == "L-WIRE" and "nope" in d.message for d in ei.value.diagnostics)


def test_repeated_diagnostics_are_coalesced(tmp_path):
    _stage_ports(tmp_path, "s", ins=("x",), outs=("ya", "yb"))
    _stage_ports(tmp_path, "t", ins=("p", "q"), outs=("y",), dtype="float16")
    edges = [("s", "ya", "t", "p"), ("s", "yb", "t", "q")]
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["s", "t"], root=tmp_path, edges=edges)
    proto = [d for d in ei.value.diagnostics if d.code == "L-PROTO"]
    assert proto and "[2 sites]" in proto[0].message


def test_dag_parity_wires_two_inputs(tmp_path):
    torch = pytest.importorskip("torch")
    _stage_ports(tmp_path, "a", ins=("x",), outs=("y",), golden="OUT = IN + 1")
    _stage_ports(tmp_path, "b", ins=("p", "q"), outs=("y",),
                 golden="OUT = IN_p * IN_q")
    # b needs both p and q; feed q from a second source stage
    _stage_ports(tmp_path, "src", ins=("x",), outs=("y",), golden="OUT = IN")
    edges = [("a", "y", "b", "p"), ("src", "y", "b", "q")]
    plan = linkmod.link(["src", "a", "b"], root=tmp_path, edges=edges)
    x = torch.arange(4, dtype=torch.float32)
    # a(IN)=IN+1, src(IN)=IN, b(p,q)=p*q  -> (IN+1)*IN
    readings = linkmod.run_parity(plan, lambda v: (v + 1) * v, root=tmp_path,
                                  feeds={"IN": x}, output_stage="b")
    assert readings[-1]["parity"] == "ok"


def test_cli_link_accepts_explicit_edges(tmp_path, capsys):
    _stage_ports(tmp_path, "a", ins=("x",), outs=("y",))
    _stage_ports(tmp_path, "b", ins=("x",), outs=("y",))
    rc = component_main(["link", "a", "b", "--root", str(tmp_path),
                         "--edge", "a.y=b.x"])
    assert rc == 0
    assert "stages: a -> b" in capsys.readouterr().out


# --- peak residency vs naive sum -----------------------------------------------------

def test_a_chain_of_resources_peaks_at_two_not_all(tmp_path):
    # four 64 KB stages in a chain: 256 KB summed, but only neighbours overlap -> 128 KB
    for nm in ("s1", "s2", "s3", "s4"):
        _stage(tmp_path, nm, resources={"smem_bytes": 65536})
    plan = linkmod.link(["s1", "s2", "s3", "s4"], root=tmp_path,
                        caps={"smem_bytes": 200 * 1024})
    assert plan.resources["smem_bytes"] == 128 * 1024       # peak, not 256 KB


def test_a_full_fanout_peaks_at_the_sum(tmp_path):
    # one source fanning into three sinks: source live until all sinks start, so the peak
    # is the source plus all sinks resident together
    _stage_ports(tmp_path, "src", ins=("x",), outs=("y",))
    for nm in ("c1", "c2", "c3"):
        _stage_ports(tmp_path, nm, ins=("x",), outs=("y",))
    # give each a smem claim by editing resources via a fresh stage helper call
    for nm, kb in (("src", 10), ("c1", 20), ("c2", 20), ("c3", 20)):
        d = tmp_path / "components" / nm
        (d / comp.COMPONENT_TOML).write_text(
            f'[component]\nname = "{nm}"\nlanguages = ["cutedsl"]\nsummary = "{nm}"\n'
            f'[resources]\nsmem_bytes = {kb * 1024}\n', encoding="utf-8")
    edges = [("src", "y", "c1", "x"), ("src", "y", "c2", "x"), ("src", "y", "c3", "x")]
    plan = linkmod.link(["src", "c1", "c2", "c3"], root=tmp_path, edges=edges)
    # src(10) live while c1,c2,c3 start; at the last sink-start all four are live
    assert plan.resources["smem_bytes"] == (10 + 20 + 20 + 20) * 1024


# --- ring depth from producer/consumer rates ----------------------------------------

def test_ring_depth_from_a_faster_producer(tmp_path):
    _stage(tmp_path, "prod", roles={"producer_rate": 8})
    _stage(tmp_path, "cons", roles={"consumer_rate": 2})
    plan = linkmod.link(["prod", "cons"], root=tmp_path)
    assert plan.ring_depth == 4          # ceil(8 / 2), so the producer never stalls


def test_ring_depth_is_one_when_rates_match(tmp_path):
    _stage(tmp_path, "a", roles={"producer_rate": 2, "consumer_rate": 2})
    _stage(tmp_path, "b", roles={"consumer_rate": 2})
    assert linkmod.link(["a", "b"], root=tmp_path).ring_depth == 1


def test_an_explicit_ring_floor_beats_the_rate_need(tmp_path):
    _stage(tmp_path, "prod", roles={"producer_rate": 8, "ring_min": 6})
    _stage(tmp_path, "cons", roles={"consumer_rate": 2})
    assert linkmod.link(["prod", "cons"], root=tmp_path).ring_depth == 6


# --- bridge suggestion on a mismatched seam -----------------------------------------

def test_a_mismatched_seam_names_a_bridge_that_would_fit(tmp_path):
    _stage(tmp_path, "prod", out_dtype="float32")
    _stage(tmp_path, "cons", in_dtype="float16")
    # a bridge that takes float32 and gives float16
    b = tmp_path / "components" / "cast"
    b.mkdir(parents=True)
    (b / comp.COMPONENT_TOML).write_text(
        '[component]\nname = "cast"\nlanguages = ["cutedsl"]\nsummary = "cast"\n',
        encoding="utf-8")
    (b / comp.CONTRACT_JSON).write_text(json.dumps({
        "axes": ["m"],
        "inputs": {"x": {"shape": ["m"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["m"], "dtype": "float16"}},
        "semantics": "cast"}), encoding="utf-8")
    slot = b / "cutedsl"; slot.mkdir()
    (slot / "kernel.py").write_text("# impl\n", encoding="utf-8")
    (slot / comp.VERIFY_JSON).write_text(json.dumps(
        {"shape": {"m": 4}, "tolerance": 0.0, "golden": {"torch": "OUT = IN"}}),
        encoding="utf-8")
    with pytest.raises(linkmod.LinkError) as ei:
        linkmod.link(["prod", "cons"], root=tmp_path)
    proto = [d for d in ei.value.diagnostics if d.code == "L-PROTO"][0]
    assert "insert cast" in proto.message


# --- summarize / diff / graph -------------------------------------------------------

def test_summarize_explains_peak_and_ring(tmp_path):
    _stage(tmp_path, "prod", resources={"smem_bytes": 1024}, roles={"producer_rate": 4})
    _stage(tmp_path, "cons", roles={"consumer_rate": 2})
    plan = linkmod.link(["prod", "cons"], root=tmp_path)
    text = "\n".join(linkmod.summarize(plan))
    assert "smem_bytes: 1024" in text
    assert "rate mismatch" in text and "ring_depth: 2" in text


def test_diff_reports_a_swapped_stage(tmp_path):
    _stage(tmp_path, "a", resources={"smem_bytes": 1000})
    _stage(tmp_path, "b", resources={"smem_bytes": 2000})
    _stage(tmp_path, "c", resources={"smem_bytes": 4000})
    before = linkmod.link(["a", "b"], root=tmp_path)
    after = linkmod.link(["a", "c"], root=tmp_path)
    d = linkmod.diff_plans(before, after)
    assert d["stages_added"] == ["c"] and d["stages_removed"] == ["b"]
    # chain peak: before a+b overlap at 3000; after a+c overlap at 5000
    assert d["resources_changed"]["smem_bytes"]["delta"] == 2000


def test_render_graph_ascii_and_mermaid(tmp_path):
    _stage(tmp_path, "a")
    _stage(tmp_path, "b")
    plan = linkmod.link(["a", "b"], root=tmp_path)
    ascii_g = linkmod.render_graph(plan, "ascii")
    assert "a --[y->x]--> b" in ascii_g
    mermaid = linkmod.render_graph(plan, "mermaid")
    assert mermaid.startswith("flowchart LR") and "n_a" in mermaid


def test_expected_ms_budget_is_summed_into_reasons(tmp_path):
    _stage(tmp_path, "a", roles={"expected_ms": 0.5})
    _stage(tmp_path, "b", roles={"expected_ms": 0.3})
    plan = linkmod.link(["a", "b"], root=tmp_path)
    assert abs(plan.reasons["expected_ms"] - 0.8) < 1e-9
    assert "expected_ms: 0.8" in "\n".join(linkmod.summarize(plan))


# --- CLI: plan / diff / json --------------------------------------------------------

def test_cli_plan_prints_summary_and_graph(tmp_path, capsys):
    _stage(tmp_path, "a")
    _stage(tmp_path, "b")
    rc = component_main(["plan", "a", "b", "--root", str(tmp_path), "--graph", "ascii"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "stages: a -> b" in out and "--[y->x]-->" in out


def test_cli_link_json_is_machine_readable(tmp_path, capsys):
    _stage(tmp_path, "a")
    _stage(tmp_path, "b")
    rc = component_main(["link", "a", "b", "--root", str(tmp_path), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["stages"] == ["a", "b"]


def test_cli_link_json_reports_refusal(tmp_path, capsys):
    _stage(tmp_path, "a", out_dtype="float32")
    _stage(tmp_path, "b", in_dtype="float16")
    rc = component_main(["link", "a", "b", "--root", str(tmp_path), "--format", "json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["diagnostics"][0]["code"] == "L-PROTO"


def test_cli_diff_reports_the_swap(tmp_path, capsys):
    _stage(tmp_path, "a", resources={"smem_bytes": 1000})
    _stage(tmp_path, "b", resources={"smem_bytes": 2000})
    _stage(tmp_path, "c", resources={"smem_bytes": 4000})
    rc = component_main(["diff", "a", "b", "--then", "a", "c", "--root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "stages added: c" in out and "stages removed: b" in out

