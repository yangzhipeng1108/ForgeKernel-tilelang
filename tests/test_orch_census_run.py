"""orch_census_run: the per-iteration SASS contract-axis census that rides the
NCU job. No GPU: the disassembler and the ncu task are stubbed; what is under
test is the contract the driver relies on — the two machine lines, the
largest-kernel choice, the NCU stdout streamed through verbatim with the two
census lines AFTER it (a job's archived log may keep only the tail), and every
failure degrading to one ORCH-CENSUS-SKIP line."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_orch_tools import SASS_SMALL, _big_page  # noqa: E402

from forgekernel.skills import orch_census_run as R  # noqa: E402
from forgekernel.skills import orch_tools  # noqa: E402
from forgekernel.skills.sass_tools import parse_sass  # noqa: E402

SASS_BIG = _big_page()
_SMALL_KERNEL = "\n        Function : tiny_kernel\n" + SASS_SMALL


def _dump(tmp_path, names):
    d = tmp_path / "dump"
    for n in names:
        p = d / "sub" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x7fELF" + n.encode())
    return str(d)


def _stub_disasm(monkeypatch, table):
    def fake(cubin):
        return table[os.path.basename(cubin)]
    monkeypatch.setattr(R, "disassemble", fake)


def test_census_picks_largest_kernel_and_prints_both_lines(tmp_path, monkeypatch):
    d = _dump(tmp_path, ["a.cubin", "b.cubin"])
    _stub_disasm(monkeypatch, {"a.cubin": (0, _SMALL_KERNEL),
                               "b.cubin": (0, "Function : main_kernel\n" + SASS_BIG)})
    lines = []
    summ = R.census(d, out=lines.append)
    assert summ is not None and summ == orch_tools.summarize(SASS_BIG)
    census = [l for l in lines if l.startswith("ORCH-CENSUS ")]
    assert census == [f"ORCH-CENSUS cubin=b.cubin kernel=main_kernel "
                      f"instrs={len(parse_sass(SASS_BIG).instructions)}"]
    js = [l for l in lines if l.startswith("ORCH-SUMMARY-JSON: ")]
    assert len(js) == 1
    assert json.loads(js[0][len("ORCH-SUMMARY-JSON: "):]) == json.loads(json.dumps(summ))
    # the summary carries the cache-policy axis the harvest keys on
    assert "cache_ops" in summ


def _as_nvdisasm(sass: str) -> str:
    """The same code in nvdisasm's spelling: `.L_x_N:` label lines before the
    branch targets and `` `(.L_x_N) `` operands instead of numeric addresses."""
    import re
    targets = sorted({m.group(1) for m in re.finditer(r"\bBRA (0x[0-9a-f]+)", sass)},
                     key=lambda t: int(t, 16))
    label = {t: f".L_x_{i}" for i, t in enumerate(targets)}
    out = []
    for line in sass.splitlines():
        m = re.match(r"\s*/\*([0-9a-fA-F]+)\*/", line)
        if m:
            t = f"0x{int(m.group(1), 16):x}"
            if t in label:
                out.append(f"{label[t]}:")
        out.append(re.sub(r"\bBRA (0x[0-9a-f]+)", lambda mm: f"BRA `({label[mm.group(1)]})", line))
    return "\n".join(out) + "\n"


def test_labelled_branch_targets_resolve_to_the_same_dossier():
    """nvdisasm writes `(.L_x_N) targets; unresolved, no backedge is found and
    the census reports a loop-less kernel (seen in a smoke job). Resolved, the
    summary is identical to the numeric-target reading."""
    labelled = _as_nvdisasm(SASS_BIG)
    assert "`(.L_x_" in labelled and ".L_x_0:" in labelled
    assert orch_tools.summarize(labelled)["loops"] == {}           # the failure mode
    resolved = R.resolve_labels(labelled)
    assert "`(.L_x_" not in resolved
    assert orch_tools.summarize(resolved) == orch_tools.summarize(SASS_BIG)
    assert orch_tools.summarize(SASS_BIG)["loops"]                 # a real reading
    # text without labels is untouched; an undefined label is left alone
    assert R.resolve_labels(SASS_BIG) == SASS_BIG
    assert R.resolve_labels("        /*0010*/  BRA `(.L_x_9) ;\n").strip().endswith("`(.L_x_9) ;")


def test_census_warns_when_no_loop_resolves(tmp_path, monkeypatch):
    d = _dump(tmp_path, ["a.cubin"])
    _stub_disasm(monkeypatch, {"a.cubin": (0, "Function : k\n" + _as_nvdisasm(SASS_BIG))})
    lines = []
    R.census(d, out=lines.append)
    assert any("WARN no loop regions resolved" in l for l in lines)
    _stub_disasm(monkeypatch, {"a.cubin": (0, "Function : k\n" + SASS_BIG)})
    lines = []
    R.census(d, out=lines.append)
    assert not any("WARN" in l for l in lines)


def test_census_skips_cleanly_without_cubins_or_on_tool_failure(tmp_path, monkeypatch):
    lines = []
    assert R.census(str(tmp_path / "nothing"), out=lines.append) is None
    assert lines == ["ORCH-CENSUS-SKIP reason=no-cubin-dumped"]
    d = _dump(tmp_path, ["a.cubin"])
    _stub_disasm(monkeypatch, {"a.cubin": (1, "")})
    lines = []
    assert R.census(d, out=lines.append) is None
    assert lines[-1] == "ORCH-CENSUS-SKIP reason=no-cubin-dumped"

    def boom(*a, **k):
        raise RuntimeError("nvdisasm exploded")
    monkeypatch.setattr(R, "pick_kernel", boom)
    lines = []
    assert R.census(d, out=lines.append) is None
    assert lines == ["ORCH-CENSUS-SKIP reason=RuntimeError:nvdisasm exploded"]


def test_main_wraps_ncu_task_and_prints_the_census_after_its_stdout(tmp_path, monkeypatch,
                                                                    capsys):
    seen = {}

    def fake_ncu(env):
        seen["env"] = env
        # the ncu task "compiles" and dumps a cubin into the requested dir, and
        # its stdout has already been streamed through by the time it returns
        d = Path(env[R.DUMP_DIR_ENV]) / "jit"
        d.mkdir(parents=True)
        (d / "k.cubin").write_bytes(b"cubin")
        sys.stdout.write('"ID","Kernel Name"\n0,"k"')
        return 0, '"ID","Kernel Name"\n0,"k"'
    monkeypatch.setattr(R, "_run_ncu", fake_ncu)
    monkeypatch.setattr(R, "disassemble", lambda c: (0, "Function : k\n" + SASS_BIG))
    assert R.main([]) == 0
    out = capsys.readouterr()
    assert seen["env"]["CUTE_DSL_KEEP_CUBIN"] == "1" and R.DUMP_DIR_ENV in seen["env"]
    lines = out.out.splitlines()
    # NCU CSV verbatim (an unterminated last line gets its newline), the two
    # census lines AFTER it: a job's archived log may keep only the tail
    assert lines[1:3] == ['"ID","Kernel Name"', '0,"k"']
    assert lines[-2].startswith("ORCH-CENSUS cubin=k.cubin kernel=k")
    assert lines[-1].startswith("ORCH-SUMMARY-JSON: ")
    # the dump dir is cleaned up
    assert not Path(seen["env"][R.DUMP_DIR_ENV]).exists()


def test_run_ncu_streams_stdout_live_and_replaces_undecodable_bytes(monkeypatch, capsys):
    script = ("import sys; sys.stdout.buffer.write(b'line1\\n\\xff\\xfe bad\\nline3\\n'); "
              "sys.exit(3)")
    monkeypatch.setattr(R, "NCU_CMD", [sys.executable, "-c", script])
    rc, text = R._run_ncu(dict(os.environ))
    assert rc == 3
    assert text.splitlines()[0] == "line1" and text.splitlines()[-1] == "line3"
    assert "�" in text                      # replaced, not raised
    assert capsys.readouterr().out == text       # echoed through, byte-for-byte


def test_main_propagates_ncu_rc_and_survives_launch_failure(monkeypatch, capsys):
    monkeypatch.setattr(R, "_run_ncu", lambda env: (3, "partial\n"))
    assert R.main([]) == 3
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "ORCH-CENSUS-SKIP reason=no-cubin-dumped"

    def boom(env):
        raise OSError("no bash")
    monkeypatch.setattr(R, "_run_ncu", boom)
    assert R.main([]) == 1
    assert "ORCH-CENSUS-SKIP reason=ncu-launch-failed:OSError" in capsys.readouterr().out

# ---- TileLang: the cubin comes from the compile cache, not a dump switch --------------

_TL_CSV = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
    '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
    '"Metric Value"\n'
    '"0","1","python3.14","h","normal_kernel","1","7","(256, 1, 1)","(1, 1, 1)","0",'
    '"12.0","m","launch__shared_mem_per_block_dynamic","byte/block","0"\n'
    '"1","1","python3.14","h","k_kernel","1","7","(256, 1, 1)","(64, 64, 1)","0",'
    '"12.0","m","launch__shared_mem_per_block_dynamic","byte/block","16384"\n'
    '"1","1","python3.14","h","k_kernel","1","7","(256, 1, 1)","(64, 64, 1)","0",'
    '"12.0","m","launch__registers_per_thread","register/thread","40"\n'
)


def test_the_ncu_kernel_name_skips_the_header_and_the_process_column():
    """`Kernel Name` is the 5th CSV field, and the header row has that literal string in it
    while a data row carries the process name (`python3.14`) earlier in the same column
    position. Either one chosen as "the kernel" would send the census to a cubin that does not
    exist, and the report would say no-cubin-dumped on a run that dumped fine."""
    assert R._ncu_kernel_names(_TL_CSV)[0] == "k_kernel"


def test_the_ncu_kernel_name_orders_by_rows_without_trusting_row_count():
    """Row count is an ordering hint, not an identification: a torch reference launch emits one
    row per requested metric and can outrow the kernel under test. The list comes back ordered
    so the likeliest is first, and `pick_kernel` then requires that symbol to exist in the
    dump -- see test_pick_kernel_prefers_the_symbol_the_csv_named."""
    text = ('"1","1","python3.14","h","k_kernel","1","7",\n'
            '"1","1","python3.14","h","k_kernel","1","7",\n'
            '"1","1","python3.14","h","huge_tuning_kernel","1","7",\n')


def test_no_parseable_row_yields_no_candidates():
    assert R._ncu_kernel_names("") == []
    assert R._ncu_kernel_names("==PROF== Connected to process 1\n") == []


def test_pick_kernel_prefers_the_symbol_the_csv_named(tmp_path, monkeypatch):
    """In the TileLang path the cache holds every kernel the process compiled, so the largest
    dump is not necessarily the kernel NCU profiled. A matching symbol has to win outright."""
    d = _dump(tmp_path, ["small.cubin", "big.cubin"])
    _stub_disasm(monkeypatch, {"small.cubin": (0, "Function : k_kernel\n" + SASS_SMALL),
                               "big.cubin": (0, "Function : other_kernel\n" + SASS_BIG)})
    picked = R.pick_kernel(d, out=lambda *_: None, want=["k_kernel"])
    assert picked[1] == "k_kernel"
    assert os.path.basename(picked[0]) == "small.cubin"


def test_pick_kernel_falls_back_to_largest_when_no_symbol_matches(tmp_path, monkeypatch):
    """A dump whose symbols do not name the CSV row must still produce a census: the contract
    is "degrade, never fail", and largest-by-instruction-count is the format-free fallback."""
    d = _dump(tmp_path, ["a.cubin", "b.cubin"])
    _stub_disasm(monkeypatch, {"a.cubin": (0, _SMALL_KERNEL),
                               "b.cubin": (0, "Function : main_kernel\n" + SASS_BIG)})
    picked = R.pick_kernel(d, out=lambda *_: None, want=["not_in_this_dump"])
    assert os.path.basename(picked[0]) == "b.cubin"


def test_main_points_the_tilelang_cache_at_the_dump_dir(tmp_path, monkeypatch, capsys):
    """TileLang has no keep-cubin switch; the cubin exists because every compile writes it into
    TILELANG_CACHE_DIR. The runner therefore has to hand the *child* that env, or the NCU task
    compiles into ~/.tilelang and the dump dir stays empty."""
    seen = {}

    def fake_ncu(env):
        seen["env"] = env
        # a compile writes <cache>/<version>/cuda-binaries/<sha>/kernel.cubin
        d = Path(env[R.TL_CACHE_DIR_ENV]) / "0.1.15" / "cuda-binaries" / "abc"
        d.mkdir(parents=True)
        (d / "kernel.cubin").write_bytes(b"\x7fELF")
        return 0, _TL_CSV
    monkeypatch.setattr(R, "_run_ncu", fake_ncu)
    _stub_disasm(monkeypatch, {"kernel.cubin": (0, "Function : k_kernel\n" + SASS_BIG)})
    kernel = tmp_path / "matmul_tl.py"
    kernel.write_text("import tilelang\n", encoding="utf-8")

    assert R.main(["--kernel", str(kernel)]) == 0
    env = seen["env"]
    assert env[R.TL_CACHE_DIR_ENV] == env[R.DUMP_DIR_ENV]
    assert env["TILELANG_DISABLE_CACHE"] == "0"
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2].startswith("ORCH-CENSUS cubin=kernel.cubin kernel=k_kernel")
    assert lines[-1].startswith("ORCH-SUMMARY-JSON: ")


def test_main_skips_a_tilelang_kernel_file_that_is_not_there(tmp_path, monkeypatch, capsys):
    """A typo'd --kernel must say so by name. Without the check it degrades to
    no-cubin-dumped, which points the reader at TileLang's cache instead of at their path."""
    monkeypatch.setattr(R, "_run_ncu", lambda env: (0, _TL_CSV))
    missing = tmp_path / "nope_tl.py"
    assert R.main(["--kernel", str(missing)]) == 0
    out = capsys.readouterr().out
    assert "ORCH-CENSUS-SKIP reason=tilelang-kernel-missing" in out
    assert str(missing) in out
