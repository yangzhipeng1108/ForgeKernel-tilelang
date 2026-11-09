"""The TileLang mode (loop language tilelang, a kernel_tl.py student), kept apart from the
CuTe DSL and CUDA modes: which prompts / KB / provenance each mode reads
(loop/lang_mode.py), the language-vs-kernel-file guard across three suffixes, the KB tree
opening rule, the prompt overlay's fallback, and the DSL / CUDA modes staying exactly as
they were.

Phase 0 of porting\\ForgeKernel接入TileLang改造计划.md: what can be verified without a GPU
and without a working TileLang install. The instrument chain (timeline probe, cubin census,
job-env toolchain check) arrives with P0-a and P1 and is tested separately.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from forgekernel.loop import cli as loop_cli
from forgekernel.loop import lang_mode

REPO = Path(__file__).resolve().parents[1]
TL_KB = REPO / "hw_kb" / "tilelang"
TL_PROMPTS = REPO / "meta" / "tilelang"


# ---------------------------------------------------------------- modes
def test_a_language_belongs_to_one_mode_and_tilelang_is_its_own():
    assert lang_mode.mode_of("tilelang") == lang_mode.MODE_TILELANG
    assert lang_mode.mode_of("cutedsl") == lang_mode.MODE_CUTEDSL
    assert lang_mode.mode_of("cuda-sol") == lang_mode.MODE_CUDA
    # an unset language is still the DSL mode, byte for byte
    assert lang_mode.mode_of(None) == lang_mode.MODE_CUTEDSL
    assert lang_mode.mode_of("") == lang_mode.MODE_CUTEDSL


def test_the_legacy_labels_keep_their_old_mode_including_inline_ptx():
    """Pre-D1 this was a blacklist ("not cuda => cutedsl"). The explicit map has to
    reproduce it for every label the leaderboard / recon stages still emit, inline-ptx
    included: it is a CuTe DSL kernel that drops to PTX inside, and test_cuda_mode.py
    pins it to the DSL mode."""
    assert lang_mode.mode_of("triton") == lang_mode.MODE_CUTEDSL
    assert lang_mode.mode_of("inline-ptx") == lang_mode.MODE_CUTEDSL
    assert lang_mode.mode_of("cuda-cpp") == lang_mode.MODE_CUDA
    assert lang_mode.mode_of("cutlass-cpp") == lang_mode.MODE_CUDA


def test_an_unknown_language_is_refused_not_silently_filed_under_the_dsl_mode():
    """The silent fallback is the failure this map exists to remove: a run that reads
    the DSL prompts while the student writes something else shows it only as a ceiling
    nobody can explain."""
    with pytest.raises(ValueError, match="unknown language"):
        lang_mode.mode_of("metal")
    with pytest.raises(ValueError, match="LANGUAGE_MODE"):
        lang_mode.mode_of("CUDA")


def test_each_mode_reads_its_own_kb_root_and_provenance_subdir(tmp_path):
    assert lang_mode.kb_root_name("cutedsl") == "hw_kb/cutedsl"
    assert lang_mode.kb_root_name("cuda-sol") == "hw_kb/cuda"
    assert lang_mode.kb_root_name("tilelang") == "hw_kb/tilelang"
    assert lang_mode.kb_repo_root(tmp_path / "hw_kb" / "tilelang") == tmp_path
    assert lang_mode.provenance_dir(tmp_path, "tilelang") == (
        tmp_path / "docs" / "kb-provenance" / "tilelang")
    assert lang_mode.provenance_dir(tmp_path, "cutedsl") == tmp_path / "docs" / "kb-provenance"


def test_provenance_ownership_is_decided_by_the_subdir_not_by_not_being_cuda():
    """Pre-D3 ``owns_provenance`` was ``in_cuda if is_cuda else not in_cuda``, which
    hands every non-CUDA subdir to the DSL mode. Three modes need the owner looked up
    from PROVENANCE_SUBDIR, or a TileLang recipe is claimed by the DSL tree."""
    tl = "docs/kb-provenance/tilelang/gpu_probe_ladder.md"
    cu = "docs/kb-provenance/cuda/gpu_probe_ladder.md"
    dsl = "docs/kb-provenance/gpu_probe_ladder.md"
    gaps = "docs/kb-provenance/gaps/op.md"
    assert lang_mode.owns_provenance(tl, "tilelang")
    assert not lang_mode.owns_provenance(tl, "cutedsl")
    assert not lang_mode.owns_provenance(tl, "cuda-sol")
    assert lang_mode.owns_provenance(cu, "cuda-sol")
    assert not lang_mode.owns_provenance(cu, "tilelang")
    # the shared root and the per-op gaps archive stay the DSL mode's
    assert lang_mode.owns_provenance(dsl, "cutedsl")
    assert lang_mode.owns_provenance(gaps, "cutedsl")
    assert not lang_mode.owns_provenance(dsl, "tilelang")
    assert not lang_mode.owns_provenance(gaps, "cuda-sol")


def test_the_kernel_file_picks_the_language_and_three_suffixes_stay_apart():
    assert lang_mode.language_for_kernel_file("kernel.py") is None
    assert lang_mode.language_for_kernel_file("kernel.cu") == "cuda-sol"
    assert lang_mode.language_for_kernel_file("kernel_tl.py") == "tilelang"
    assert lang_mode.is_tilelang_kernel_file("kernel_tl.py")
    assert not lang_mode.is_tilelang_kernel_file("kernel.py")
    assert not lang_mode.is_cuda_kernel_file("kernel_tl.py")
    # a TileLang file must not be mistaken for the DSL default by its .py suffix either
    assert loop_cli._resolve_language(None, {}, "kernel_tl.py") == "tilelang"
    assert loop_cli._resolve_language(None, {}, "kernel.py") == "cutedsl"
    assert loop_cli._resolve_language(None, {}, "kernel.cu") == "cuda-sol"
    assert loop_cli._resolve_language(None, {}) == "cutedsl"


def test_a_kernel_file_the_language_contradicts_is_named():
    ok = [("cutedsl", "kernel.py"), ("cuda-sol", "kernel.cu"), ("tilelang", "kernel_tl.py")]
    for language, kernel in ok:
        assert lang_mode.language_problem(language, kernel) is None
    # each mode over each other mode's suffix, in both directions
    bad = [("cuda-sol", "kernel.py"), ("cuda-sol", "kernel_tl.py"),
           ("cutedsl", "kernel.cu"), ("cutedsl", "kernel_tl.py"),
           ("tilelang", "kernel.py"), ("tilelang", "kernel.cu")]
    for language, kernel in bad:
        problem = lang_mode.language_problem(language, kernel)
        assert problem, (language, kernel)
        assert "kernel_" in problem or "kernel." in problem


def test_the_tilelang_mismatch_names_the_right_kernel_file():
    assert "kernel_tl.py" in lang_mode.language_problem("tilelang", "kernel.py")
    assert "TileLang" in lang_mode.language_problem("cutedsl", "kernel_tl.py")
    assert "tilelang" in lang_mode.language_problem("cutedsl", "kernel_tl.py")


def test_the_modes_stay_in_step_with_the_two_task_and_cli_tables():
    """Two different language tuples ship in the repo -- task.py's (task.toml [task]
    language) and loop/cli.py's (--language choices). A mode that lands in one and not
    the other is a run that scaffolds with a language the loop refuses, or the reverse."""
    from forgekernel import task as task_mod
    from forgekernel import taskcard

    assert "tilelang" in task_mod.LANGUAGES
    assert task_mod.LOOP_LANGUAGE["tilelang"] == "tilelang"
    assert "tilelang" in loop_cli.LANGUAGES
    assert taskcard.KERNEL_FILE["tilelang"] == "kernel_tl.py"
    # every task language is a language the loop accepts
    for language in task_mod.LANGUAGES:
        assert task_mod.LOOP_LANGUAGE[language] in loop_cli.LANGUAGES, language


# ---------------------------------------------------------------- prompt overlay
def test_the_tilelang_mode_reads_its_prompt_overlay_and_falls_back_to_the_shared_prompt(tmp_path):
    (tmp_path / "tilelang").mkdir()
    for n in ("student_prompt.md", "diagnose_prompt.md"):
        (tmp_path / n).write_text("shared")
    (tmp_path / "tilelang" / "student_prompt.md").write_text("tilelang")
    assert lang_mode.prompt_path(tmp_path, "student_prompt.md", "tilelang") == (
        tmp_path / "tilelang" / "student_prompt.md")
    assert lang_mode.prompt_path(tmp_path, "student_prompt.md", "cutedsl") == (
        tmp_path / "student_prompt.md")
    # no overlay for this prompt yet: the shared one, not a missing file
    assert lang_mode.prompt_path(tmp_path, "diagnose_prompt.md", "tilelang") == (
        tmp_path / "diagnose_prompt.md")


def test_every_shipped_tilelang_overlay_keeps_the_shared_prompts_bindings_and_sections():
    """Same contract as the CUDA overlay: the driver renders the overlay with the shared
    prompt's bindings and parses the same output, so a dropped {TOKEN} or a renamed
    section is a silent difference."""
    overlays = sorted(TL_PROMPTS.glob("*.md"))
    if not overlays:
        pytest.skip("no meta/tilelang/ overlay yet: every prompt falls back to the shared one")
    for ov in overlays:
        shared = REPO / "meta" / ov.name
        assert shared.is_file(), f"meta/tilelang/{ov.name} overlays no shared prompt"
        a = shared.read_text(encoding="utf-8")
        b = ov.read_text(encoding="utf-8")
        assert _tokens(b) == _tokens(a), (ov.name, _tokens(a) ^ _tokens(b))
        assert _sections(b) == _sections(a), ov.name


def test_the_tilelang_prompts_teach_tilelang_not_the_dsl_or_the_cuda_card():
    dsl = re.compile(r"cutlass\.cute|cute\.compile|CUTE_DSL|make_tiled_copy|const_expr|"
                     r"load_cuda_entry|PYBIND11_MODULE")
    overlays = sorted(TL_PROMPTS.glob("*.md"))
    for ov in overlays:
        hits = [ln for ln in ov.read_text(encoding="utf-8").splitlines() if dsl.search(ln)]
        assert not hits, (ov.name, hits[:3])
    student = TL_PROMPTS / "student_prompt.md"
    if student.is_file():          # it falls back to the shared prompt until written
        text = student.read_text(encoding="utf-8")
        assert "kernel_tl.py" in text or "{KERNEL_FILE}" in text
        assert "TileLang" in text


def _tokens(text: str) -> set:
    return set(re.findall(r"\{[A-Z][A-Z_]*\}", text))


def _sections(text: str) -> list:
    return [ln for ln in text.splitlines() if ln.startswith("## ")]


# ---------------------------------------------------------------- KB tree
def test_the_tilelang_kb_is_its_own_lint_clean_tree():
    from forgekernel.loop.kb_lint import lint_kb_dir

    assert (TL_KB / "TEMPLATE.md").is_file() and (TL_KB / "README.md").is_file()
    for arch in ("sm90", "sm100", "sm120"):
        assert loop_cli._kb_arch_problem(TL_KB, arch) is None, arch
        assert lint_kb_dir(TL_KB / arch) == [], arch


def test_the_tilelang_kb_opens_only_where_it_has_a_tree():
    """exit 14 is a hard gate, not a warning: a run pointed at a tree with no index.md
    reads nothing and files its recipes where nobody indexes them."""
    problem = loop_cli._kb_arch_problem(TL_KB, "sm150")
    assert problem and "sm150" in problem
    assert "sm90" in problem and "sm100" in problem      # names what does exist


def test_the_tilelang_kb_is_not_prefilled_from_the_dsl_or_cuda_trees():
    """hw_kb/cutedsl/README.md: the tree is demand-driven and must not be pre-filled.
    A TileLang entry copied from a cute.* reading is exactly the leak this tree exists
    to prevent. So: no entry names cute.* / CUTLASS DSL symbols, and every entry beyond
    the index carries the "Measured on" block that says which TileLang version and arch
    its readings came from -- an entry without it is a hypothesis, not evidence."""
    dsl_spelling = re.compile(r"\bcute\.[A-Za-z_]|cutlass\.cute|CUTE_DSL|@cute|"
                              r"cutlass\._mlir|CuTe DSL|tcgen05\.mma\.|wgmma\.mma_async")
    trees = sorted(p.name for p in TL_KB.iterdir() if p.is_dir())
    assert "sm90" in trees and "sm100" in trees and "sm120" in trees
    for arch in trees:
        entries = sorted((TL_KB / arch).glob("*.md"))
        assert "index.md" in [p.name for p in entries], arch
        for entry in entries:
            text = entry.read_text(encoding="utf-8")
            assert not dsl_spelling.search(text), (entry.name, dsl_spelling.search(text))
            if entry.name != "index.md":
                assert "## Measured on" in text, entry.name


# ---------------------------------------------------------------- kernel stub
def _task(language: str):
    """Only the three fields render_kernel_stub reads; a full Task needs a directory,
    a reference, two agent specs and a budget, none of which the stub looks at."""
    import types

    from forgekernel.task import Shape

    return types.SimpleNamespace(
        name="rmsnorm", language=language, arch="sm90",
        shapes=(Shape(name="default", axes={"M": 4096, "N": 1024}),))


def _check():
    from forgekernel.judge import Check

    return Check(definition={"name": "rmsnorm",
                             "inputs": {"x": {"dtype": "F32", "shape": [4096, 1024]}},
                             "outputs": {"y": {"dtype": "F32", "shape": [4096, 1024]}}},
                 judge_src="")


def test_every_stub_language_renders_a_kernel_its_own_language_can_parse():
    """A stub is what the student starts from, so it has to be *of the target language*.
    The failure this replaces was a `def run(...): raise NotImplementedError` handed to a
    TileLang student: valid Python, so nothing complained, and it taught CuTe DSL in a
    file the mode reads as TileLang.

    The TileLang stub is checked the way the language checks it: it must parse, and it
    must contain the current spelling (`@T.prim_func` + `T.Kernel(..., threads=)`) and
    none of the dead one (`T.alloc_buffer` / `T.launch_thread` / `T.const`), because a
    symbolic extent is what fails to lower on this card."""
    from forgekernel import taskcard

    for language in taskcard.STUB_RENDERER:
        text = taskcard.render_kernel_stub(_task(language), _check())
        assert text.strip(), language
        if language == "cuda":
            continue                                   # C++, not parsed as Python
        compile(text, f"<{language} stub>", "exec")     # a parse error here is the bug

    dsl = taskcard.render_kernel_stub(_task("cutedsl"), _check())
    assert "NotImplementedError" in dsl and "tilelang" not in dsl

    from forgekernel.judge import Check

    named = Check(definition={"name": "rmsnorm",
                              "axes": {"M": {"type": "var"},
                                       "N": {"type": "const", "value": 1024}},
                              "inputs": {"x": {"dtype": "F32", "shape": ["M", "N"]}},
                              "outputs": {"y": {"dtype": "F32", "shape": ["M", "N"]}}},
                  judge_src="")
    tl = taskcard.render_kernel_stub(_task("tilelang"), named)
    assert "@T.prim_func" in tl and "T.Kernel(" in tl and "threads=" in tl
    # The prose must be allowed to name the dead API (that is the warning); only the
    # code after the closing docstring is the skeleton.
    body = tl.split(chr(34) * 3)[-1]
    for dead in ("T.alloc_buffer", "T.launch_thread", "T.const(", "T.symbolic",
                 "NotImplementedError", "@tilelang.autotune", "T.Buffer("):
        assert dead not in body, dead
    # the extents are the concrete ints of the card's first shape, not a placeholder
    assert "M = 4096" in tl and "N = 1024" in tl    # the first shape, concretely
    # the staging fragments must not shadow the parameters (T.copy(x[...], x) is a no-op)
    assert "T.copy(x[bx * BLK, 0], _tile_in)" in tl


def test_a_rank_the_stub_cannot_tile_is_refused_instead_of_guessed():
    """A rank-3 tensor needs a tile shape the stub cannot know; guessing one teaches a
    sweep the student did not choose, so it refuses and says to write the file by hand."""
    from forgekernel import taskcard
    from forgekernel.judge import Check

    task = _task("tilelang")
    check = Check(definition={"name": "three", "axes": {"A": {"type": "const", "value": 4}},
                              "inputs": {"x": {"dtype": "F32", "shape": [4, 8, 16]}},
                              "outputs": {"y": {"dtype": "F32", "shape": [4, 8, 16]}}},
                  judge_src="")
    with pytest.raises(ValueError, match="rank-3"):
        taskcard.render_kernel_stub(task, check)


def test_the_two_stub_tables_are_the_same_set_and_the_loop_agrees():
    """taskcard and leaderboard each write a kernel file (forge new / forge new
    --leaderboard). Their renderer tables drifting apart is how one of them starts
    writing a skeleton the other refuses."""
    from forgekernel import leaderboard as lbmod
    from forgekernel import taskcard

    assert set(taskcard.STUB_RENDERER) == set(lbmod.STUB_RENDERER)
    assert set(taskcard.STUB_RENDERER) == taskcard._STUB_LANGUAGES
    # every language with a stub is a registered mode, so the loop can drive what the
    # scaffold just wrote
    # taskcard keys on the task.toml language ("cuda"), the loop on the loop one
    # ("cuda-sol"); both must resolve to a registered mode.
    loop_language = {"cutedsl": "cutedsl", "cuda": "cuda-sol", "tilelang": "tilelang"}
    for language in taskcard.STUB_RENDERER:
        assert lang_mode.mode_of(loop_language[language])


def test_the_leaderboard_card_writes_a_tilelang_loop_toml_that_agrees_with_its_stub(tmp_path):
    """write_card used to refuse a _tl.py outright. It now writes one -- and the failure
    it has to avoid is the *other* one: a loop.toml.example that says cutedsl beside a
    TileLang stub, which every later run inherits."""
    from forgekernel import leaderboard as lbmod

    problem = lbmod.Problem(
        suite=lbmod.SUITE_SOL,
        definition={"name": "rmsnorm", "axes": {"M": {"type": "var"},
                                                "N": {"type": "const", "value": 1024}},
                    "inputs": {"x": {"dtype": "F32", "shape": ["M", "N"]}},
                    "outputs": {"y": {"dtype": "F32", "shape": ["M", "N"]}}},
        workload={"uuid": "w", "axes": {"M": 4096, "N": 1024}},
        reference_impl={"kind": "definition", "name": "torch"},
        source={}, blobs={},
    )
    op = tmp_path / "op"
    op.mkdir()
    (op / "eval").mkdir()
    # write_card copies leaderboard.py / timing.py / precision.py from the card template,
    # so hand it the real one rather than a empty fixture directory.
    template = REPO / "src" / "forgekernel" / "card_template" / "eval"
    lbmod.write_card(op, problem, template, kernel_file="kernel_tl.py")

    stub = (op / "kernel_tl.py").read_text(encoding="utf-8")
    compile(stub, "<tilelang leaderboard stub>", "exec")
    assert "@T.prim_func" in stub and "T.Kernel(" in stub
    loop_toml = (op / "loop.toml.example").read_text(encoding="utf-8")
    assert 'language    = "tilelang"' in loop_toml
    assert "cutedsl" not in loop_toml


def test_the_leaderboard_stub_refuses_a_kernel_file_no_language_claims(tmp_path):
    """The remaining refusal has to stay: a kernel file whose suffix names no mode must
    not be handed the CuTe DSL skeleton, and nothing may be written before the refusal."""
    from forgekernel import leaderboard as lbmod

    with pytest.raises(ValueError, match="no leaderboard card stub"):
        lbmod.write_card(tmp_path / "op", None, tmp_path / "eval", kernel_file="kernel.metal")
    assert not (tmp_path / "op").exists() or not any((tmp_path / "op").iterdir())


# ---------------------------------------------------------------- source dirs
def _tilelang_site(site: Path, version: str = "0.1.15") -> Path:
    """A site directory laid out like the ``tilelang`` wheel: one plain package
    (``tilelang/``) with no ``dsl_packages`` indirection, and the version in its
    dist-info. The CuTe DSL family's ``_fake_install`` (test_dsl_source.py) is the
    other shape, and the point of this pair is that the probe tells them apart."""
    pkg = site / "tilelang"
    (pkg / "language").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "language" / "__init__.py").write_text("")
    meta = site / f"tilelang-{version}.dist-info"
    meta.mkdir()
    (meta / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: tilelang\nVersion: {version}\n")
    return pkg


def _probe_env(site: Path) -> dict:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(site)
    return env


def _resolve_tl(cfg, site, *, submit_env=None, language="tilelang"):
    import sys

    from forgekernel.loop import dsl_source as ds

    return ds.resolve(cfg, language=language, submit_env=submit_env or {},
                      python=sys.executable, probe_env=_probe_env(site))


def test_an_empty_key_defaults_a_tilelang_run_to_the_installed_package(tmp_path):
    """The CuTe DSL default walks ``nvidia_cutlass_dsl/dsl_packages`` out of RECORD;
    TileLang ships one plain package, so its import root IS the directory the student
    reads. Both go through the same resolve() and must not be confused for each other:
    the directory handed over, and the name the version is reported under."""
    site = tmp_path / "site"
    pkg = _tilelang_site(site, "0.1.15")
    r = _resolve_tl({"loop": {"dsl_source_dirs": []}}, site,
                    submit_env={"DSL_PIN": "0.1.15"})
    assert r.defaulted and r.origin == "default"
    assert r.dirs == (pkg.resolve(),)
    assert r.version == "0.1.15"
    assert r.distribution == "tilelang"          # not the CuTe DSL's name
    assert "tilelang 0.1.15" in r.notes[0] and "matches [env] DSL_PIN" in r.notes[0]
    assert r.describe().startswith("default ") and "tilelang 0.1.15" in r.describe()


def test_a_tilelang_pin_mismatch_refuses_the_default(tmp_path):
    site = tmp_path / "site"
    _tilelang_site(site, "0.1.15")
    r = _resolve_tl({"loop": {}}, site, submit_env={"DSL_PIN": "0.1.16"})
    assert not r.defaulted and r.origin == "none"
    assert r.version == "0.1.15" and "0.1.16" in r.notes[0]


def test_a_declared_tilelang_source_dir_wins_over_the_default(tmp_path):
    from forgekernel.loop import dsl_source as ds

    site = tmp_path / "site"
    _tilelang_site(site)
    mine = tmp_path / "tilelang-checkout"
    mine.mkdir()
    r = _resolve_tl({"loop": {"dsl_source_dirs": [str(mine)]}}, site)
    assert r.origin == ds.ORIGIN_EXPLICIT and r.dirs == (mine.resolve(),)


def test_a_tilelang_install_overlapping_the_reference_is_refused(tmp_path):
    site = tmp_path / "site"
    pkg = _tilelang_site(site)
    r = _resolve_tl({"loop": {"ref_source_paths": [str(pkg)]}}, site)
    assert not r.defaulted
    assert "overlaps the declared reference source" in r.notes[0]


def test_cuda_sol_still_has_no_dsl_default_to_offer(tmp_path):
    """A CUDA student reads headers named in ``[loop] cuda_source_dirs``; there is no
    Python API to default to, and inventing one (the DSL package) would be the wrong
    door. The message says language=, so it is not mistaken for 'nothing installed'."""
    site = tmp_path / "site"
    _tilelang_site(site)
    r = _resolve_tl({"loop": {}}, site, language="cuda-sol")
    assert r.origin == "none" and "no DSL source to default" in r.notes[0]


# ---------------------------------------------------------------- policy
def test_the_tilelang_notice_names_tilelang_sources_not_the_dsl_and_not_headers(tmp_path):
    from forgekernel.loop import agent_policy as ap

    src = tmp_path / "tl-src"
    src.mkdir()
    cfg = {"loop": {"tilelang_source_dirs": [str(src)]}}
    pol = ap.from_config(cfg, oracle_mode="trace", repo_root=tmp_path,
                         teacher_trees=[], language="tilelang")
    notice = pol.notice("student")
    assert "TileLang package sources" in notice
    assert "DSL package sources" not in notice
    assert "CUDA / CUTLASS headers" not in notice
    assert str(src.resolve()) in notice


def test_tilelang_source_dirs_is_its_own_key_and_falls_back_to_the_old_one(tmp_path):
    """Its own key, the way cuda_source_dirs is: one mode's package tree must not be
    handed to another mode's student. A config written before the key existed still
    reads dsl_source_dirs."""
    from forgekernel.loop import agent_policy as ap

    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    both = ap.from_config({"loop": {"dsl_source_dirs": [str(old)],
                                    "tilelang_source_dirs": [str(new)]}},
                          oracle_mode="trace", repo_root=tmp_path,
                          teacher_trees=[], language="tilelang")
    assert both.dsl_source_dirs == (new.resolve(),)
    only_old = ap.from_config({"loop": {"dsl_source_dirs": [str(old)]}},
                              oracle_mode="trace", repo_root=tmp_path,
                              teacher_trees=[], language="tilelang")
    assert only_old.dsl_source_dirs == (old.resolve(),)
    # and the DSL mode does not read the TileLang key
    dsl = ap.from_config({"loop": {"tilelang_source_dirs": [str(new)]}},
                         oracle_mode="trace", repo_root=tmp_path,
                         teacher_trees=[], language="cutedsl")
    assert dsl.dsl_source_dirs == ()


def test_an_empty_tilelang_source_key_warns_only_in_the_tilelang_mode(tmp_path):
    from forgekernel.loop import agent_policy as ap

    cfg = {"loop": {}}
    tl = ap.declaration_warnings(cfg, language="tilelang")
    assert any("tilelang_source_dirs is empty" in w for w in tl)
    cu = ap.declaration_warnings(cfg, language="cuda-sol")
    assert any("cuda_source_dirs is empty" in w for w in cu)
    assert not any("tilelang_source_dirs" in w for w in cu)
    dsl = ap.declaration_warnings(cfg, language="cutedsl")
    assert not any("tilelang_source_dirs" in w for w in dsl)


# ---------------------------------------------------------------- driver routing
def _ctx(kernel_file: str, language=None, workspace=None):
    import types

    return types.SimpleNamespace(kernel_file=kernel_file, language=language,
                                 workspace=workspace, orch_census=False)


def test_the_driver_routes_a_tilelang_run_by_its_kernel_file_alone():
    """``[loop] language`` may be unset and follow the file (lang_mode), so the
    driver's test must read the file too, exactly as _is_cuda does. A ``_tl.py``
    under the CUDA mode never gets here: language_problem refuses it first."""
    from forgekernel.loop import driver

    assert driver._is_tilelang(_ctx("kernel_tl.py"))
    assert driver._is_tilelang(_ctx("kernel_tl.py", "tilelang"))
    assert not driver._is_tilelang(_ctx("kernel.py", "cutedsl"))
    assert not driver._is_tilelang(_ctx("kernel.cu", "cuda-sol"))
    assert not driver._is_cuda(_ctx("kernel_tl.py"))


def test_the_probe_jobs_name_the_tilelang_kernel_file(tmp_path):
    """The probe's default is kernel.py; a TileLang run's file is kernel_tl.py, and a
    probe pointed at the wrong name measures nothing. Same fix as the CUDA mode's."""
    from forgekernel.loop import driver

    cmd = driver._timeline_cmd(_ctx("kernel_tl.py", "tilelang"))
    assert "--kernel kernel_tl.py" in cmd
    assert driver._timeline_cmd(_ctx("kernel.py", "cutedsl")) == driver._TIMELINE_CMD

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / driver._ORCH_CENSUS_RUNNER).write_text("", encoding="utf-8")
    census_ctx = _ctx("kernel_tl.py", "tilelang", tmp_path)
    census_ctx.orch_census = True
    assert "--kernel kernel_tl.py" in driver._ncu_mode(census_ctx)


def test_the_tilelang_markers_are_read_with_the_python_spelling(tmp_path):
    """A ``_tl.py`` is Python: ``def _seg`` + literal-string calls, the same parser as
    the CuTe DSL one. The C++ spelling (``#define FK_SEG``) is the CUDA mode's only."""
    from forgekernel.loop import driver

    src = tmp_path / "kernel_tl.py"
    src.write_text('def _seg(site, row=0):\n    return None\n\n\n'
                   'def k():\n    _seg("load.begin", 0)\n    _seg("load.end", 0)\n',
                   encoding="utf-8")
    has_def, names = driver._timeline_markers(_ctx("kernel_tl.py", "tilelang", tmp_path))
    assert has_def and names == ["load.begin", "load.end"]


def test_the_tilelang_student_note_says_the_probe_cannot_stamp_not_add_markers(tmp_path):
    """The stamped preamble imports cutlass, so asking a TileLang student to add _seg
    calls is asking them to restructure the kernel around a probe that cannot run.
    The note has to say that plainly — and it must not carry the CuTe DSL convention
    line (cutlass.const_expr and friends)."""
    from forgekernel.loop import driver

    (tmp_path / "kernel_tl.py").write_text("def k():\n    pass\n", encoding="utf-8")
    ctx = _ctx("kernel_tl.py", "tilelang", tmp_path)
    ctx.timeline_probe = "on"
    note = driver._timeline_student_note(ctx)
    assert "does not support TileLang yet" in note
    assert "do not restructure the kernel around the probe" in note
    assert "cutlass.const_expr" not in note
    assert "keep `def _seg(site, row=0): return None`" not in note

# ---------------------------------------------------------------- the KB measurement script
# scripts/measure_kb_entries.py exists so that filling the four skeletons is a command, not a
# manual reading. Two things about it have to stay true or the readings it prints are wrong
# in a way nobody would notice: the NCU keys it asks for must be the ones inner/ncu.py parses,
# and the KB directory name must not be mistaken for the CUDA target name.
def _measure_script():
    import importlib.util

    path = REPO / "scripts" / "measure_kb_entries.py"
    spec = importlib.util.spec_from_file_location("measure_kb_entries", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_kb_measurement_script_asks_for_metrics_the_parser_knows():
    """ncu prints whatever you ask for; the parser only reads KEY_METRICS. A metric that is
    requested but not parsed costs an NCU counter slot and yields nothing, and the report would
    print an empty axis without saying why."""
    from forgekernel.inner import ncu

    mod = _measure_script()
    unknown = set(mod.NCU_SHORT) - set(ncu.KEY_METRICS)
    assert not unknown, f"requested but not parsed by inner/ncu.py: {sorted(unknown)}"
    assert set(mod.NCU_SHORT.values()) <= set(ncu.KEY_METRICS.values())


def test_the_kb_measurement_script_maps_directory_names_to_real_cuda_targets():
    """`sm90` is a directory; tilelang refuses it as a target ("Target {...} is not supported").
    The map has to name the accelerated variant, and it has to cover every KB tree so a new
    tree cannot silently fall through.

    sm120 is the case that punishes a careless map: `sm_120` lowers without complaint and then
    fails inside nvcc with a warpgroup_reg_dealloc static assertion, so a tree mapped to the
    plain form would look healthy right up to the first real compile."""
    mod = _measure_script()
    trees = {p.name for p in TL_KB.iterdir() if p.is_dir()}
    assert trees == set(mod.KB_ARCH_TO_TARGET)
    assert mod.KB_ARCH_TO_TARGET["sm90"] == "sm_90a"
    assert mod.KB_ARCH_TO_TARGET["sm100"] == "sm_100a"
    assert mod.KB_ARCH_TO_TARGET["sm120"] == "sm_120a"
    for tree, target in mod.KB_ARCH_TO_TARGET.items():
        assert target.startswith("sm_") and target[len("sm_"):][-1].isalpha(), (tree, target)


def test_every_kb_skeleton_still_carries_an_unfilled_measured_on_block():
    """Writing a number into `## Measured on` without running the script is the failure mode this
    guards: a skeleton whose Certification line was filled by hand is a claim, not a reading.
    Filled entries move out of this assertion by being renamed to a non-skeleton file."""
    skeletons = ["software_pipeline_stage_budget", "tiled_copy_scope_and_swizzle",
                 "gemm_operand_layout_and_policy", "gpu_probe_ladder"]
    for name in skeletons:
        text = (TL_KB / "sm90" / f"{name}.md").read_text(encoding="utf-8")
        assert "## Measured on" in text, name
        assert "[to be measured]" in text, name
        assert "- arch: `[to be measured]`" in text, name
