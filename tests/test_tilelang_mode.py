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
    for arch in ("sm90", "sm100"):
        assert loop_cli._kb_arch_problem(TL_KB, arch) is None, arch
        assert lint_kb_dir(TL_KB / arch) == [], arch


def test_the_tilelang_kb_opens_only_where_it_has_a_tree():
    """exit 14 is a hard gate, not a warning: a run pointed at a tree with no index.md
    reads nothing and files its recipes where nobody indexes them."""
    problem = loop_cli._kb_arch_problem(TL_KB, "sm120")
    assert problem and "sm120" in problem
    assert "sm90" in problem and "sm100" in problem      # names what does exist


def test_the_tilelang_kb_is_not_prefilled_from_the_dsl_or_cuda_trees():
    """hw_kb/cutedsl/README.md: the tree is demand-driven and must not be pre-filled.
    A TileLang entry copied from a cute.* reading is exactly the leak this tree exists
    to prevent. So: no entry names cute.* / CUTLASS DSL symbols, and every entry beyond
    the index carries the "Measured on" block that says which TileLang version and arch
    its readings came from -- an entry without it is a hypothesis, not evidence."""
    dsl_spelling = re.compile(r"\bcute\.[A-Za-z_]|cutlass\.cute|CUTE_DSL|@cute|"
                              r"cutlass\._mlir|CuTe DSL|tcgen05\.mma\.|wgmma\.mma_async")
    for arch in ("sm90", "sm100"):
        entries = sorted((TL_KB / arch).glob("*.md"))
        assert "index.md" in [p.name for p in entries], arch
        for entry in entries:
            text = entry.read_text(encoding="utf-8")
            assert not dsl_spelling.search(text), (entry.name, dsl_spelling.search(text))
            if entry.name != "index.md":
                assert "## Measured on" in text, entry.name
