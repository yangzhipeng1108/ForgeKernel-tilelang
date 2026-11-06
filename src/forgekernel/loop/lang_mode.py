"""Language modes: the CuTe DSL, CUDA and TileLang modes, kept apart.

A loop ``language`` (loop.toml ``[loop].language`` / ``forge loop --language``)
used to steer one thing only -- the ``{LANGUAGE}`` token in the student and
distill prompts. Everything else the loop hands a run (the prompts, the hw_kb
tree, the timeline instrumentation, the census and pricing harnesses) was
written for a Python CuTe DSL ``kernel.py``. A CUDA C++ student (``kernel.cu``,
language ``cuda-sol``) then read DSL recipes, was told to keep a Python
``def _seg`` marker, and bought timeline / pricing jobs that could not read or
outran its kernel.

This module is the one place that says which MODE a language belongs to and
what each mode reads, so the modes stay independent:

- the **cutedsl mode** reads ``meta/<prompt>``, ``hw_kb/cutedsl/<arch>/``,
  ``docs/kb-provenance/``. Nothing here changes a byte of its path.
- the **cuda mode** (``CUDA_LANGUAGES``) reads its own prompt set
  (``meta/cuda/<prompt>`` wherever one exists, the shared language-neutral
  prompt otherwise), its own KB root (``hw_kb/cuda/<arch>/``) and files its
  provenance under ``docs/kb-provenance/cuda/``. Its tools key on the kernel
  file's suffix (``.cu``), which ``language_problem`` keeps in step with the
  language.
- the **tilelang mode** (``TILELANG_LANGUAGES``) is the same shape again:
  ``meta/tilelang/<prompt>``, ``hw_kb/tilelang/<arch>/`` and
  ``docs/kb-provenance/tilelang/``, keyed on the ``_tl.py`` suffix.
"""
from __future__ import annotations

from pathlib import Path

MODE_CUTEDSL = "cutedsl"
MODE_CUDA = "cuda"
MODE_TILELANG = "tilelang"

# The languages whose student writes a CUDA C++ kernel file the card builds with
# nvcc. cuda-sol is the leaderboard cuda_cpp solution (one .cu with its own
# PYBIND11_MODULE binding, built the way SOL-ExecBench builds it); it is the only
# CUDA language with a card harness behind it.
CUDA_LANGUAGES = frozenset({"cuda-sol"})
CUDA_KERNEL_SUFFIXES = (".cu",)

# The languages whose student writes a TileLang kernel. The file is still Python,
# so it cannot be told apart from a CuTe DSL kernel by its interpreter: the
# suffix is what keeps the two apart.
TILELANG_LANGUAGES = frozenset({"tilelang"})
TILELANG_KERNEL_SUFFIXES = ("_tl.py",)

# Every language a mode claims, in one place. A language that is in none of these
# is refused by ``mode_of`` rather than silently defaulting to the DSL mode: the
# quiet fallback is how a run ends up reading the wrong prompts and only shows it
# as a ceiling it cannot explain.
LANGUAGE_MODE = {
    "cutedsl": MODE_CUTEDSL,
    "cuda-sol": MODE_CUDA,
    "tilelang": MODE_TILELANG,
}
# Labels the leaderboard / recon stages still emit that name a mode but are not
# loop languages a student writes in. Registered so they keep their old mode --
# including ``inline-ptx``: it is a CuTe DSL kernel that drops to PTX inside, and
# ``tests/test_cuda_mode.py`` pins it to the DSL mode.
LEGACY_LANGUAGE = {
    "triton": MODE_CUTEDSL,
    "inline-ptx": MODE_CUTEDSL,
    "cuda-cpp": MODE_CUDA,
    "cutlass-cpp": MODE_CUDA,
}

KB_ROOT = {MODE_CUTEDSL: "hw_kb/cutedsl", MODE_CUDA: "hw_kb/cuda",
           MODE_TILELANG: "hw_kb/tilelang"}
# Under the shared docs/kb-provenance/ root, so every "startswith docs/kb-provenance/"
# listing (harvest leftovers, publish) still covers it, while an entry name the
# KBs share (gpu_probe_ladder, ...) keeps one provenance file per KB.
PROVENANCE_SUBDIR = {MODE_CUTEDSL: "", MODE_CUDA: "cuda", MODE_TILELANG: "tilelang"}
PROMPT_OVERLAY_DIR = {MODE_CUTEDSL: None, MODE_CUDA: "cuda", MODE_TILELANG: "tilelang"}

DEFAULT_CUDA_LANGUAGE = "cuda-sol"
DEFAULT_TILELANG_LANGUAGE = "tilelang"


def mode_of(language: str | None) -> str:
    """The mode a loop language belongs to; an unset language is the DSL mode.

    A language this module does not know raises instead of falling back: the
    fallback used to be silent, and the failure it hides (DSL prompts handed to a
    non-DSL student) is indistinguishable from a bad kernel.
    """
    if not language:
        return MODE_CUTEDSL
    if language in LANGUAGE_MODE:
        return LANGUAGE_MODE[language]
    if language in LEGACY_LANGUAGE:
        return LEGACY_LANGUAGE[language]
    raise ValueError(f"unknown language {language!r}: register it in LANGUAGE_MODE")


def is_cuda(language: str | None) -> bool:
    return mode_of(language) == MODE_CUDA


def is_tilelang(language: str | None) -> bool:
    return mode_of(language) == MODE_TILELANG


def is_cuda_kernel_file(kernel_file: str | Path | None) -> bool:
    return str(kernel_file or "").endswith(CUDA_KERNEL_SUFFIXES)


def is_tilelang_kernel_file(kernel_file: str | Path | None) -> bool:
    return str(kernel_file or "").endswith(TILELANG_KERNEL_SUFFIXES)


def kb_root_name(language: str | None) -> str:
    """The KB root directory (under the repo) the language's mode reads."""
    return KB_ROOT[mode_of(language)]


def kb_repo_root(kb_dir: Path) -> Path:
    """The repository a mode's KB root (``hw_kb/<mode>``) sits in."""
    return Path(kb_dir).parents[1]


def provenance_dir(repo_root: Path, language: str | None) -> Path:
    """Where the mode's KB provenance lives (human review only, no agent read set)."""
    base = Path(repo_root) / "docs" / "kb-provenance"
    sub = PROVENANCE_SUBDIR[mode_of(language)]
    return base / sub if sub else base


def _provenance_owner(rel_path: str) -> str:
    """The mode whose ``PROVENANCE_SUBDIR`` prefixes a repo-relative path.

    Longest subdir wins, so the shared ``""`` (the DSL mode) only claims what no
    other mode's prefix does -- ``docs/kb-provenance/tilelang/x.md`` belongs to
    tilelang, never to the DSL mode, even though ``""`` matches its start.
    """
    text = str(rel_path)
    owners = [(len(sub), mode) for mode, sub in PROVENANCE_SUBDIR.items() if sub]
    for _, mode in sorted(owners, reverse=True):
        if text.startswith(f"docs/kb-provenance/{PROVENANCE_SUBDIR[mode]}/"):
            return mode
    return MODE_CUTEDSL


def owns_provenance(rel_path: str, language: str | None) -> bool:
    """Whether a repo-relative ``docs/kb-provenance/...`` path belongs to the
    language's mode: each mode owns ``docs/kb-provenance/<its PROVENANCE_SUBDIR>/``,
    and the DSL mode's empty subdir owns everything no other mode claims (the
    shared per-op ``gaps/`` archive included)."""
    return _provenance_owner(rel_path) == mode_of(language)


def prompt_path(meta_dir: Path, name: str, language: str | None) -> Path:
    """``meta/<overlay>/<name>`` for a mode that has an overlay and ships that
    file, else ``meta/<name>``. A prompt with no overlay is language-neutral and
    shared, so an overlay may start with three prompts and grow."""
    overlay = PROMPT_OVERLAY_DIR[mode_of(language)]
    if overlay:
        p = Path(meta_dir) / overlay / name
        if p.is_file():
            return p
    return Path(meta_dir) / name


def language_for_kernel_file(kernel_file: str | Path | None) -> str | None:
    """The language a kernel file implies when the run names none: a ``.cu`` is
    the CUDA mode's student, a ``_tl.py`` the TileLang mode's; anything else
    leaves the old default alone."""
    if is_cuda_kernel_file(kernel_file):
        return DEFAULT_CUDA_LANGUAGE
    if is_tilelang_kernel_file(kernel_file):
        return DEFAULT_TILELANG_LANGUAGE
    return None


def language_problem(language: str, kernel_file: str | Path) -> str | None:
    """Why ``language`` cannot drive a student that edits ``kernel_file``, or None.

    The language picks the prompts and the KB, the file picks what the card
    builds; out of step, the student is told to write CuTe DSL into a ``.cu`` (or
    C++ into a ``.py``) and every tool reads the wrong thing.

    Three suffix modes, so three ways to be out of step: a CUDA language over a
    non-``.cu``, a TileLang language over a non-``_tl.py``, and either special
    suffix under a language of another mode."""
    cu = is_cuda_kernel_file(kernel_file)
    tl = is_tilelang_kernel_file(kernel_file)
    if is_cuda(language) and not cu:
        return (f"language {language!r} is the CUDA mode but the kernel file is "
                f"{kernel_file!r}: a CUDA student edits a .cu (scaffold the card with "
                f"--kernel-file kernel.cu)")
    if is_tilelang(language) and not tl:
        return (f"language {language!r} is the TileLang mode but the kernel file is "
                f"{kernel_file!r}: a TileLang student edits a *_tl.py (scaffold the card "
                f"with --kernel-file kernel_tl.py)")
    if cu and not is_cuda(language):
        return (f"the kernel file {kernel_file!r} is CUDA C++ but language is "
                f"{language!r}: set loop.toml [loop].language = "
                f"{DEFAULT_CUDA_LANGUAGE!r} (or leave it unset to follow the file)")
    if tl and not is_tilelang(language):
        return (f"the kernel file {kernel_file!r} is a TileLang kernel but language is "
                f"{language!r}: set loop.toml [loop].language = "
                f"{DEFAULT_TILELANG_LANGUAGE!r} (or leave it unset to follow the file)")
    return None
