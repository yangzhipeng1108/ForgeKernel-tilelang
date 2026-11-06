"""Where an offline student may read the DSL's own Python source from.

The student runs without network (loop/agent_policy.py), so the CuTe-DSL API it writes
against has to be readable from disk. ``[loop] dsl_source_dirs`` names that checkout
explicitly; when a run leaves it empty, the student is left to discover ``cutlass.cute``
by trial and error — in one GDN run a large share of the probe failures were spellings
of the 4.6.0 API, not ideas being tested.

This module derives a default for a ``cutedsl`` run from the ``nvidia-cutlass-dsl``
distribution installed in a Python interpreter on this machine (the one running the
loop, or ``[loop] dsl_python``): the wheel family puts the whole Python DSL under one
directory (``nvidia_cutlass_dsl/dsl_packages/``, holding ``cutlass`` and its siblings;
a ``.pth`` file adds it to ``sys.path``), and that directory is what the student needs.

Three things keep the default honest:

* **Version.** The job pins the DSL with ``[env] DSL_PIN`` (submit.toml); a local
  install of another version is not the API the kernel is compiled against and must
  not be handed over silently. Mismatch → no default, a warning that names both.
* **Reference source.** A default that lies inside (or contains) a declared
  ``ref_source_paths`` entry would hand the student the reference through the DSL
  door. Overlap → no default, a note that says so.
* **Explicit wins.** A declared ``dsl_source_dirs`` is used as written, whatever is
  installed here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

LANGUAGE_CUTEDSL = "cutedsl"
LANGUAGE_TILELANG = "tilelang"
KEY_DSL_PYTHON = "dsl_python"
ENV_DSL_PIN = "DSL_PIN"

# Per language: where the Python API the student writes against lives on disk. The
# CuTe DSL wheel family hides the packages under ``nvidia_cutlass_dsl/dsl_packages/``
# and adds that to sys.path with a .pth; TileLang ships one plain package, so the
# import root IS the answer and the metadata walk has nothing to find.
@dataclass(frozen=True)
class DslPackage:
    shell: str              # the distribution that carries the version
    import_name: str         # the module to import when metadata names no file
    core: str | None = None  # a second dist-info naming the files, when the shell has none
    subdir: str | None = None  # the directory under the package the student reads

    @property
    def label(self) -> str:
        return self.shell


DSL_PACKAGES = {
    LANGUAGE_CUTEDSL: DslPackage(shell="nvidia-cutlass-dsl", import_name="cutlass",
                                 core="nvidia-cutlass-dsl-libs-core",
                                 subdir="dsl_packages"),
    LANGUAGE_TILELANG: DslPackage(shell="tilelang", import_name="tilelang"),
}
DIST_SHELL = DSL_PACKAGES[LANGUAGE_CUTEDSL].shell
DIST_CORE = DSL_PACKAGES[LANGUAGE_CUTEDSL].core
PROBE_TIMEOUT_S = 60

ORIGIN_EXPLICIT = "explicit"
ORIGIN_DEFAULT = "default"
ORIGIN_NONE = "none"

# Runs in the target interpreter: report the distribution's version and the package
# roots the DSL imports from. Metadata first (no import side effects, works for an
# interpreter that cannot import the DSL, e.g. a CPU-only venv holding the wheel);
# the import as the fallback for an install without dist-info (a source checkout on
# sys.path).
#
# ``subdir`` is what the two wheel families differ in: the CuTe DSL one puts its
# packages under ``nvidia_cutlass_dsl/dsl_packages/`` (a .pth adds that to sys.path),
# so RECORD has to be walked to find it; TileLang ships one plain package and its
# import root is already the directory the student reads.
_PROBE = r"""
import importlib.metadata as m, json, os, sys
out = {"version": None, "roots": [], "how": None}
SUBDIR = %(subdir)r
IMPORT = %(import_name)r
def _dist(name):
    if name is None:
        return None
    try:
        return m.distribution(name)
    except m.PackageNotFoundError:
        return None
for name in (%(shell)r, %(core)r):
    d = _dist(name)
    if d is not None and out["version"] is None:
        out["version"] = d.version
roots = []
if SUBDIR:
    core = _dist(%(core)r)
    if core is not None and core.files:
        for f in core.files:
            parts = f.parts
            if len(parts) >= 3 and parts[-1].endswith(".py") and SUBDIR in parts:
                i = parts.index(SUBDIR)
                roots.append(str((core.locate_file(f).resolve().parents[len(parts) - i - 2])))
        out["how"] = "metadata"
if not roots:
    try:
        mod = __import__(IMPORT)
        # A wheel that adds the packages one by one (the CuTe DSL family) leaves
        # __path__ pointing at each package, so the directory to hand over is its
        # parent. A plain package's __path__ entry IS the directory.
        take_parent = bool(SUBDIR)
        roots = [str(os.path.dirname(os.path.abspath(p)) if take_parent
                     else os.path.abspath(p)) for p in mod.__path__]
        out["how"] = "import"
    except Exception:
        pass
seen = []
for r in roots:
    if r not in seen:
        seen.append(r)
out["roots"] = seen
print(json.dumps(out))
"""


@dataclass(frozen=True)
class DslSource:
    """The resolved ``dsl_source_dirs`` and where they came from.

    ``origin``: ``explicit`` (declared in loop.toml), ``default`` (derived from an
    installed distribution), ``none`` (nothing usable). ``version`` is the installed
    distribution's version when one was found, whether or not it was accepted.
    ``distribution`` is the name that version belongs to, so a TileLang default is
    not described as the CuTe DSL one.
    ``notes`` are what the loop prints — one line each, ``[loop]``-prefixed by the
    caller."""

    dirs: tuple[Path, ...] = ()
    origin: str = ORIGIN_NONE
    version: str | None = None
    distribution: str = DIST_SHELL
    notes: tuple[str, ...] = ()

    @property
    def defaulted(self) -> bool:
        return self.origin == ORIGIN_DEFAULT

    def describe(self) -> str:
        """The dry-run line's value."""
        if self.origin == ORIGIN_EXPLICIT:
            return f"explicit {[str(p) for p in self.dirs]}"
        if self.origin == ORIGIN_DEFAULT:
            return (f"default {[str(p) for p in self.dirs]}  "
                    f"({self.distribution} {self.version}, from the installed distribution)")
        why = self.notes[-1] if self.notes else "not declared"
        return f"(none — {why})"


def _abs(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def probe_interpreter(python: str, *, env: dict | None = None,
                      timeout_s: float = PROBE_TIMEOUT_S,
                      language: str = LANGUAGE_CUTEDSL) -> dict | None:
    """``{"version", "roots", "how"}`` from ``python``, or None when the probe could
    not run (missing interpreter, timeout, unparsable output)."""
    pkg = DSL_PACKAGES[language]
    code = _PROBE % {"shell": pkg.shell, "core": pkg.core, "subdir": pkg.subdir,
                     "import_name": pkg.import_name}
    try:
        proc = subprocess.run([python, "-c", code], capture_output=True, text=True,
                              timeout=timeout_s, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("roots"), list):
        return None
    return data


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def resolve(cfg: dict, *, language: str, submit_env: dict | None = None,
            python: str | None = None, probe_env: dict | None = None,
            pin_readable: bool = True) -> DslSource:
    """The run's ``dsl_source_dirs``: as declared, else derived, else none.

    ``cfg`` is the parsed loop.toml; ``language`` the resolved kernel language;
    ``submit_env`` the card's ``[env]`` (for ``DSL_PIN``); ``python`` overrides the
    interpreter (``[loop] dsl_python`` then ``sys.executable``).

    ``pin_readable`` False means the caller could not read the card's submit config
    at all, so ``submit_env`` is not evidence of "this card pins nothing": the
    version gate would be off without saying so, and a default derived here would
    hand the student whatever this machine happens to have installed. Refuse to
    default instead — an unreadable card is a config problem the operator can see,
    a silent version mismatch is not."""
    lc = cfg.get("loop", {}) if isinstance(cfg, dict) else {}
    declared = lc.get("dsl_source_dirs") or []
    if isinstance(declared, list) and declared:
        return DslSource(dirs=tuple(_abs(p) for p in declared), origin=ORIGIN_EXPLICIT)
    pkg = DSL_PACKAGES.get(language)
    if pkg is None:
        # A CUDA C++ student reads headers, which [loop] cuda_source_dirs names; there
        # is no Python API to default to.
        return DslSource(notes=(f"language={language!r}: no DSL source to default",))
    shell = pkg.shell
    if not pin_readable:
        return DslSource(notes=(
            f"dsl_source_dirs is empty and the card's submit config could not be read "
            f"(submit.toml / submit.local.toml under operators/) — its [env] {ENV_DSL_PIN} "
            f"cannot be checked against the installed {shell}, so nothing is "
            f"defaulted. Fix the card, or declare [loop] dsl_source_dirs explicitly",))

    interp = python or lc.get(KEY_DSL_PYTHON) or sys.executable
    interp_note = f"[loop] {KEY_DSL_PYTHON}" if lc.get(KEY_DSL_PYTHON) else "the loop's interpreter"
    data = probe_interpreter(str(interp), env=probe_env, language=language)
    if data is None:
        return DslSource(notes=(
            f"dsl_source_dirs is empty and {shell} could not be located in "
            f"{interp_note} ({interp}) — the student is offline, so it can read the "
            f"DSL's API only from a local checkout named in [loop] dsl_source_dirs",))
    version = data.get("version")
    roots = [Path(r).resolve() for r in data.get("roots", []) if r]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        return DslSource(version=version, distribution=shell, notes=(
            f"dsl_source_dirs is empty and {shell} is not installed in "
            f"{interp_note} ({interp}) — the student is offline, so it can read the "
            f"DSL's API only from a local checkout named in [loop] dsl_source_dirs",))

    pin = (submit_env or {}).get(ENV_DSL_PIN)
    pin = str(pin).strip() if pin is not None else ""
    if pin and version and pin != version:
        return DslSource(version=version, distribution=shell, notes=(
            f"dsl_source_dirs is empty; {shell} {version} is installed in "
            f"{interp_note} but the job pins [env] {ENV_DSL_PIN} = {pin} — not defaulting "
            f"to another version's source. Install {pin} there, point [loop] "
            f"{KEY_DSL_PYTHON} at an interpreter that has it, or declare [loop] "
            f"dsl_source_dirs explicitly",))
    if pin and not version:
        return DslSource(notes=(
            f"dsl_source_dirs is empty; a DSL package was found in {interp_note} but "
            f"without a {shell} version to check against [env] {ENV_DSL_PIN} = {pin} "
            f"— declare [loop] dsl_source_dirs explicitly",))

    refs = [_abs(p) for p in (lc.get("ref_source_paths") or [])
            if isinstance(p, str)]
    clash = [(r, ref) for r in roots for ref in refs if _overlaps(r, ref)]
    if clash:
        r, ref = clash[0]
        return DslSource(version=version, distribution=shell, notes=(
            f"dsl_source_dirs is empty; the installed {shell} at {r} overlaps the "
            f"declared reference source {ref} — not defaulting (the DSL door must not "
            f"open onto the reference). Declare [loop] dsl_source_dirs explicitly",))

    return DslSource(
        dirs=tuple(roots), origin=ORIGIN_DEFAULT, version=version, distribution=shell,
        notes=(f"dsl_source_dirs: defaulted to {[str(r) for r in roots]} "
               f"({shell} {version or 'unversioned'}, {interp_note}"
               + (f", matches [env] {ENV_DSL_PIN}" if pin else "") + ")",))
