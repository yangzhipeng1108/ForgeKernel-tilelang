"""GPU architecture tokens: parsing, device probing, KB lookup and instruction gating.

A task names its target GPU as an ``arch`` token (``sm86`` on an RTX 3090, ``sm90`` on
an H100, ``sm100`` on a B200). That one string does three things, and this module is
the one place that keeps them consistent:

- it names the compile target (CUDA ``-arch=``, TileLang ``{"arch": "smXX"}``);
- it picks the hardware knowledge base ``hw_kb/<language>/<arch>/``;
- it decides **which instruction families the student may use**: ``wgmma`` is sm90+,
  ``tcgen05`` / TMEM is sm100+, TMA descriptor copies are sm90+, CTA clusters sm90+.
  On Ampere (sm80/sm86) none of those exist, and a kernel that emits them is a
  compile error or a silent fall to a slower path -- so the gate has to be mechanical,
  not a token in a prompt.

``sm_86`` and ``sm86`` are the same architecture spelled two ways (CUDA uses the
underscore in ``-arch=sm_86``; the KB directory and the task token drop it). The helpers
here normalise to the **no-underscore** form used by ``task.toml`` and the KB tree, and
offer a one-shot device probe for ``forge init`` / a run with no explicit arch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ``sm86`` / ``sm100`` / ``sm90a`` -- the token form task.toml and the KB tree use.
_ARCH_TOKEN_RE = re.compile(r"^sm(?P<major>\d+)(?P<suffix>[a-z]?)$", re.IGNORECASE)


@dataclass(frozen=True)
class Arch:
    """A parsed architecture token: ``sm`` + compute-capability major + optional letter."""

    token: str          # normalised, no underscore: "sm86", "sm90a", "sm100"
    capability: int     # numeric major: 86, 90, 100 (minor is folded in for 8.x)
    suffix: str         # "a" for arch-specific (sm90a/sm100a), else ""

    @property
    def sm_code(self) -> str:
        """The CUDA ``-arch=`` spelling: ``sm_86`` (underscore)."""
        return self.token.replace("sm", "sm_", 1)


def parse(token: str | None) -> Arch:
    """Parse ``sm86`` / ``sm_86`` / ``sm90a`` into an :class:`Arch`. Raises on a bad token.

    The CUDA spelling ``sm_86`` is accepted and normalised to ``sm86``; ``sm90a`` keeps
    its letter (an arch-specific profile that only some instructions require).
    """
    raw = (token or "").strip().lower().replace("_", "")
    m = _ARCH_TOKEN_RE.match(raw)
    if not m:
        raise ValueError(
            f"not a GPU architecture token: {token!r} (want e.g. 'sm86', 'sm90', "
            f"'sm100' or the CUDA form 'sm_86')"
        )
    major = int(m.group("major"))
    suffix = m.group("suffix")
    return Arch(token=f"sm{major}{suffix}", capability=major, suffix=suffix)


def capability_from_device_index(index: int = 0) -> int | None:
    """Query the GPU's compute capability major (``(8, 6)`` -> ``86``) via torch.

    Returns None when torch or a GPU is unavailable -- the caller then falls back to
    the task's declared arch rather than guessing. Never raises: a missing GPU is a
    normal state on a CPU dev box, not an error worth a traceback.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(index)
    except Exception:
        return None
    return major * 10 + minor


def detect_arch(index: int = 0) -> str | None:
    """The arch token for the GPU at ``index``, or None when no GPU/torch is present.

    ``forge init`` uses this to fill ``[task].arch`` from the machine it runs on
    instead of defaulting every task to the Blackwell ``sm100``.
    """
    cap = capability_from_device_index(index)
    if cap is None:
        return None
    return f"sm{cap}"


def kb_dir(repo: Path, language: str, token: str) -> Path:
    """``hw_kb/<language>/<token>/`` -- the knowledge base a task's [task].arch picks."""
    from forgekernel.task import LOOP_LANGUAGE
    from forgekernel.loop import lang_mode

    root = Path(repo) / lang_mode.kb_root_name(LOOP_LANGUAGE[language])
    return root / parse(token).token


def available_arches(repo: Path, language: str) -> list[str]:
    """The arch tokens that actually have a knowledge base for ``language``, sorted.

    Used to turn ``arch = "sm86": no KB`` into an error that names what *does* exist,
    so a typo or an unsupported GPU is one message away from the fix.
    """
    from forgekernel.task import LOOP_LANGUAGE
    from forgekernel.loop import lang_mode

    root = Path(repo) / lang_mode.kb_root_name(LOOP_LANGUAGE[language])
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and _ARCH_TOKEN_RE.match(p.name)
    )


# -- instruction gating ------------------------------------------------------------------
#
# The families below are the ones whose presence changes what a *correct* kernel may
# look like. Each threshold is the first compute capability that ships the feature:
#
#   cp.async              sm80+   (Ampere)
#   mma.sync HMMA/tf32    sm80+
#   ldmatrix / stmatrix   sm75+ / sm90+ (stmatrix)
#   wgmma (warpgroup)     sm90+   (Hopper)
#   TMA tensor descriptor  sm90+  (Hopper)
#   CTA clusters / DSMEM  sm90+
#   tcgen05 / TMEM        sm100+  (Blackwell)
#   blockscaled (nvfp4)   sm100+/sm120+

_MIN_CAPABILITY = {
    "cp.async": 80,
    "mma.sync": 80,
    "ldmatrix": 75,
    "stmatrix": 90,
    "wgmma": 90,
    "tma": 90,
    "cluster": 90,
    "tcgen05": 100,
    "tmem": 100,
    "blockscaled": 100,
}


def supports(token: str, feature: str) -> bool:
    """Whether architecture ``token`` ships ``feature`` (see ``_MIN_CAPABILITY``).

    Unknown features return False and a message-worthy KeyError is avoided on purpose:
    a feature this table does not know cannot be vouched for, and pretending it is
    supported is how a wgmma instruction reaches an Ampere card.
    """
    need = _MIN_CAPABILITY.get(feature)
    if need is None:
        raise KeyError(f"unknown GPU feature {feature!r}; add it to _MIN_CAPABILITY")
    return parse(token).capability >= need


def unsupported(token: str, features) -> list[str]:
    """The subset of ``features`` that ``token`` does **not** ship (empty = all present).

    The loop consults this before it hands a student an instruction family: on sm86 it
    answers ``['wgmma', 'tcgen05', ...]`` so the prompt / KB guidance can forbid exactly
    what this GPU lacks, per-arch instead of per-prompt.
    """
    return [f for f in features if not supports(token, f)]
