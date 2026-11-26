"""Wrap a TileLang kernel as a callable the server can hold, and patch it into a module slot.

Two concerns, kept separate:

* :func:`wrap_callable` turns a bare ``fn(*tensors) -> tensor`` (a TileLang kernel's
  ``tilelang.compile(...)(...)`` callable, or any torch-in/torch-out function) into an
  ``nn.Module``-friendly callable that is explicitly marked inference-only. The server's decode
  path calls it without autograd; we do not fake a backward. Wrapping also lets the harness tag
  the object (e.g. its deploy gates) without touching the kernel source.
* :func:`patch_attr` / :func:`install` replace a dotted ``module.attr`` with our callable and
  return an :class:`Install` handle that can undo it. The *target* is data (a dotted string),
  so "which server slot to swap" is configuration, and the patch logic is testable against a
  fake module on the CPU -- no server, no torch import at module load.

Nothing here imports torch at module load; the wrapper takes torch tensors opaquely.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


class InferenceOnly:
    """A thin holder around a torch-in/torch-out callable, marked inference-only.

    Calling it forwards to the wrapped function. It deliberately does not implement autograd:
    swapping a kernel into a serving decode path is inference, and pretending to support grad
    would let a caller silently get a wrong (un-differentiated) backward. A caller that needs
    grad wraps the kernel itself with its own ``autograd.Function`` (see the porting docs)."""

    def __init__(self, fn, *, name: str = "kernel"):
        if not callable(fn):
            raise TypeError("InferenceOnly needs a callable")
        self.fn = fn
        self.name = name
        self.gates: dict[str, bool | None] = {}

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"InferenceOnly({self.name})"


def wrap_callable(fn, *, name: str = "kernel") -> InferenceOnly:
    """Wrap a torch-in/torch-out function for serving. Idempotent for an already-wrapped fn."""
    if isinstance(fn, InferenceOnly):
        return fn
    return InferenceOnly(fn, name=name)


def split_attr(dotted: str) -> tuple[str, str]:
    """``"a.b.c.attr"`` -> (``"a.b.c"``, ``"attr"``). Raises on a string with no dot."""
    if "." not in dotted:
        raise ValueError(f"target must be 'module.attr', got {dotted!r}")
    module, attr = dotted.rsplit(".", 1)
    if not module or not attr:
        raise ValueError(f"target must be 'module.attr', got {dotted!r}")
    return module, attr


@dataclass
class Install:
    """The record of one patch: where it landed, what was there before, how to undo it."""

    dotted: str
    module: object
    attr: str
    previous: object
    existed: bool = True

    def active(self, ) -> bool:
        return getattr(self.module, self.attr, None) is not None

    def uninstall(self) -> None:
        """Put the original attribute back exactly as it was (delete it if it did not exist)."""
        if self.existed:
            setattr(self.module, self.attr, self.previous)
        else:
            try:
                delattr(self.module, self.attr)
            except AttributeError:
                pass


def patch_attr(target: str, replacement, *, importer=importlib.import_module) -> Install:
    """Replace ``target`` (a dotted ``module.attr``) with ``replacement``; return an undo handle.

    The module is imported via ``importer`` (injectable, so a test passes a fake importer that
    maps the dotted module to a stand-in object). The previous value -- and whether it existed --
    is captured so :meth:`Install.uninstall` restores the module exactly, deleting the attribute
    if it was absent before. A patched server can thus be reset between readings without a
    restart."""
    module_name, attr = split_attr(target)
    module = importer(module_name)
    existed = hasattr(module, attr)
    previous = getattr(module, attr, None)
    setattr(module, attr, replacement)
    return Install(dotted=target, module=module, attr=attr, previous=previous, existed=existed)


def install(target: str, fn, *, name: str | None = None,
            importer=importlib.import_module) -> Install:
    """Wrap ``fn`` for serving and patch it into ``target`` in one step."""
    wrapped = wrap_callable(fn, name=name or target)
    return patch_attr(target, wrapped, importer=importer)
