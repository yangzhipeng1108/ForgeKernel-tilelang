"""Build a CUDA ``.cu`` kernel into a loadable extension and hand back its ``run`` callable.

A CUDA student kernel is a single ``.cu`` with ``void/torch::Tensor run(...)`` and a
``PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`` that binds it (the card template's contract, see
``leaderboard.load_cuda_entry``). To swap it into a server, build it with
``torch.utils.cpp_extension.load`` and expose the bound ``run`` -- exactly what the eval path
already does, factored here for the serving side.

``torch`` is imported lazily inside :func:`build_cuda_extension`, so importing this module (and
running its CPU tests with a fake torch) needs no GPU. The build itself needs CUDA and is the
only part that cannot be unit-tested without hardware; the argument assembly (flags, module
name, the ``.cu`` check) is pure and tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# The card contract's default flags (mirror of leaderboard.CUDA_DEFAULT_CFLAGS + -lcuda); kept
# here so the serving build matches what the card was judged with.
CUDA_DEFAULT_CFLAGS = ["-O3", "--use_fast_math"]

# A .cu must define the pybind module or cpp_extension has nothing to import.
_PYBIND = re.compile(r"PYBIND11_MODULE\s*\(\s*TORCH_EXTENSION_NAME\s*,")


def has_pybind_module(source: str) -> bool:
    """Whether a ``.cu`` source defines the ``PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)``."""
    return bool(_PYBIND.search(source))


@dataclass
class CudaBuild:
    """What :func:`build_cuda_extension` needs: the source path and the build knobs."""

    cu_path: str
    name: str
    extra_cflags: list[str] = field(default_factory=list)
    extra_cuda_cflags: list[str] = field(default_factory=list)
    verbose: bool = False


def build_kwargs(build: CudaBuild) -> dict:
    """The kwargs dict for ``torch.utils.cpp_extension.load`` (pure; testable without torch).

    Refuses a source without the ``PYBIND11_MODULE`` line before any compile, naming the file:
    the failure mode this prevents is a build that succeeds but exports nothing importable."""
    source = open(build.cu_path, encoding="utf-8").read()
    if not has_pybind_module(source):
        raise ValueError(
            f"{build.cu_path} defines no PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...); a .cu "
            f"kernel must bind its 'run' for the server to load it")
    return {
        "name": build.name,
        "sources": [build.cu_path],
        "extra_cflags": list(build.extra_cflags or CUDA_DEFAULT_CFLAGS),
        "extra_cuda_cflags": list(build.extra_cuda_cflags),
        "verbose": build.verbose,
    }


def build_cuda_extension(build: CudaBuild, *, load_fn=None):
    """Build the extension (lazy torch import) and return its bound ``run`` callable.

    ``load_fn`` lets a test inject ``torch.utils.cpp_extension.load`` (a fake returning a module
    with ``.run``) so the dispatch is exercised without a compiler. Without it, torch is imported
    here -- the one line that needs CUDA."""
    kwargs = build_kwargs(build)
    if load_fn is None:
        from torch.utils import cpp_extension
        load_fn = cpp_extension.load
    module = load_fn(**kwargs)
    run = getattr(module, "run", None)
    if run is None:
        raise ValueError(f"built extension {build.name!r} exposes no 'run'")
    return run
