"""Kernel injection: hand a written kernel to a live server by swapping a module slot.

Two entry kinds, one swap mechanism:

* TileLang / python callables -> :mod:`forgekernel.serve.injectors.til_inputs`
  (wrap as an inference-only callable, patch a ``module.attr``);
* CUDA ``.cu`` -> :mod:`forgekernel.serve.injectors.cuda_ext`
  (build a pybind extension, expose its ``run``, then patch the same way).

The patch target is data (a dotted string in the deployment recipe), so which server slot is
replaced is configuration, and the swap is testable on the CPU against a fake module. Nothing
here imports torch at module load.
"""

from __future__ import annotations

from .til_inputs import (
    InferenceOnly,
    Install,
    install,
    patch_attr,
    split_attr,
    wrap_callable,
)
from .cuda_ext import (
    CudaBuild,
    build_cuda_extension,
    build_kwargs,
    has_pybind_module,
)

__all__ = [
    "InferenceOnly",
    "Install",
    "install",
    "patch_attr",
    "split_attr",
    "wrap_callable",
    "CudaBuild",
    "build_cuda_extension",
    "build_kwargs",
    "has_pybind_module",
]
