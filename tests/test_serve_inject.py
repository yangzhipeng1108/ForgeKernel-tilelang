"""Tests for kernel injection — no GPU, no compiler, no server.

The swap logic (wrap, patch a dotted ``module.attr``, undo) and the CUDA build-argument assembly
are pure and tested here with a fake importer, a fake ``cpp_extension.load`` and a temp ``.cu``.
The only untested part is the real compile/execution, which needs hardware.
"""

from __future__ import annotations

import types

import pytest

from forgekernel.serve.injectors import (
    CudaBuild,
    InferenceOnly,
    build_cuda_extension,
    build_kwargs,
    has_pybind_module,
    install,
    patch_attr,
    split_attr,
    wrap_callable,
)


# ---------------------------------------------------------------- wrapping

def test_wrap_forwards_calls_and_is_marked_inference_only():
    kernel = wrap_callable(lambda x: x + 1, name="mla")
    assert isinstance(kernel, InferenceOnly)
    assert kernel(41) == 42
    assert "mla" in repr(kernel)


def test_wrap_is_idempotent():
    k = wrap_callable(lambda: 0)
    assert wrap_callable(k) is k


def test_wrap_rejects_non_callable():
    with pytest.raises(TypeError):
        wrap_callable(123)


def test_wrapped_callable_carries_no_autograd_by_design():
    k = wrap_callable(lambda: 0)
    # no backward on purpose: swapping into a serving path is inference-only
    assert not hasattr(k, "backward")


# ---------------------------------------------------------------- patch / install

def test_split_attr():
    assert split_attr("a.b.c.run") == ("a.b.c", "run")
    with pytest.raises(ValueError):
        split_attr("nodot")
    with pytest.raises(ValueError):
        split_attr(".attr")


class _FakeModule(types.SimpleNamespace):
    pass


def _importer(modules: dict):
    def imp(name):
        return modules[name]
    return imp


def test_patch_attr_swaps_and_records_previous():
    mod = _FakeModule(run="ORIGINAL")
    store = {"pkg.backend": mod}
    inst = patch_attr("pkg.backend.run", "NEW", importer=_importer(store))
    assert mod.run == "NEW"
    assert inst.previous == "ORIGINAL"
    inst.uninstall()
    assert mod.run == "ORIGINAL"


def test_patch_attr_restores_a_missing_attribute_to_none():
    mod = _FakeModule()                       # attr did not exist
    store = {"pkg.backend": mod}
    inst = patch_attr("pkg.backend.decode", "NEW", importer=_importer(store))
    assert mod.decode == "NEW"
    inst.uninstall()
    assert not hasattr(mod, "decode")


def test_install_wraps_then_patches():
    mod = _FakeModule(decode="ORIG")
    store = {"srv.attn": mod}
    inst = install("srv.attn.decode", lambda q: ("out", q), importer=_importer(store))
    assert isinstance(mod.decode, InferenceOnly)
    assert mod.decode("q") == ("out", "q")
    inst.uninstall()
    assert mod.decode == "ORIG"


# ---------------------------------------------------------------- cuda build assembly

_CU_GOOD = """
#include <torch/extension.h>
torch::Tensor run(torch::Tensor x) { return x; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""

_CU_NO_MODULE = """
#include <torch/extension.h>
torch::Tensor run(torch::Tensor x) { return x; }
"""


def test_has_pybind_module_detects_the_binding():
    assert has_pybind_module(_CU_GOOD)
    assert not has_pybind_module(_CU_NO_MODULE)


def test_build_kwargs_refuses_a_cu_without_binding(tmp_path):
    p = tmp_path / "kernel.cu"
    p.write_text(_CU_NO_MODULE, encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        build_kwargs(CudaBuild(cu_path=str(p), name="k"))
    assert "PYBIND11_MODULE" in str(ei.value)


def test_build_kwargs_defaults_flags_and_source(tmp_path):
    p = tmp_path / "kernel.cu"
    p.write_text(_CU_GOOD, encoding="utf-8")
    kw = build_kwargs(CudaBuild(cu_path=str(p), name="mla_ext"))
    assert kw["name"] == "mla_ext"
    assert kw["sources"] == [str(p)]
    assert "-O3" in kw["extra_cflags"] and "--use_fast_math" in kw["extra_cflags"]


def test_build_cuda_extension_returns_bound_run_via_injected_load(tmp_path):
    p = tmp_path / "kernel.cu"
    p.write_text(_CU_GOOD, encoding="utf-8")
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(run=lambda x: ("ran", x))

    run = build_cuda_extension(CudaBuild(cu_path=str(p), name="mla_ext"), load_fn=fake_load)
    assert seen["name"] == "mla_ext"
    assert run("in") == ("ran", "in")


def test_build_cuda_extension_refuses_extension_without_run(tmp_path):
    p = tmp_path / "kernel.cu"
    p.write_text(_CU_GOOD, encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        build_cuda_extension(CudaBuild(cu_path=str(p), name="x"),
                             load_fn=lambda **kw: types.SimpleNamespace())
    assert "no 'run'" in str(ei.value)
