"""Real-hardware integration for the serving subsystem.

These are the paths that cannot be verified with a fake: a real CUDA build through
``cpp_extension``, a real CUDA graph capture, and the real process + HTTP + subprocess wiring in
``serve.runtime``. Each test skips (never silently passes) when its hardware is absent, so a
CPU-only CI stays green while a GPU box actually exercises the leaf.

Kept separate from the fake-driven unit tests on purpose: a reviewer can see, in one file, exactly
which behaviours were proven on real hardware and which are merely unit-tested.
"""

from __future__ import annotations

import http.server
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="serving hardware tests need torch")

try:
    from forgekernel.serve.injectors.cuda_ext import CudaBuild, build_cuda_extension, build_kwargs
    from forgekernel.serve import runtime as rt
    from forgekernel.serve.harness import ServeConfig
    from forgekernel.serve.checks import graph_capturable, make_graph_capture
except Exception as exc:  # pragma: no cover - import failure is a real error, surface it
    raise


def _have_cuda() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _have_nvcc() -> bool:
    import shutil
    if shutil.which("nvcc"):
        return True
    return Path("/usr/local/cuda/bin/nvcc").is_file()


HAVE_CUDA = _have_cuda()
HAVE_NVCC = _have_nvcc()

_CU_MIN = r"""
#include <torch/extension.h>
torch::Tensor run(torch::Tensor x) { return x + 1; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "add one"); }
"""


@pytest.mark.skipif(not (HAVE_CUDA and HAVE_NVCC),
                    reason="needs CUDA + nvcc to build a real extension")
def test_build_cuda_extension_compiles_and_runs_a_real_cu(tmp_path):
    cu = tmp_path / "minimal.cu"
    cu.write_text(_CU_MIN, encoding="utf-8")
    build = CudaBuild(cu_path=str(cu), name="fk_serve_hw_smoke", extra_cuda_cflags=["-O3"])
    assert build_kwargs(build)["sources"] == [str(cu)]          # pure assembly, no compile
    run = build_cuda_extension(build)                            # the real cpp_extension.load
    x = torch.arange(6, dtype=torch.float32, device="cuda")
    y = run(x)
    assert y.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert y.is_cuda


@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA to capture a real graph")
def test_graph_capturable_on_a_real_torch_op():
    x = torch.randn(1024, 1024, device="cuda")
    w = torch.randn(1024, 1024, device="cuda")
    out = torch.empty(1024, 1024, device="cuda")           # stable output buffer for graph replay
    result = graph_capturable(make_graph_capture(lambda a, b: torch.mm(a, b, out=out), (x, w)))
    assert result.passed is True


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_real_health_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def test_wait_for_health_polls_a_real_local_http_server():
    # real urllib.request against a real socket; no fake http_get
    server, port = _start_real_health_server()
    try:
        elapsed = rt.wait_for_health(f"http://127.0.0.1:{port}", timeout_s=10, poll_s=0.05)
        assert elapsed >= 0
        assert rt._default_http_get(f"http://127.0.0.1:{port}/health") == 200
    finally:
        server.shutdown()


def test_run_bench_runs_a_real_subprocess_and_parses_stdout(monkeypatch):
    # real subprocess.run: a child python prints a bench JSON summary on its last line
    summary = {"output_throughput": 450.5, "mean_ttft_ms": 120.0, "max_concurrency": 16}
    code = ("print('Namespace(...)'); "
            f"print({json.dumps(json.dumps(summary))})")
    monkeypatch.setattr(rt, "bench_command",
                        lambda cfg, *, concurrency: [sys.executable, "-c", code])
    cfg = ServeConfig(model_path="x", python=sys.executable)
    res = rt.run_bench(cfg, concurrency=16)
    assert res.returncode == 0
    assert res.summary.get("output_throughput") == 450.5


def test_start_server_spawns_a_real_process_and_stop_terminates_it(monkeypatch, tmp_path):
    # real subprocess.Popen of a real health server; wait_healthy via real urllib; real stop()
    port = _free_port()
    stub = tmp_path / "health.py"
    stub.write_text(
        "import sys\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        f"HTTPServer(('127.0.0.1', {port}), type('H', (BaseHTTPRequestHandler,), {{\n"
        "  'do_GET': lambda s: (s.send_response(200), s.end_headers(), s.wfile.write(b'ok')),\n"
        "  'log_message': lambda *a: None,\n"
        "})).serve_forever()\n", encoding="utf-8")
    monkeypatch.setattr(rt, "serve_command",
                        lambda cfg: [sys.executable, str(stub)])
    cfg = ServeConfig(model_path="x", port=port, python=sys.executable)
    handle = rt.start_server(cfg, wait_healthy=True)
    try:
        assert handle.proc.poll() is None                 # alive
        assert rt._default_http_get(handle.base_url + "/health") == 200
    finally:
        handle.stop()
    assert handle.proc.poll() is not None                 # terminated
