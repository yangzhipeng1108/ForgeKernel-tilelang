"""Re-run an archived sm120 probe against real hardware and record its output.

The probes under ``hw_kb/tilelang/sm120/_probes/`` were written while the recipes
were being earned; most are top-level scripts with no CLI, and several are
modules whose builder functions were called from a REPL. This runner gives them
one entry point and one place to keep the readings.

It runs a probe **as a subprocess** (so a hang or a crash cannot take the runner
with it), under a wall-clock budget, and appends the tail of its output to
``_probes/_readings.md`` with the date, the TileLang version and the GPU.

Usage (on the GPU box, inside the env that has tilelang):

    python revalidate_probes.py --probe sk.py
    python revalidate_probes.py --all --budget 600
    python revalidate_probes.py --list

Design notes:

  * Subprocess, not import. Two of the probes hang (a barrier that never fills
    is the whole point of several recipes); importing them would hang the runner.
  * The readings file is append-only and keyed by probe name + date. A probe that
    starts failing stays visible as a line, it does not erase the old reading.
  * It does not try to normalize probe output. The probes print different shapes
    on purpose; the runner records what they printed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# where things live (relative to this file, which sits in the probes dir)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "fk_probes_20261111_0247"
READINGS = HERE / "_readings.md"

# Probes known to need an explicit C (atomic accumulate). The archived ones that
# used out_idx=[2] against T.atomic_add gave wrong max_diff (455, 1.49e5); the
# fixed pair below is what revalidating actually runs when asked for "sk".
FIXED = {
    "sk": "sk_fixed.py",
}

DEFAULT_BUDGET = 600


def list_probes() -> list[str]:
    if not ARCHIVE.is_dir():
        return []
    return sorted(p.name for p in ARCHIVE.glob("*.py"))


def _tilelang_version(py: str) -> str:
    try:
        out = subprocess.run([py, "-c", "import tilelang;print(tilelang.__version__)"],
                             capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or "?"
    except Exception:                                             # noqa: BLE001
        return "?"


def _gpu_name(py: str) -> str:
    try:
        out = subprocess.run(
            [py, "-c", "import torch;print(torch.cuda.get_device_name(0),"
                       "torch.cuda.get_device_capability(0))"],
            capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or "?"
    except Exception:                                             # noqa: BLE001
        return "?"


def run_one(probe: str, budget: int, py: str, extra_env: dict[str, str]) -> tuple[int, str]:
    path = ARCHIVE / probe
    if not path.exists():
        return 127, f"{probe}: not found in archive"
    env = dict(os.environ)
    env.update(extra_env)
    try:
        proc = subprocess.run([py, str(path)], cwd=str(ARCHIVE), capture_output=True,
                              text=True, timeout=budget, env=env)
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-12:])
        return proc.returncode, tail
    except subprocess.TimeoutExpired:
        return 124, f"{probe}: TIMEOUT after {budget}s (this is a real result -- record it)"


def append_readings(probe: str, rc: int, tail: str, tl: str, gpu: str) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    block = [
        f"\n### {probe} — {stamp}",
        f"- rc={rc}  tilelang={tl}  gpu={gpu}",
        "```",
        tail if tail else "(no output)",
        "```",
    ]
    if not READINGS.exists():
        READINGS.write_text(
            "# sm120 probe revalidations\n\n"
            "> Append-only. Each block is one run of one archived probe on real hardware.\n"
            "> A later run does not delete an earlier one, so a regression stays visible.\n",
            encoding="utf-8")
    with READINGS.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(block) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", help="a file name from the archive, e.g. ca2.py")
    ap.add_argument("--all", action="store_true", help="run every archived probe")
    ap.add_argument("--list", action="store_true", help="list the archived probes")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="seconds per probe")
    ap.add_argument("--python", default=sys.executable, help="interpreter that has tilelang")
    ap.add_argument("--no-record", action="store_true", help="print only, do not append readings")
    args = ap.parse_args()

    if args.list:
        for name in list_probes():
            print(name)
        return 0

    targets: list[str] = []
    if args.probe:
        targets = [args.probe]
    elif args.all:
        targets = list_probes()
    else:
        ap.error("pass --probe <name>, --all, or --list")
        return 2

    tl = _tilelang_version(args.python)
    gpu = _gpu_name(args.python)
    print(f"python={args.python}\ntilelang={tl}\ngpu={gpu}\n")

    worst = 0
    for name in targets:
        rc, tail = run_one(name, args.budget, args.python, {})
        print(f"===== {name} (rc={rc}) =====")
        print(tail)
        if not args.no_record:
            append_readings(name, rc, tail, tl, gpu)
        if rc != 0:
            worst = rc
    return worst


if __name__ == "__main__":
    sys.exit(main())
