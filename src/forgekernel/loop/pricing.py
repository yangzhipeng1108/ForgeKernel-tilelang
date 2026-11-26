"""Lever pricing: the dossier contract signature and price-staleness re-pricing.

Extracted from ``loop/driver.py`` (see ``docs/driver-split-spec.md`` step 4). Two
concerns that share one vocabulary -- the structural signature a KEEP "moved an axis"
on, and the re-pricing that catches a lever whose price was read on an older tree.

The functions are byte-for-byte the ones that lived in ``driver``; the 411-test gate in
``tests/test_loop_driver.py`` proves it.

This section is **not** a leaf: its functions call back into ``driver`` for milestone
IO, evidence, cluster and toolkit helpers. To keep the import graph acyclic those calls
go through the thin lazy shims below -- each imports the real function from ``driver``
at call time, so there is no import-time cycle. The shims are behaviour-neutral (one
call, same signature) and exist only to break the cycle, not to change anything.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from forgekernel.inner.trajectory import gate_ms, load_records
from forgekernel.loop import meta_shapes
from forgekernel.loop.parse import JOB_END_KILLED, job_abend
# ``BackendError`` is defined in ``driver`` (idx ~160) before driver reaches its
# ``from forgekernel.loop.pricing import ...`` (idx ~3800), so this partial import
# resolves; it must be the real class because ``except BackendError`` is evaluated
# at runtime. Everything else driver-local goes through the lazy shims below.
from forgekernel.loop.driver import BackendError

if TYPE_CHECKING:
    from forgekernel.loop.driver import LoopContext


def _log(msg: str) -> None:
    print(f"[loop] {msg}", flush=True)


# -- lazy shims back into driver (break the import cycle; see module docstring) ----------
def archive_is_whole(log, task=""):  # noqa: F811
    from forgekernel.loop.driver import archive_is_whole as f
    return f(log, task)


def _read_milestone(ctx):  # noqa: F811
    from forgekernel.loop.driver import _read_milestone as f
    return f(ctx)


def _write_milestone(ctx, text):  # noqa: F811
    from forgekernel.loop.driver import _write_milestone as f
    return f(ctx, text)


def _on_cluster(records, ctx, cluster=None):  # noqa: F811
    from forgekernel.loop.driver import _on_cluster as f
    return f(records, ctx, cluster)


def _evidence_dir(ctx, create=False):  # noqa: F811
    from forgekernel.loop.driver import _evidence_dir as f
    return f(ctx, create)


def _bank_evidence(ctx, why):  # noqa: F811
    from forgekernel.loop.driver import _bank_evidence as f
    return f(ctx, why)


def _seed_pricing_toolkits(root, backend):  # noqa: F811
    from forgekernel.loop.driver import _seed_pricing_toolkits as f
    return f(root, backend)


def _fmt_pct(x):  # noqa: F811
    from forgekernel.loop.driver import _fmt_pct as f
    return f(x)


def _lever_spec_refused(ctx, spec):  # noqa: F811
    from forgekernel.loop.driver import _lever_spec_refused as f
    return f(ctx, spec)


def _persist_lever_table(ctx, table, best_ref, job, *a, **k):  # noqa: F811
    from forgekernel.loop.driver import _persist_lever_table as f
    return f(ctx, table, best_ref, job, *a, **k)


def _table_noise_pct(table):  # noqa: F811
    from forgekernel.loop.driver import _table_noise_pct as f
    return f(table)

#
# The count-free structural core of an orch dossier, per contract axis. Read by
# the KB axis harvest (`_axis_changes`): a KEEP "moved an axis" when this
# signature differs between the censused KEEPs of its lineage, not when the raw
# summary does. (It also gated the residual-ledger window, a late-plateau
# diagnosis mode since removed: on the two runs that reached its regime the
# window never opened — the `!` it keys on is written by wait-encoding choices
# as much as by structure — and the lever pricing round now answers "where is
# the remaining time" with a measurement instead of a teacher-written list.)
#
# Dossier axes that encode the milestone's declared contracts ([SYNC:] ->
# named-barrier verdicts + mbarrier rings, [GATHER:] -> supply order, [PHASE:]
# -> merge placement / commit depth). The other axes count things codegen
# moves without any structural difference — spin micro-loops, unresolved
# mbarrier objects, barrier widths, a copy-producer -> spin edge — and stayed
# `!` on the grouped-gemm pages after every real fix (an offline check: 11
# residual `!`, all counting axes). Billing them would hand the teacher
# compiler noise to write recipes for.
_CONTRACT_DIFF_AXES = ("nbar_verdicts", "rings", "supply", "commit", "cache_ops")
# cache_ops rides with the contract axes: a pricing probe
# put the cp.async `.BYPASS` bit at -22.2 % device time on a dsa_sparse_fwd
# best tree — a `!` on it is as structural as a missing ring. The axis is
# compared as the set of cache-MODE classes, not the verbatim opcode set: the
# same probe read the `.L2::256B` prefetch hint (`LTC256B`) at ~0 and reachable
# only through inline PTX, and a closed-source ref carries it plus its
# `CREATEPOLICY` descriptors — verbatim equality would bill exactly the kernel
# that already has the 22 % bit right.
_L2_HINT_RE = re.compile(r"\.LTC\d+B")
_CACHE_DESCRIPTOR_FAMS = {"CREATEPOLICY"}


def _cache_class(op: str) -> str | None:
    """A cache-policy opcode with its L2 prefetch hint stripped; None for the
    policy-descriptor instructions (they carry no load/store of their own)."""
    if op.split(".")[0] in _CACHE_DESCRIPTOR_FAMS:
        return None
    return _L2_HINT_RE.sub("", op)
# Even those axes carry counts in their printed form (a verdict repeated per
# barrier id, "copies 40" vs "copies 16", one DEPBAR line per loop region), so
# a plain equality read them as `!` on the grouped-gemm pages (an offline
# replay) while both kernels carry the same contracts. The signature is
# count-free per axis: the SET of barrier verdict kinds, the ring
# set, the supply KIND (serial gather vs prefetched), the DEPBAR depth set per
# role. Merge placement is a nomination axis (§2g: a ledger reading only earns a nomination) and stays out.
_SUPPLY_TIGHT_RE = re.compile(r"tight-chained copies (\d+)")


def _contract_signature(summary: dict) -> dict:
    """The count-free structural core of a dossier summary, per contract axis."""
    s = json.loads(json.dumps(summary or {}))
    kinds = sorted({re.sub(r"\s*width=\S+", "", v).strip()
                    for v in (s.get("nbar_verdicts") or [])})
    supply = []
    for line in (s.get("supply") or []):
        # orch_tools now emits the class itself ("...; serial-gather" /
        # "...; prefetched"); summaries banked before that carried the raw
        # counted line, which the tight-chain count still classifies.
        if "serial-gather" in line or "prefetched" in line:
            supply.append("serial-gather" if "serial-gather" in line else "prefetched")
            continue
        m = _SUPPLY_TIGHT_RE.search(line)
        tight = int(m.group(1)) if m else 0
        supply.append("serial-gather" if tight > 0 else "prefetched")
    depths = {}
    for line in (s.get("commit") or []):
        role, _, rest = line.partition(":")
        depths.setdefault(role.strip(), set()).update(re.findall(r"LE (0x[0-9a-fA-F]+)", rest))
    return {"nbar_verdicts": kinds,
            "rings": sorted(s.get("rings") or []),
            "supply": sorted(set(supply)),
            "commit": {r: sorted(v) for r, v in sorted(depths.items())},
            "cache_ops": sorted({c for c in map(_cache_class, s.get("cache_ops") or [])
                                 if c})}
_LEVER_LINE_RE = re.compile(
    r"^LEVER (?P<name>\S+) add=(?P<add>[+-]?[\d.]+%|n/a) remove=(?P<remove>[+-]?[\d.]+%|n/a) "
    r"add_ms=(?P<add_ms>\S+) remove_ms=(?P<remove_ms>\S+) class=(?P<cls>\S+) "
    r"spin_add=(?P<spin_add>\S+) spin_all_minus=(?P<spin_rem>\S+)", re.M)
_STACK_LINE_RE = re.compile(
    r"^STACK all=(?P<all>[+-]?[\d.]+%|n/a) sum_singles=(?P<sum>[+-]?[\d.]+%|n/a) "
    r"coupling=(?P<coupling>[+-]?[\d.]+%|n/a)"
    r"(?: all_pos=(?P<all_pos>[+-]?[\d.]+%|n/a) all_pos_levers=(?P<pos_levers>\S+) "
    r"dropped=(?P<dropped>\S+))?", re.M)
# the teacher's own subsets (and the tool's automatic all-pos), one line each
_VARIANT_LINE_RE = re.compile(
    r"^VARIANT (?P<name>\S+) levers=(?P<levers>\S+) delta=(?P<delta>[+-]?[\d.]+%|n/a) "
    r"ms=(?P<ms>\S+) spin=(?P<spin>\S+) q=(?P<q>.*)$", re.M)
_PRICE_OK_RE = re.compile(r"^PRICE-OK levers=(\d+) rounds=(\d+) base_ms=(\S+) all_ms=(\S+)"
                          r"(?: spin_base=(\S+) spin_all=(\S+) tensor_base=(\S+) tensor_all=(\S+))?", re.M)
_PRICE_FAIL_RE = re.compile(r"^PRICE-FAIL reason=(.*)$", re.M)
_PRICE_NOISE_RE = re.compile(r"^PRICE noise_floor=([\d.]+)% metric=(\S+)", re.M)
_PRICE_TIMED_RE = re.compile(r"^PRICE-TIMED levers=", re.M)
# a group card priced on its representative workload only (lever_pricing_run
# WORKLOAD_SCOPE_ENV): the table is a reading on that shape, the gate the group's geomean
_PRICE_SCOPE_RE = re.compile(r"^PRICE-SCOPE workloads=(\S+) rep=(\S+) of=(\S+)", re.M)
_ABLATE_OK_RE = re.compile(r"^ABLATE-OK ablations=", re.M)
_TOOL_TRACEBACK = "Traceback (most recent call last)"
# the ablation pass: one line per ablation (an edit that removes a mechanism the
# tree has, timed on the NCU kernel clock so a broken kernel still reads)
_ABLATE_LINE_RE = re.compile(
    r"^ABLATE (?P<name>\S+) delta=(?P<delta>[+-]?[\d.]+%|n/a) ncu_ns=(?P<ns>\S+) "
    r"base_ncu_ns=(?P<base_ns>\S+) correctness=(?P<corr>\S+) device_delta=(?P<dd>[+-]?[\d.]+%|n/a) "
    # the tool prints the kernel as one token; a log from before it did may
    # carry a demangled name with spaces — the field runs to the next key
    r"kernel=(?P<kernel>.+?)(?: status=(?P<status>\S+))? h=(?P<h>.*)$", re.M)
_ABLATE_NOISE_RE = re.compile(r"^PRICE ablation_noise_floor=([\d.]+)% metric=(\S+)", re.M)
_ABLATE_FAIL_RE = re.compile(r"^ABLATE-FAIL reason=(.*)$", re.M)
# The seeded pricing tool's verdict when a timed-out variant left the GPU held
# (lever_pricing_run.NODE_DIRTY_MARKER): PRICE-FAIL / ABLATE-FAIL reason=node-dirty
# and a NODE-DIRTY line. Readings after that line are the orphan's share of the
# GPU, not the variant's; the design is re-priced once.
_NODE_DIRTY_MARKER = "NODE-DIRTY"


def _price_log_verdict(log: str) -> str | None:
    """The closing verdict a lever_pricing_run log carries, None when it carries
    none. The tool ends every run with one of PRICE-OK / PRICE-FAIL / ABLATE-OK /
    ABLATE-FAIL; PRICE-TIMED (the timing copy of the table, roles pending) counts
    only when the log says how the job ended — the archive is whole (the role
    pass died on its own and the entry script closed), or the job was KILLED
    after the timing copy (observed: the per-variant `stall_price_run --pm` role
    pass outlived TASK_TIMEOUT on two B200 runs, `timeout --signal=KILL` took
    the job at rc=137, and a killed entry script never prints its closing line;
    the table those 4 jobs printed was complete and consistent, and refusing it
    bought the same table again, lost it the same way, and diagnosed blind).
    Not when the log simply ends there, which is an archive read between the
    two copies. A log with none of these is not a table, whatever LEVER rows it
    holds: the rows are the first copy, cut wherever the archiver was."""
    for name, rx in (("PRICE-OK", _PRICE_OK_RE), ("PRICE-FAIL", _PRICE_FAIL_RE),
                     ("ABLATE-OK", _ABLATE_OK_RE), ("ABLATE-FAIL", _ABLATE_FAIL_RE)):
        if rx.search(log or ""):
            return name
    if _PRICE_TIMED_RE.search(log or "") and (archive_is_whole(log) or _price_role_pass_killed(log)):
        return "PRICE-TIMED"
    return None


def _price_role_pass_killed(log: str) -> str | None:
    """The kill sentence when ``log`` holds the timing copy of a price table
    (PRICE-TIMED) and then the job's kill evidence (``parse.job_abend``), None
    otherwise. The timing copy is printed once every timed variant has its
    rounds; whatever the kill took was the role pass (spin / tensor columns) or
    the ablation pass — the LEVER / VARIANT / STACK rows stand as timed."""
    if not _PRICE_TIMED_RE.search(log or ""):
        return None
    end = job_abend(log or "")
    return end.reason if end.kind == JOB_END_KILLED else None


def _price_log_settled(be, log: str, what: str) -> str:
    """``log`` once it carries a verdict: as read, or the job's archive read
    again: a first read that stopped short of the verdict may be complete on a
    second read — a re-read is the free recovery. Unchanged when the backend cannot re-read (test fakes)
    or the re-read is no longer."""
    if _price_log_verdict(log) is not None:
        return log
    job = str(getattr(be, "last_job_id", "") or "")
    reread = getattr(be, "reread_log", None)
    if not job or not callable(reread):
        return log
    _log(f"{what}: job {job} log carries no PRICE/ABLATE verdict — re-reading the archive")
    try:
        again = reread(job)
    except BackendError as e:
        _log(f"{what}: re-read of job {job} failed ({e})")
        return log
    return again if len(again or "") > len(log or "") else log


def _price_log_lost_reason(log: str) -> str:
    """Why a verdict-less pricing log is not a table, for the log and the teacher."""
    if _TOOL_TRACEBACK in (log or ""):
        return "the pricing tool crashed"
    return "archive incomplete or the tool never ran"


def _node_dirty_reason(fail: str | None, log: str) -> str | None:
    if fail and fail.startswith("node-dirty"):
        return f"PRICE-FAIL {fail}"
    if (m := _ABLATE_FAIL_RE.search(log or "")) and m.group(1).strip().startswith("node-dirty"):
        return f"ABLATE-FAIL {m.group(1).strip()}"
    if _NODE_DIRTY_MARKER in (log or ""):
        return "the job reported NODE-DIRTY"
    return None


def _lever_price_job(be, branch: str, cmd: str, what: str,
                     rerun: bool = True) -> "tuple[str, str]":
    """One pricing job, settled: (log, job id). Re-bought once (``rerun``) when its GPU
    was dirty or its log still carries no verdict (a crashed tool is not re-bought). BackendError
    propagates."""
    attempts = 2 if rerun else 1
    for attempt in range(attempts):
        log = be.submit_gpu(branch, cmd)
        log = _price_log_settled(be, log, what)
        _, _, fail = parse_lever_prices(log)
        last = attempt == attempts - 1
        if not last and _node_dirty_reason(fail, log):
            # the job's GPU held an orphan (a hung variant's launch target): the
            # readings after it are the orphan's share of the GPU — the same design
            # is re-priced once, no design turn spent
            _log(f"{what}: {_node_dirty_reason(fail, log)} — re-pricing the same "
                 f"design once")
            continue
        if not last and _price_log_verdict(log) is None and _TOOL_TRACEBACK not in log:
            # still no verdict after the archive was re-read: the job is lost to the
            # archiver, not to the design — the same spec is bought once more (a tool
            # that crashed would crash again: no resubmit for that)
            _log(f"{what}: job {getattr(be, 'last_job_id', '') or '?'} log still carries no "
                 f"PRICE/ABLATE verdict — resubmitting the same design once")
            continue
        break
    return log, str(getattr(be, "last_job_id", "") or "")


def _lever_price_submit(ctx: LoopContext, be, branch: str, cmd: str, what: str,
                        rerun: bool = True) -> "tuple[str, str]":
    """The pricing log the parser reads, and the job it cites: one job, or -- a group card
    listing its meta workloads (loop/meta_shapes.py) -- one job per workload
    (``--workload <name>``) merged into one table. The representative's own failure (an
    edit that does not apply, the job environment, a lost archive) is the table's and
    comes back as its raw log, handled as on a one-workload card; another workload's is
    named in the merged table and leaves it out. BackendError on the representative's
    job propagates. The workloads are priced one after the other, the representative
    first."""
    shapes = meta_shapes.card_meta_workloads(ctx.workspace)
    if not shapes:
        return _lever_price_job(be, branch, cmd, what, rerun)

    def price(wl):
        if wl != shapes[0] and ctx.now() > ctx.deadline_epoch - 1800:
            return ("budget",)
        try:
            # the role pass on the representative alone: on the others its columns were
            # never read and took a third of each job
            roles = "" if wl == shapes[0] else " --no-roles"
            log, job = _lever_price_job(be, branch, f"{cmd}{roles} --workload {wl}",
                                        f"{what} [{wl}]", rerun)
        except BackendError as e:
            return ("error", e)
        return ("log", log, job)

    results: dict = {shapes[0]: price(shapes[0])}
    r = results[shapes[0]]
    if r[0] == "error":
        raise r[1]
    _, _, fail = parse_lever_prices(r[1])
    if _price_log_verdict(r[1]) is None or fail:
        return r[1], r[2]
    for wl in shapes[1:]:
        results[wl] = price(wl)
    per: list = []
    missing: dict = {}
    for wl in shapes:
        r = results.get(wl) or ("budget",)
        if r[0] == "budget":
            missing[wl] = "budget"
            continue
        if r[0] == "error":
            missing[wl] = f"job failed ({r[1]})"[:80]
            continue
        _, log, job = r
        _, _, fail = parse_lever_prices(log)
        if _price_log_verdict(log) is None or fail:
            missing[wl] = fail or "no verdict"
            _log(f"{what}: workload {wl} has no table ({missing[wl]}) — merged without it")
            continue
        per.append((wl, log, job))
    return meta_shapes.merge_lever_logs(per, missing=missing), per[0][2]


# ablations per pricing job / per JSON verdict probe (== lever_pricing_run.MAX_ABLATIONS)
_LEVER_MAX_ABLATIONS = 4
_LEVER_NOISE_DEFAULT_PCT = 0.5   # == lever_pricing_run.NOISE_FLOOR_PCT
_LEVER_SPEC_NAME = "levers.json"
_LEVER_PRICING_CMD = "cmd:python3 notes/lever_pricing_run.py notes/levers.json"
# design-turn re-prompts after a PRICE-FAIL (an `old` that does not apply)
_LEVER_PRICE_RETRIES = 1
# the teacher's own subsets per pricing job (each is one more variant timed)
_LEVER_MAX_SUBSETS = 6


def _pct_or_none(tok: str):
    return None if tok in (None, "n/a") else float(tok.rstrip("%"))


def parse_lever_prices(log: str) -> "tuple[list, dict | None, str | None]":
    """(levers, stack, fail_reason) from a lever_pricing_run log."""
    # the run prints the table twice — once when timing is done (PRICE-TIMED),
    # once with the role columns filled (PRICE-OK); a job killed in between
    # still left the first copy. Keep the last copy of each lever / STACK line.
    by_name: dict = {}
    for m in _LEVER_LINE_RE.finditer(log):
        by_name[m["name"]] = {
            "name": m["name"], "add_pct": _pct_or_none(m["add"]),
            "remove_pct": _pct_or_none(m["remove"]),
            "add_ms": None if m["add_ms"] == "None" else _float_or_none(m["add_ms"]),
            "remove_ms": None if m["remove_ms"] == "None" else _float_or_none(m["remove_ms"]),
            "cls": m["cls"], "spin_add_ns": _float_or_none(m["spin_add"]),
            "spin_all_minus_ns": _float_or_none(m["spin_rem"])}
    levers = list(by_name.values())
    by_variant: dict = {}
    for m in _VARIANT_LINE_RE.finditer(log):
        by_variant[m["name"]] = {
            "kind": "variant", "name": m["name"],
            "levers": [x for x in m["levers"].split(",") if x],
            "delta_pct": _pct_or_none(m["delta"]),
            "ms": None if m["ms"] == "None" else _float_or_none(m["ms"]),
            "spin_ns": _float_or_none(m["spin"]), "question": m["q"].strip()}
    levers += list(by_variant.values())
    by_ablation: dict = {}
    for m in _ABLATE_LINE_RE.finditer(log):
        by_ablation[m["name"]] = {
            "kind": "ablation", "name": m["name"], "delta_pct": _pct_or_none(m["delta"]),
            "ncu_ns": _float_or_none(m["ns"]), "base_ncu_ns": _float_or_none(m["base_ns"]),
            "correctness": m["corr"], "device_delta_pct": _pct_or_none(m["dd"]),
            "kernel": m["kernel"], "status": m["status"] or ("ok" if m["delta"] != "n/a" else "n/a"),
            "hypothesis": m["h"].strip()}
    if by_ablation:
        an = None
        for an in _ABLATE_NOISE_RE.finditer(log):
            pass
        for row in by_ablation.values():
            row["noise_pct"] = _float_or_none(an.group(1)) if an else _LEVER_NOISE_DEFAULT_PCT
    levers += list(by_ablation.values())
    sm = None
    for sm in _STACK_LINE_RE.finditer(log):
        pass
    stack = None
    if sm:
        stack = {"all_pct": _pct_or_none(sm["all"]), "sum_singles_pct": _pct_or_none(sm["sum"]),
                 "coupling_pp": _pct_or_none(sm["coupling"]),
                 "all_pos_pct": _pct_or_none(sm["all_pos"]),
                 "all_pos_levers": [x for x in (sm["pos_levers"] or "").split(",") if x and x != "none"],
                 "dropped": [x for x in (sm["dropped"] or "").split(",") if x and x != "none"]}
        om = None
        for om in _PRICE_OK_RE.finditer(log):
            pass
        if om:
            stack.update({"base_ms": _float_or_none(om.group(3)), "all_ms": _float_or_none(om.group(4)),
                          "spin_base_ns": _float_or_none(om.group(5)), "spin_all_ns": _float_or_none(om.group(6)),
                          "tensor_base": _float_or_none(om.group(7)), "tensor_all": _float_or_none(om.group(8))})
        nm = None
        for nm in _PRICE_NOISE_RE.finditer(log):
            pass
        if nm:
            stack.update({"noise_pct": _float_or_none(nm.group(1)), "metric": nm.group(2)})
        sc = None
        for sc in _PRICE_SCOPE_RE.finditer(log):
            pass
        if sc:
            stack["workload_scope"] = {"workloads": sc.group(1), "rep": sc.group(2), "of": sc.group(3)}
    meta_shapes.attach_shape_rows(log, levers, stack)   # a table merged over workloads
    fm = _PRICE_FAIL_RE.search(log)
    return levers, stack, (fm.group(1).strip() if fm else None)


def _float_or_none(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None

#
# A price is a reading on one tree. The pricing round banks its table under the
# best sha it ran on, the teacher writes those numbers into the milestone
# (`price +25.59% via lever:regsplit_184_120 [MEASURED, job <id>]`), and the
# student walks the rung several trees later — nothing compared the two trees.
# A GDN run priced a register-split lever at +25.6% on its first green
# (the aux role spilling 8.9M sectors); the teacher wrote it in as the last rung
# of a mandatory ladder; two KEEPs later the aux spill was gone and the same
# lever read −11.6% (it serialises every wgmma), the student walked it anyway,
# the M0 gate row and the next lever were built on that tree, and a full meta
# step went to discovering the reversal. The check below runs where the tree
# changes: every KEEP that takes the watermark re-prices the milestone's cited
# levers whose price came from another tree (one job, base / +each only, the
# recipes banked with the table), marks each citation in the milestone with the
# verdict, and suspends a reversed / faded rung for the student until the
# teacher's next entry reads the revision and rewrites the rung.
_LEVER_NAME = r"[A-Za-z0-9_][A-Za-z0-9_\-]*(?:\.[A-Za-z0-9_\-]+)*"
_LEVER_CITE_RE = re.compile(r"\blever\s*[:：]\s*`?(" + _LEVER_NAME + r")`?(?![A-Za-z0-9_\-])")
# a job pointer on the citation's line, after the citation: `[MEASURED, job <id>]`
_CITE_JOB_RE = re.compile(r"job\s*\**\s*([0-9]{4,})")
_REPRICE_TAG = "RE-PRICED"
_REVISIONS_NAME = "price_revisions.jsonl"
# verdicts that change what a citation may be used for
_REVISION_SUSPENDS = ("reversed", "faded")
_REVISION_FLAGGED = _REVISION_SUSPENDS + ("emerged", "drifted", "unverifiable")


def _stamp_price_row(row: dict, best_ref: str, job: str) -> dict:
    row.setdefault("priced_on", best_ref)
    row.setdefault("job", job)
    return row


def _priced_best(priced) -> str | None:
    """The tree a reopen decided at this entry would be applied to."""
    return (priced.best_sha or priced.head_sha) if priced is not None else None


def _cited_levers(text: str) -> list:
    """Lever names the milestone cites as prices (`lever:<name>`), in order, once each."""
    seen: list = []
    for m in _LEVER_CITE_RE.finditer(text or ""):
        name = m.group(1)
        if name and name not in seen:
            seen.append(name)
    return seen


def _cited_price_jobs(text: str, name: str) -> list:
    """Job pointers written on the same line as a `lever:<name>` citation (the
    `[MEASURED, job N]` tag the teacher copied from the table) — they name the
    reading the milestone actually cites."""
    jobs: list = []
    for line in (text or "").splitlines():
        for m in re.finditer(r"\blever\s*[:：]\s*`?" + re.escape(name) + r"`?(?![A-Za-z0-9_\-])", line):
            # only the teacher's own tag counts: the driver's [RE-PRICED … job N]
            # marker after it names a re-price, and reading THAT as the citation
            # would move the baseline to the latest re-price — the drift the
            # cited-reading rule exists to stop
            after = line[m.end():m.end() + 120]
            after = after.split("[" + _REPRICE_TAG, 1)[0]
            for j in _CITE_JOB_RE.findall(after):
                if j not in jobs:
                    jobs.append(j)
    return jobs


def _lever_price_history(ctx: LoopContext, name: str) -> list:
    """Every banked table that priced ``name``, oldest first: dicts of
    ``best_ref`` / ``job`` / ``row`` (the lever or variant row, provenance
    stamped) / ``spec`` (the recipe banked with that table; None for tables
    from before recipes were banked)."""
    ev = _evidence_dir(ctx, create=False)
    if ev is None or not (ev / "lever_prices.jsonl").is_file():
        return []
    out: list = []
    try:
        for line in (ev / "lever_prices.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            row = next((lv for lv in rec.get("table") or []
                        if lv.get("kind") in ("lever", "variant") and lv.get("name") == name), None)
            if row is None:
                continue
            ref, job = str(rec.get("best_ref") or ""), str(rec.get("job") or "")
            spec = next((sp for sp in rec.get("spec_levers") or [] if sp.get("name") == name), None)
            stack = next((t for t in rec.get("table") or [] if t.get("kind") == "stack"), None)
            out.append({"best_ref": ref, "job": job, "reprice": bool(rec.get("reprice")),
                        "base_ms": (stack or {}).get("base_ms"),
                        "row": _stamp_price_row(dict(row), ref, job), "spec": spec})
    except (OSError, ValueError) as e:
        _log(f"price-check: could not read lever_prices.jsonl ({e})")
        return []
    return out


def _ms_of_tree(ctx: LoopContext, sha: str) -> float | None:
    """The gate-window reading of the KEEP that produced ``sha`` (same-cluster
    rows only), or None when no record owns it."""
    if not sha:
        return None
    try:
        rows = _on_cluster(load_records(ctx.status_path), ctx)
    except (OSError, ValueError):
        return None
    for r in reversed(rows):
        if r.sha == sha and r.verdict == "KEEP":
            return gate_ms(r, ctx.gate_window)
    return None


def _price_legs(row: dict) -> "tuple[float | None, float | None]":
    """(add, remove) of a lever row; a subset row prices as (delta, None)."""
    if row.get("kind") == "variant":
        return row.get("delta_pct"), None
    return row.get("add_pct"), row.get("remove_pct")


def _reprice_verdict(before: dict, now: dict, noise_pct: float) -> "tuple[str, str]":
    """(verdict, note) for a lever priced ``before`` on one tree and ``now`` on
    another. Read on the add leg (the reading a single-lever citation is), on
    the remove leg when the older table had no add. ``noise_pct`` is the newer
    job's own noise floor."""
    def value(row):
        # what having the lever is worth on that tree: its add leg, else the
        # negated remove leg (a landed lever is priced by taking it out)
        add, rem = _price_legs(row)
        if add is not None:
            return add, "add"
        return (None, "add") if rem is None else (-rem, "-remove")
    old, b_leg = value(before)
    new, n_leg = value(now)
    if old is None or new is None:
        return "unmeasured", "no add / remove reading on one of the two trees"
    big = lambda x: abs(x) > noise_pct  # noqa: E731
    span = f"{b_leg} {old:+.2f}% -> {n_leg} {new:+.2f}% (noise {noise_pct:.2f}%)"
    if big(old) and big(new) and (old > 0) != (new > 0):
        return "reversed", span + ": the sign flipped — on this tree the lever costs what it paid before"
    if big(old) and not big(new):
        return "faded", span + ": inside the noise band on this tree — the lever no longer pays"
    if not big(old) and big(new):
        return "emerged", span + ": a null reading became a real one on this tree"
    if big(old) and big(new) and abs(new - old) > max(2 * noise_pct, 0.25 * abs(old)):
        return "drifted", span + ": same sign, different size — cite the new reading"
    return "confirmed", span


def _append_revisions(ctx: LoopContext, revisions: list) -> None:
    if not revisions:
        return
    ev = _evidence_dir(ctx, create=True)
    if ev is None:
        return
    try:
        with (ev / _REVISIONS_NAME).open("a", encoding="utf-8") as fh:
            for rv in revisions:
                fh.write(json.dumps(rv, ensure_ascii=False) + "\n")
    except OSError as e:
        _log(f"price-check: could not persist {_REVISIONS_NAME}: {e}")
    else:
        _bank_evidence(ctx, "price revisions")


def _unretired_revisions(ctx: LoopContext) -> list:
    """The standing revisions, newest per lever."""
    ev = _evidence_dir(ctx, create=False)
    if ev is None or not (ev / _REVISIONS_NAME).is_file():
        return []
    latest: dict = {}
    try:
        for line in (ev / _REVISIONS_NAME).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rv = json.loads(line)
                if not rv.get("retired"):
                    latest[rv["lever"]] = rv
    except (OSError, ValueError) as e:
        _log(f"price-check: could not read {_REVISIONS_NAME} ({e})")
        return []
    return list(latest.values())


def _retire_revisions(ctx: LoopContext) -> None:
    """Mark every standing revision retired: a fresh table on the current best
    supersedes them, and the teacher has read them at this entry."""
    ev = _evidence_dir(ctx, create=False)
    p = (ev / _REVISIONS_NAME) if ev is not None else None
    if p is None or not p.is_file():
        return
    try:
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for rv in rows:
            rv["retired"] = True
        p.write_text("".join(json.dumps(rv, ensure_ascii=False) + "\n" for rv in rows),
                     encoding="utf-8")
    except (OSError, ValueError) as e:
        _log(f"price-check: could not retire {_REVISIONS_NAME} ({e})")


def _legs_text(d: dict | None) -> str:
    if not d:
        return "n/a"
    if d.get("delta_pct") is not None and d.get("add_pct") is None:
        return f"delta {_fmt_pct(d.get('delta_pct'))}"
    return f"add {_fmt_pct(d.get('add_pct'))} / remove {_fmt_pct(d.get('remove_pct'))}"


def _reprice_marker(rv: dict) -> str:
    """The in-milestone tag a re-priced citation gets, after its [MEASURED] tag."""
    tag = (f"[{_REPRICE_TAG} @{str(rv['priced_on_now'])[:10]}"
           + (f" job {rv['job_now']}" if rv.get("job_now") else "")
           + f": {_legs_text(rv.get('now'))} — {rv['verdict'].upper()}")
    if rv["verdict"] in _REVISION_SUSPENDS:
        tag += ("; SUSPENDED by driver — the rung citing this price may not be landed "
                "until the teacher revises it")
    elif rv["verdict"] == "unverifiable":
        tag += f"; {rv['note']}"
    return tag + "]"


def _mark_lever_citations(text: str, name: str, marker: str) -> str:
    """Put ``marker`` after every `lever:<name>` citation in ``text`` (after its
    [MEASURED …] tag when one follows), replacing any earlier RE-PRICED marker
    on that citation — one standing verdict per citation, the latest."""
    pat = re.compile(r"(\blever\s*[:：]\s*`?" + re.escape(name) + r"`?(?![A-Za-z0-9_\-]))"
                     r"((?:\s*\[MEASURED[^\]\n]*\])?)((?:\s*\[" + _REPRICE_TAG + r"[^\]\n]*\])*)")
    return pat.sub(lambda m: f"{m.group(1)}{m.group(2)} {marker}", text)


def _revision_line(rv: dict) -> str:
    return (f"`{rv['lever']}`: was {_legs_text(rv.get('before'))} @`{str(rv['priced_on_before'])[:10]}`"
            + (f" (job {rv['job_before']})" if rv.get("job_before") else "")
            + f" → now {_legs_text(rv.get('now'))} @`{str(rv['priced_on_now'])[:10]}`"
            + (f" (job {rv['job_now']})" if rv.get("job_now") else "")
            + f" → **{rv['verdict'].upper()}** ({rv['note']})")


def _price_revision_note(ctx: LoopContext) -> str:
    """Student-facing: the standing revisions of milestone-cited lever prices."""
    revs = [rv for rv in _unretired_revisions(ctx) if rv.get("verdict") in _REVISION_FLAGGED]
    if not revs:
        return ""
    lines = ["\n## Price-tag review (driver: whenever the best tree moves, the lever price tags "
             "the milestone cites are re-measured on the new tree)"]
    for rv in revs:
        lines.append("- " + _revision_line(rv))
        if rv["verdict"] in _REVISION_SUSPENDS:
            lines.append("  → the milestone clauses / ladder rungs citing it have been marked "
                         "`SUSPENDED` by the driver: **do not land them this round and do not base a "
                         "change on them**; the teacher re-checks them at its next entry.")
        elif rv["verdict"] == "unverifiable":
            lines.append("  → this price tag cannot be re-measured on the current tree: treat the "
                         "clauses citing it as **unpriced**, not as grounds to land anything.")
        else:
            lines.append("  → the reading on the new tree is what counts; read every place that "
                         "quotes the old number with the new value.")
    return "\n".join(lines) + "\n"


def _price_revision_teacher_block(revs: list) -> str:
    lines = ["\n## Price-tag review (measured by the driver in the inner loop: when the best tree "
             "changes, the levers the milestone cites are re-measured)",
             "These levers' price tags were written into the milestone (`lever:<name>` citations); "
             "below are their re-measurements on the later best trees. "
             "**Citations that came back REVERSED / FADED have been marked `SUSPENDED` in the "
             "milestone by the driver, and the student is forbidden to land them** — "
             "this diagnosis / patch must dispose of every one: retract the rung, rewrite it at the "
             "new price, or reopen at the new price; "
             "no SUSPENDED mark may be left in the milestone."]
    for rv in revs:
        lines.append("- " + _revision_line(rv))
    return "\n".join(lines) + "\n"


def _reprice_cited_levers(ctx: LoopContext, rec) -> list:
    """After a KEEP took the watermark: re-price, on that tree, every lever the
    milestone cites whose price was read on another tree; bank the readings,
    mark each citation with its verdict, suspend reversed / faded ones for the
    student. Returns the revision records written (possibly empty)."""
    if not (ctx.reprice_on_new_best and ctx.lever_pricing_rounds > 0 and ctx.stall_pricing):
        return []
    if not rec.sha:
        return []
    ev = _evidence_dir(ctx, create=False)
    if ev is None or not (ev / "lever_prices.jsonl").is_file():
        return []
    text = _read_milestone(ctx)
    new_ms = gate_ms(rec, ctx.gate_window)
    stale: list = []          # (name, the cited reading, the latest reading)
    skipped: list = []
    for name in _cited_levers(text):
        hist = _lever_price_history(ctx, name)
        if not hist:
            continue          # cited but never priced here: the reopen gate's business
        latest = hist[-1]
        if not latest["best_ref"] or latest["best_ref"] == rec.sha:
            continue          # already read on this very tree
        # the reading the milestone cites: the table whose job the citation
        # names, else the oldest — a verdict against the latest re-price would
        # let a slow drift read "confirmed" at every step
        jobs = _cited_price_jobs(text, name)
        cited = next((h for h in hist if h["job"] and h["job"] in jobs), hist[0])
        # bound the job count: a best that moved by less than the margin since
        # the lever was last read is not re-read (the teacher's entry prices it)
        last_ms = _ms_of_tree(ctx, latest["best_ref"]) or latest.get("base_ms")
        if (new_ms is not None and last_ms
                and new_ms > last_ms * (1.0 - ctx.reprice_min_gain_pct / 100.0)):
            skipped.append(name)
            continue
        stale.append((name, cited, latest))
    if skipped:
        _log(f"iter {rec.iter}: new best {rec.sha[:10]} — cited lever price(s) {skipped} were read "
             f"on a tree within {ctx.reprice_min_gain_pct:g}% of this one; not re-pricing them")
    if not stale:
        return []
    names = [n for n, _, _ in stale]
    if ctx.now() > ctx.deadline_epoch - 3600:
        _log(f"iter {rec.iter}: new best {rec.sha[:10]} — the milestone cites lever price(s) "
             f"{names} read on other trees, but < 60 min of budget left; not re-pricing")
        return []
    be = ctx.backend
    revisions: list = []

    def revise(name, cited, verdict, note, now_row=None, job="", landed=False):
        keys = ("add_pct", "remove_pct", "delta_pct", "cls")
        revisions.append({
            "ts": ctx.now(), "iter": rec.iter, "lever": name,
            "priced_on_before": cited["best_ref"], "job_before": cited["job"],
            "before": {k: cited["row"].get(k) for k in keys},
            "priced_on_now": rec.sha, "job_now": job, "landed": landed,
            "now": ({k: now_row.get(k) for k in keys} if now_row else None),
            "verdict": verdict, "note": note, "retired": False})

    # the recipes: a lever without a banked recipe cannot be re-priced; each is
    # priced ALONE (in_stack off): the citation is a single-add reading, and
    # unrelated cited levers need not compose. A lever the student has since
    # landed (its `new` text is in the tree, its `old` is not) is priced by
    # taking it out: the reversed recipe's add is the cost of removing it.
    todo: list = []
    for name, cited, latest in stale:
        spec = latest.get("spec") or cited.get("spec")
        if cited["row"].get("kind") == "variant":
            revise(name, cited, "unverifiable", "a subset row; only single levers are re-priced")
            continue
        if not spec or not spec.get("edits"):
            revise(name, cited, "unverifiable",
                   "no edit recipe was banked with its price (priced before recipes were banked)")
            continue
        lv = {"name": name, "edits": spec["edits"], "in_stack": False}
        landed = False
        if fail := _lever_spec_refused(ctx, {"levers": [lv], "kernel_file": str(ctx.kernel_file)}):
            undo = {"name": name, "in_stack": False,
                    "edits": [{"old": e["new"], "new": e["old"]} for e in reversed(spec["edits"])]}
            if _lever_spec_refused(ctx, {"levers": [undo], "kernel_file": str(ctx.kernel_file)}):
                revise(name, cited, "unverifiable",
                       f"its edit no longer applies on {rec.sha[:10]} ({fail})")
                continue
            lv, landed = undo, True
        todo.append((name, cited, lv, landed, spec))
    if todo:
        job, parsed, stack, fail_note = "", [], None, ""
        pre_head = be.head()
        try:
            be.reset(rec.sha)
            _seed_pricing_toolkits(Path(ctx.workspace), be)
            spec_path = Path(ctx.workspace) / "notes" / _LEVER_SPEC_NAME
            # a group card re-prices on the workloads its table was priced on, merged alike
            shapes = meta_shapes.card_meta_workloads(ctx.workspace)
            spec = {"levers": [lv for _, _, lv, _, _ in todo], "variants": [],
                    "rounds": meta_shapes.rounds_for(ctx.lever_pricing_rounds, len(shapes)),
                    "roles": False, "kernel_file": str(ctx.kernel_file)}
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
            be.commit(f"meta-reprice: {len(todo)} cited lever(s) on {rec.sha[:10]}")
            branch = f"{ctx.branch_prefix}-reprice"
            be.push_branch(branch)
            _log(f"iter {rec.iter}: new best {rec.sha[:10]} — re-pricing {len(todo)} milestone-cited "
                 f"lever(s) [{', '.join(n + (' (landed: by removal)' if ld else '') for n, _, _, ld, _ in todo)}] "
                 f"whose price was read on another tree ("
                 + (f"one job per workload of {', '.join(shapes)}" if shapes else "one job")
                 + f": base / +each x {spec['rounds']}, no role pass)")
            log, job = _lever_price_submit(ctx, be, branch, _LEVER_PRICING_CMD,
                                           f"iter {rec.iter}: re-price", rerun=False)
            parsed, stack, fail = parse_lever_prices(log)
            if fail:
                _log(f"iter {rec.iter}: re-price PRICE-FAIL {fail}")
                parsed, fail_note = [], f"the re-price job refused the recipe ({fail})"
            elif _price_log_verdict(log) is None:
                # not a table (see _price_log_verdict): the cited prices stay
                # unverified rather than being "re-read" off a cut log
                why = _price_log_lost_reason(log)
                _log(f"iter {rec.iter}: re-price job {job or '?'} log carries no PRICE verdict "
                     f"({why}) — the cited prices stay unverified")
                parsed, fail_note = [], f"the re-price job's log carries no PRICE verdict ({why})"
            elif (killed := _price_role_pass_killed(log)):
                _log(f"iter {rec.iter}: re-price job {job or '?'}: {killed} after the timing "
                     f"table (PRICE-TIMED) — re-priced from the timing copy")
        except BackendError as e:
            _log(f"iter {rec.iter}: re-price job failed ({e}) — the cited prices stay unverified")
            parsed, fail_note = [], f"the re-price job failed ({e})"
        finally:
            try:
                be.reset(pre_head)
            except BackendError as e:
                _log(f"iter {rec.iter}: could not restore head {pre_head[:10]} after the re-price ({e})")
        by_name = {lv.get("name"): lv for lv in parsed}
        landed_names = {n for n, _, _, ld, _ in todo if ld}
        if parsed:
            attribution = {n: list(cited["row"].get("decisions") or []) for n, cited, _, _, _ in todo}
            table = list(parsed)
            for lv in table:
                lv.setdefault("kind", "lever")
                if lv.get("name") in landed_names:
                    # the reversed recipe's add IS the lever's remove leg
                    lv["remove_pct"], lv["remove_ms"] = lv.get("add_pct"), lv.get("add_ms")
                    lv["add_pct"], lv["add_ms"], lv["landed"] = None, None, True
                lv["decisions"] = attribution.get(lv.get("name"), [])
                lv["priced_on"], lv["job"] = rec.sha, job
            if stack:
                stack.update(kind="stack", priced_on=rec.sha, job=job)
                table.append(stack)
            # the forward recipes stay the banked recipes (a landed lever's undo
            # is derived from them whenever it is needed again)
            _persist_lever_table(ctx, table, rec.sha, job, reprice=True,
                                 spec_levers=[sp for _, _, _, _, sp in todo])
        noise = _table_noise_pct([stack] if stack else [])
        for name, cited, _, landed, _ in todo:
            row = by_name.get(name)
            if row is None:
                revise(name, cited, "unverifiable",
                       fail_note or "the re-price job printed no reading for it", job=job, landed=landed)
                continue
            verdict, note = _reprice_verdict(cited["row"], row, noise)
            if landed:
                note += " — the lever is landed on this tree; read by taking it out"
            revise(name, cited, verdict, note, now_row=row, job=job, landed=landed)
    _append_revisions(ctx, revisions)
    for rv in revisions:
        _log(f"iter {rec.iter}: price-check `{rv['lever']}` {rv['verdict'].upper()} — {rv['note']}"
             + (" (the citing rung is SUSPENDED for the student)"
                if rv["verdict"] in _REVISION_SUSPENDS else ""))
    marked = text
    for rv in revisions:
        marked = _mark_lever_citations(marked, rv["lever"], _reprice_marker(rv))
    if marked != text:
        _write_milestone(ctx, marked)
    return revisions
