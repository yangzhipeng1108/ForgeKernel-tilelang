"""Shared module-level constants used across the loop sections.

These constants are referenced from several sections that still live in driver.py
(timeline, gap queue) as well as the phase-exhaustion block, so they are defined
here once and imported back by driver early (before any of those uses) rather than
living deep inside one section. Moving them out is what lets the phase-exhaustion
block be extracted later without reaching back into timeline/gap for names.

Only depends on the already-extracted pricing module (for the lever limits used to
format _LEVER_DESIGN_BLOCK); no import cycle.
"""

from __future__ import annotations

import re

from forgekernel.loop.pricing import _LEVER_MAX_ABLATIONS, _LEVER_MAX_SUBSETS

_DECISION_ID = re.compile(r"^D\d+\.\d+$")

TRUNCATION_STORM_BUDGET = 8  # per phase; past this the channel is sick, not unlucky

_LEVER_DESIGN_BLOCK = (
    "\n## Pricing round (this turn only designs candidates; no diagnosis)\n"
    "Before you write cause_class, the driver **measures** on the best tree what each candidate "
    "you list is worth: add-one (base + candidate), remove-one (all-on − candidate) and all-on, as "
    "same-job paired ratios, {PRICING_JOBS}. Your task: of **all the directions in the evidence "
    "above that you would consider patching**, take those that can be written as **local, "
    "stackable exact-text replacements** in `{KERNEL_FILE}` (at most "
    "{MAX_LEVERS} groups) and write them into `notes/levers.json`.\n\n"
    "**Group first, then list.** One candidate = a group of edits that **only work together** "
    "(one place, or three). "
    "Grouping is a mechanism judgement and it is yours: when two edits touch the two ends of the "
    "same wait, or one is the precondition of the other (registers must be freed before the "
    "prefetch can deepen; the holding scheme must change before the batch depth can rise), put "
    "them in **the same candidate's `edits` list**, and the driver measures them as one unit. "
    "Listing them apart makes the driver measure them separately — the half that is useless "
    "alone reads 0, the other half reads negative, both are judged null, and the real gain was "
    "never measured. Conversely, write two edits that **each stand on their own** as two "
    "candidates, so the driver can tell whether stacking them helps or cancels. When unsure, "
    "add one more \"merged group\" candidate "
    "(A+B as a third candidate); all-on / remove-one will tell you the answer.\n\n"
    "```json\n"
    "{\"rounds\": 3, \"levers\": [\n"
    "  {\"name\": \"<short name>\", \"decisions\": [\"D<x>.<y>\"], "
    "\"expect\": \"<one-sentence mechanism prediction: which wait of which role this group changes>\",\n"
    "   \"edits\": [{\"old\": \"<original text occurring exactly once in the kernel>\", \"new\": \"<replacement>\"},\n"
    "             {\"old\": \"...\", \"new\": \"...\"}],\n"
    "   \"in_stack\": true},\n"
    "  ...],\n"
    " \"variants\": [\n"
    "  {\"name\": \"<subset name>\", \"levers\": [\"<lever name>\", \"<lever name>\"], "
    "\"question\": \"<the question this combination answers>\"},\n"
    "  ...]}\n"
    "```\n"
    "**You decide which combinations to measure.** The driver always measures base and every "
    "add-one (the floor under every citation; not optional); "
    "all-on / all-on minus one are measured by default too, but they cannot answer two questions: "
    "whom a `coupled` lever actually depends on, and how much information the other readings keep "
    "once a lever that is harmful on its own got into all-on. You answer those with subsets in "
    "`variants` "
    "(at most {MAX_SUBSETS}, each a set of lever names + one sentence on the question it answers); "
    "each subset gets its own "
    "`VARIANT` reading against base, which the diagnosis can cite directly as "
    "`priced_by=<subset name>`. Typical uses: suspect A depends on B → "
    "list `{A, B}`; suspect C is harmful → give C `\"in_stack\": false` (only its add-one is "
    "measured; it enters neither all-on "
    "nor any subset), or list a combination without C. In addition, when a lever is judged harmful "
    "(negative add-one, and all-on does not depend on it) the driver automatically measures "
    "\"all-on minus these\" once more (`all-pos`, re-measured interleaved with base); you do not "
    "write it.\n"
    "Rules: (1) every `old` occurs exactly once in the kernel, and still exactly once after all "
    "candidates are stacked in order "
    "(the edit regions of different candidates do not overlap; several edits inside one candidate "
    "are applied in list order); (2) write only local edits of the **timing / holding scheme / "
    "issue organisation** kind (wait placement, index holding, ring acquisition granularity, "
    "DEPBAR depth, cache bits, ...); "
    "edits to **geometry / axes / stage shape** are not candidates — that is R1 reopen territory, "
    "it goes into the diagnosis, not here; but a reopen "
    "itself likewise **must not happen unpriced**: its price comes from a verdict experiment "
    "(PROBE_REQUEST builds a minimal variant on the best tree, paired same-job measurement), and "
    "the diagnosis cites the reading with `reopen_price:`; (3) no "
    "new implementation bodies and no reference code; (4) the members inside one candidate group "
    "are never priced individually — if you want to know what one place inside a group is worth "
    "on its own, also list it as an independent candidate; (5) **every candidate must carry "
    "`decisions`**: which "
    "milestone decision(s) it moves the value of (decision ids such as `D1.1`). The driver uses "
    "this column to map the pricing table back onto the phase order — "
    "when every candidate of the current phase measures null, the driver rules \"this phase's "
    "levers are exhausted\" and asks you to rule on closing the phase in the diagnosis, "
    "so the student stops re-measuring the same knobs point by point. A candidate without "
    "`decisions` is still priced but does not count toward any phase's exhaustion ruling.\n"
    "**Ablations (optional, same job, at most {MAX_ABLATIONS} entries in `ablations`)**: a "
    "candidate asks \"is it faster with this added\"; "
    "an ablation asks \"how much faster is it once a mechanism **already on** the tree is removed "
    "/ degraded\" — strip one layer of barrier, make a pipelined stretch serial, "
    "stub a stage out to its naive form. It is **a conviction tool, not a candidate**: applied "
    "alone on base only, never in all-on or any "
    "subset, and **allowed to compute wrong results** (the reading is the NCU kernel clock, "
    "re-measured in the same pass as base; bench only records whether it happens to stay "
    "correct). "
    "Each entry carries a `hypothesis` (the one sentence you want to confirm / refute) and "
    "`decisions`. Δ positive = faster without it = the upper bound of what this mechanism costs "
    "now; Δ negative = slower without it = it is helping. When to use one: when no candidate can "
    "be written — you do not know how the reference does it "
    "(blackbox run), the fix is geometry / stage shape (R1 territory), or the probe script failed "
    "to produce a verdict several times — convict with an ablation first, then decide whether "
    "a reopen is worth it; an R1 can cite `reopen_price: <Δ> via ablation:<name>` (only a "
    "positive Δ above the noise floor counts as a price). "
    "A hanging ablation: ncu runs first, a timeout marks it hung and it is not touched again (no "
    "bench, no later rounds), and the ablation pass has its own total budget; "
    "**an ablation that deadlocks still costs one ncu timeout**, so prefer a degraded form "
    "(swap in a coarser synchronisation that still runs) over a deletion that will hang. "
    "Names must not equal a lever / subset name.\n"
    "```json\n"
    "{\"ablations\": [\n"
    "  {\"name\": \"<short name>\", \"decisions\": [\"D<x>.<y>\"], "
    "\"hypothesis\": \"<why it is faster without it, and by how much>\",\n"
    "   \"edits\": [{\"old\": \"<original text occurring exactly once in the kernel>\", "
    "\"new\": \"<the removed / degraded form>\"}]}\n"
    "  ...]}\n"
    "```\n"
    "After writing the file, end this turn with a single line `LEVERS: <n>` (no cause_class); if "
    "there really is no stackable candidate, write "
    "`LEVERS: none` (if the file holds only `ablations`, the driver still runs the ablation "
    "pass). Next turn you get the pricing table and then diagnose.\n")
_LEVER_DESIGN_BLOCK = _LEVER_DESIGN_BLOCK.replace("{MAX_SUBSETS}", str(_LEVER_MAX_SUBSETS)).replace(
    "{MAX_ABLATIONS}", str(_LEVER_MAX_ABLATIONS))
