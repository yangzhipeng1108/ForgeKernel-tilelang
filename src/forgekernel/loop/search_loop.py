"""Evolutionary / bandit search over optimisation techniques — CUDA-L2's RL loop, on paper.

CUDA-L2 frames kernel optimisation as an RL problem: a policy proposes a code change, the
change is measured against a real reference on the GPU, and the *measured speedup* is the
reward that steers what to try next. This repo already measures that reward every iteration
(``IterationRecord.vs_ref`` / ``vs_ref_device`` / ``vs_ref_server`` — the same-job ratio) and
already records the objective change each row applied (``IterationRecord.change``). What it
did NOT have is the part that makes such a loop *search* rather than wander: a memory that
says **which kinds of technique have paid off on this kernel**, and an exploration rule that
keeps trying under-sampled ones instead of greedily re-walking a proven null.

This module is that memory + rule, and nothing that needs a GPU:

* **arm** — one optimisation technique, canonicalised from a row's ``change`` text by a small
  vocabulary of keywords (``occupancy`` / ``retile`` / ``swizzle`` / ``prefetch`` / ``tmem`` /
  ``pipeline`` / ... — see ``TECHNIQUES``). Two rows that say "raise occupancy" and "more
  warps per block" land on the same arm, so the evidence for an idea aggregates instead of
  fragmenting across phrasings.
* **reward** — the measured outcome of one row, in the run's gate window: a KEEP's same-job
  ratio is positive reward when it beat the run's best ratio before it, ~0 when it merely
  held, and negative for a REVERT (a correctness break is the most negative, because it cost
  an iteration AND erased the tree). Reward is *measured*, never read from the student's
  prose (the trajectory contract: subjective analysis is not recorded for a reason).
* **selection** — UCB1 over the arms: exploit the arm with the best mean reward, but add an
  optimism term ``sqrt(2 ln N / n)`` so an arm with few samples keeps getting tried until its
  evidence catches up. This is the explore/exploit balance a bandit gives for free and a
  greedy "do what worked" rule lacks; it is exactly what keeps a search from locking onto one
  local optimum after two lucky KEEPs.
* **proposal** — ``propose`` turns the ranking into the ranked directions a prompt can carry
  ("try X next — under-sampled; avoid Y — priced null"), plus the per-arm table the digest
  shows. The driver reads it as ADVISORY text; the student still decides, so a bad ranking can
  never wedge the loop.

It is a *decision aid* over recorded facts, not a policy network: there is no gradient, no
learned weights, no dependence on a training run. The claim it makes is narrow and testable —
"among techniques already tried, this one has the best evidence-weighted prospects and the
least evidence, so try it" — and it is stated as that, never as a predicted speedup.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from forgekernel.inner.trajectory import (
    GATE_WINDOW_DEFAULT,
    GATE_WINDOW_DEVICE,
    GATE_WINDOW_SERVER,
    gate_window_name,
    gate_ms,
)

# The technique vocabulary. Each arm key maps to the keywords that put a ``change`` text on
# it; first keyword found (in arm order) wins, so a text mentioning two ideas lands on the
# earlier arm rather than being double-counted. Keys are the stable ids the evidence is
# aggregated under; the phrases are what a ``change`` line might literally say.
TECHNIQUES: dict[str, tuple[str, ...]] = {
    "retile": ("retile", "tile size", "block tile", "threadblock", "bm", "bn", "bk",
               "tile shape"),
    "occupancy": ("occupancy", "regs per thread", "register pressure", "more warps",
                  "launch bounds", "maxntid", "minnctapersm", "warp per block"),
    "swizzle": ("swizzle", "bank conflict", "shared memory layout", "padding", "xor swizzle"),
    "prefetch": ("prefetch", "__ldg", "read-only cache", "ld.global.nc", "cp.async",
                 "async copy", "preload"),
    "pipeline": ("software pipeline", "pipelined", "double buffer", "num_stages",
                 "multi-stage", "overlap load"),
    "vectorize": ("vectorize", "float4", "ld.global.v4", "128-bit", "wide load",
                  "coalesc"),
    "tmem": ("tmem", "tcgen05", "warp-speciali", "warpgroup", "accumulator in tmem"),
    "layout": ("layout", "row-major", "column-major", "transpose", "stride", "coalescing of"),
    "epilogue": ("epilogue", "fused epilogue", "activation fuse", "store path"),
    "splitk": ("split-k", "splitk", "reduction across blocks", "atomics"),
    "precision": ("tf32", "fp16 accumulate", "fp32 accumulate", "reduced precision",
                  "mixed precision"),
    "scheduler": ("scheduler", "instruction order", "reorder", "warp scheduler",
                  "independent instruction"),
}
_OTHER = "other"

# Reward shaping. A KEEP that improved the best ratio earns its improvement directly; a KEEP
# that held is a small non-negative (it is progress the loop keeps, so not failure); a REVERT
# is negative by the cost it paid (an iteration and a reset). A correctness break is the worst
# outcome — it did not just fail to help, it produced a wrong kernel — so it is the most
# negative. Units are "ratio improvement", same as ``vs_ref``, so a reward of 0.05 means "5%
# faster than the previous best".
REWARD_KEEP_HOLD = 0.0
REWARD_REVERT = -0.05
REWARD_CORRECTNESS_BREAK = -0.25


def technique_of(change: str) -> str:
    """The arm a ``change`` text belongs to: the first keyword match, else ``other``.

    Canonicalisation is what lets two phrasings of one idea share evidence. Matching is
    case-insensitive substring; the arm order in ``TECHNIQUES`` is the tie-break, so a text
    naming "retile" and "occupancy" lands on ``retile`` (tile choices usually drive the
    occupancy that follows). An empty or unmatched change is ``other`` — not dropped, because
    an unclassifiable change still cost (or earned) a measured reward and the counts must sum."""
    low = (change or "").lower()
    for arm, phrases in TECHNIQUES.items():
        if any(p in low for p in phrases):
            return arm
    return _OTHER


def _gate_ratio(rec, gate_window: str) -> float | None:
    """The record's same-job ratio (ref / student) in the gate window, or None."""
    window = gate_window_name(gate_window)
    if window == GATE_WINDOW_DEVICE:
        return rec.vs_ref_device
    if window == GATE_WINDOW_SERVER:
        return getattr(rec, "vs_ref_server", None)
    return rec.vs_ref


@dataclass
class ArmStats:
    """Evidence for one technique arm: how often tried, and the reward sum/mean."""

    arm: str
    n: int = 0
    reward_sum: float = 0.0
    keeps: int = 0
    reverts: int = 0
    correctness_breaks: int = 0
    best_gain: float = 0.0     # largest single positive reward seen for this arm

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.n if self.n else 0.0


@dataclass
class RewardRow:
    """One record's reward, kept so a caller can trace a proposal back to its evidence."""

    iter: int
    arm: str
    reward: float
    verdict: str
    ratio: float | None


def reward_of(rec, best_before: float | None,
              gate_window: str = GATE_WINDOW_DEFAULT) -> float:
    """The measured reward of one record, given the best gate ratio before it.

    ``best_before`` is the best same-job ratio among evaluated KEEPs earlier in the run (None
    at the start). A KEEP's reward is how much its ratio improved on that best — the actual
    progress, not the absolute ratio (which would reward an easy kernel for being easy). A
    KEEP with no measured ratio earns ``REWARD_KEEP_HOLD`` (it is kept but unmeasured here — a
    delivery that only moved scaffolding, say). A REVERT earns ``REWARD_REVERT``; a REVERT with
    a correctness break earns ``REWARD_CORRECTNESS_BREAK`` — the wrong-kernel penalty."""
    if rec.verdict == "REVERT":
        if rec.correctness is False:
            return REWARD_CORRECTNESS_BREAK
        return REWARD_REVERT
    if rec.verdict != "KEEP":
        return REWARD_KEEP_HOLD
    ratio = _gate_ratio(rec, gate_window)
    if ratio is None:
        return REWARD_KEEP_HOLD
    if best_before is None:
        # No frontier stood before this KEEP: there is no progress to measure, only a
        # starting point. Rewarding its whole ratio would reward an easy kernel for being
        # easy (and let the first luckiest draw dominate the arm's mean), so it is a hold.
        return REWARD_KEEP_HOLD
    return max(0.0, ratio - best_before)


def build_stats(records, gate_window: str = GATE_WINDOW_DEFAULT
                ) -> tuple[dict[str, ArmStats], list[RewardRow]]:
    """Fold the run's records into per-arm evidence.

    Rewards are computed *in order* (best_before advances as KEEPs land), so each row is
    scored against the frontier that actually stood when it was measured, not the final one.
    Returns the arm table (in ``TECHNIQUES`` order then ``other``) and the per-row trace."""
    stats: dict[str, ArmStats] = {arm: ArmStats(arm) for arm in
                                  list(TECHNIQUES) + [_OTHER]}
    rows: list[RewardRow] = []
    best_before: float | None = None
    ordered = sorted(records, key=lambda r: getattr(r, "iter", 0))
    for rec in ordered:
        arm = technique_of(getattr(rec, "change", ""))
        ratio = _gate_ratio(rec, gate_window)
        reward = reward_of(rec, best_before, gate_window)
        st = stats[arm]
        st.n += 1
        st.reward_sum += reward
        if rec.verdict == "KEEP":
            st.keeps += 1
            if reward > st.best_gain:
                st.best_gain = reward
        elif rec.verdict == "REVERT":
            st.reverts += 1
            if getattr(rec, "correctness", None) is False:
                st.correctness_breaks += 1
        rows.append(RewardRow(iter=getattr(rec, "iter", 0), arm=arm, reward=reward,
                              verdict=rec.verdict, ratio=ratio))
        if rec.verdict == "KEEP" and ratio is not None:
            best_before = ratio if best_before is None else max(best_before, ratio)
    return stats, rows


def ucb_scores(stats: dict[str, ArmStats], *, c: float = 1.0) -> dict[str, float]:
    """UCB1 score per arm: ``mean + c * sqrt(2 ln N / n)``.

    ``N`` is the total number of tries across all arms; ``n`` an arm's own. An untried arm
    (``n == 0``) scores ``+inf`` so it is tried first — the exploration a greedy rule never
    does. A tried-but-empty frontier (``N == 0``) scores every arm 0. The optimism constant
    ``c`` (default 1.0, the textbook value) scales how much unexplored arms are favoured."""
    total = sum(s.n for s in stats.values())
    out: dict[str, float] = {}
    if total == 0:
        for arm in stats:
            out[arm] = 0.0
        return out
    log_term = math.log(total)
    for arm, s in stats.items():
        if s.n == 0:
            out[arm] = math.inf
        else:
            out[arm] = s.mean_reward + c * math.sqrt(2.0 * log_term / s.n)
    return out


@dataclass
class Proposal:
    """The search loop's recommendation for the next iteration, or the table for a digest."""

    ranked: list[str]                       # arms by UCB score, best first (untried lead)
    scores: dict[str, float]
    stats: dict[str, ArmStats]
    tried: list[str] = field(default_factory=list)      # arms with n > 0
    untried: list[str] = field(default_factory=list)    # arms with n == 0
    priced_null: list[str] = field(default_factory=list)  # tried, never a KEEP
    best_arm: str | None = None                          # the arm with the best mean reward


def propose(records, gate_window: str = GATE_WINDOW_DEFAULT, *,
            novelty_bonus: float = 0.0) -> Proposal:
    """Rank techniques to try next from the run's measured evidence.

    ``ranked`` orders every arm by UCB score; untried arms lead (infinite score) in
    ``TECHNIQUES`` order, so the first suggestion on a fresh run is the first listed
    technique rather than an unexplained pick. ``novelty_bonus`` adds a flat bonus to arms
    never tried *successfully* (still tried but never a KEEP get the bonus relative to arms
    that have paid off) — a knob for pushing a search off a proven local optimum, 0 by default.

    ``priced_null`` is the set a digest should WARN against: arms with samples but zero KEEPs
    — the "tried and never paid" ideas a greedy student would otherwise re-walk. They are
    still rankable (their UCB may be high early when ``n`` is small), but the warning keeps
    the reader from mistaking "under-sampled" for "promising"."""
    stats, _rows = build_stats(records, gate_window)
    scores = ucb_scores(stats)
    if novelty_bonus:
        for arm, s in stats.items():
            if s.keeps == 0:
                scores[arm] += novelty_bonus

    order = list(TECHNIQUES) + [_OTHER]
    ranked = sorted(order, key=lambda a: (-scores[a], order.index(a)))
    tried = [a for a in order if stats[a].n > 0]
    untried = [a for a in order if stats[a].n == 0]
    priced_null = [a for a in order if stats[a].n > 0 and stats[a].keeps == 0]
    best_arm = None
    if tried:
        best_arm = max(tried, key=lambda a: (stats[a].mean_reward, -stats[a].reverts))
    return Proposal(ranked=ranked, scores=scores, stats=stats, tried=tried,
                    untried=untried, priced_null=priced_null, best_arm=best_arm)


def render_table(prop: Proposal) -> str:
    """A markdown table of the arm evidence — the digest's search-loop view.

    Columns say, per technique: tries, KEEPs / REVERTs, best single gain, mean reward, and
    the UCB score. An untried arm shows "—" for its means so a reader never reads a zero as
    "tried and scored 0"."""
    lines = ["| technique | tries | keep/rev | best gain | mean reward | UCB |",
             "|---|---|---|---|---|---|"]
    for arm in prop.ranked:
        s = prop.stats[arm]
        if s.n == 0:
            lines.append(f"| {arm} | 0 | — | — | — | untried |")
            continue
        gain = f"{s.best_gain:+.3f}"
        mean = f"{s.mean_reward:+.3f}"
        score = prop.scores[arm]
        uc = "inf" if math.isinf(score) else f"{score:+.3f}"
        lines.append(f"| {arm} | {s.n} | {s.keeps}/{s.reverts} | {gain} | {mean} | {uc} |")
    return "\n".join(lines)


def render_advice(prop: Proposal, *, top: int = 3) -> str:
    """The one-paragraph prompt block: try these next, avoid these, grounded in evidence.

    Deliberately advisory ("consider", "avoid unless you have a new reason") — the student
    owns the decision; a ranking that read as an order would let a stale table overrule a
    fresh measurement the loop has not folded in yet."""
    parts: list[str] = []
    leads = [a for a in prop.ranked if a in prop.untried][:top]
    if not leads:
        leads = prop.ranked[:top]
    if leads:
        parts.append("Search-loop suggests trying: " + ", ".join(leads)
                     + " (best evidence-weighted prospects; untried arms lead).")
    if prop.best_arm:
        s = prop.stats[prop.best_arm]
        parts.append(f"Best paying technique so far: {prop.best_arm} "
                     f"(mean reward {s.mean_reward:+.3f} over {s.n} tries, "
                     f"best gain {s.best_gain:+.3f}).")
    if prop.priced_null:
        parts.append("Priced null on this kernel (tried, never a KEEP): "
                     + ", ".join(prop.priced_null) + " — avoid unless you have a new reason.")
    if not parts:
        return "Search loop: no evaluated iterations yet — every technique is untried."
    return " ".join(parts)


# --- evolutionary population search (CUDA-L2's RL/evolution loop, on paper) -------------
#
# The bandit above steers ONE trajectory: it says which technique to try next given what has
# been measured. CUDA-L2 runs a population instead — it generates a batch of kernel variants,
# measures each, keeps the best, and breeds the next batch from them (selection + mutation),
# which escapes the local optimum a single greedy path walks into. This layer is that loop,
# with the GPU kept at arm's length: a ``Genome`` is a candidate strategy (an ordered set of
# technique arms to apply), and ``evaluate(genome) -> reward`` is the ONLY thing that needs a
# GPU. In production ``evaluate`` builds the genome's kernel, times it against the reference,
# and returns the measured speedup; in tests it is a plain function, so the whole
# generate -> evaluate -> select -> mutate cycle is verified deterministically with no GPU.


@dataclass(frozen=True)
class Genome:
    """One candidate strategy: the ordered technique arms it applies.

    A genome is just an ordered tuple of arm names from ``TECHNIQUES`` (plus possibly
    ``other``). Order matters the way application order matters on a real kernel (retile then
    pipeline is a different kernel than pipeline then retile), but for selection purposes two
    genomes with the same arm *set* in a different order are still distinct candidates — the
    search may find one ordering pays and the other does not."""

    arms: tuple[str, ...]

    def key(self) -> tuple[str, ...]:
        return self.arms

    def describe(self) -> str:
        return " -> ".join(self.arms) if self.arms else "(empty)"


def _all_arms() -> list[str]:
    return list(TECHNIQUES) + [_OTHER]


def random_genome(rng, *, length: int | None = None) -> Genome:
    """A random genome of distinct arms (no repeats — applying one technique twice is noise).

    ``length`` picks how many arms; None draws a length in 1..len(arms). Distinct arms keep a
    genome a *set-in-an-order*, so mutation (which swaps one arm) always changes the strategy."""
    arms = _all_arms()
    n = length if length is not None else rng.randint(1, len(arms))
    n = max(1, min(n, len(arms)))
    picked = rng.sample(arms, n)
    return Genome(tuple(picked))


def mutate(genome: Genome, rng, *, rate: float = 1.0) -> Genome:
    """A child genome: with probability ``rate`` swap one arm for an unused one.

    Mutation is the exploration operator: it replaces one arm with an arm the genome does not
    already carry, so it cannot produce a duplicate (which would evaluate the same kernel
    twice). With probability ``1 - rate`` the genome passes through unchanged (a clone), which
    keeps a good parent representable without a gratuitous perturbation. If every arm is
    already used, there is nothing to swap in and the genome is returned as-is."""
    if rng.random() >= rate:
        return genome
    used = set(genome.arms)
    unused = [a for a in _all_arms() if a not in used]
    if not unused:
        return genome
    i = rng.randrange(len(genome.arms))
    new_arm = rng.choice(unused)
    arms = list(genome.arms)
    arms[i] = new_arm
    return Genome(tuple(arms))


@dataclass
class EvolutionConfig:
    population: int = 8          # candidates per generation
    elite: int = 2               # top-K carried to the next generation unchanged
    generations: int = 3         # number of evaluate/select/mutate rounds
    mutation_rate: float = 0.7   # per-child probability of a swap
    genome_length: int | None = None   # fixed arm count, or None = random per genome


@dataclass
class GenerationRecord:
    index: int
    genomes: list[Genome]
    rewards: list[float]

    def ranked(self) -> list[tuple[Genome, float]]:
        """(genome, reward) best-first; ties keep generation order (stable)."""
        pairs = list(zip(self.genomes, self.rewards))
        return sorted(pairs, key=lambda gr: gr[1], reverse=True)


@dataclass
class EvolutionResult:
    history: list[GenerationRecord]
    best: Genome | None
    best_reward: float
    evaluations: int             # total evaluate() calls — the GPU budget the loop spent

    @property
    def improved(self) -> bool:
        """Whether the best reward rose above the first generation's best."""
        if len(self.history) < 2:
            return False
        first = self.history[0].ranked()
        return bool(first) and self.best_reward > first[0][1]


def evolve(evaluate, *, config: EvolutionConfig | None = None, seed: int = 0) -> EvolutionResult:
    """Run the generate -> evaluate -> select -> mutate loop; return the best genome found.

    ``evaluate(genome) -> float`` is the measured reward (GPU hook in production; any function
    in tests). The loop:

    1. **generate** — the first generation is ``config.population`` random genomes; later
       generations are ``config.elite`` carried-over elites plus children mutated from the
       previous generation's parents (tournament-chosen, so a lucky genome does not monopolise).
    2. **evaluate** — every genome is scored once; a genome seen before (same key) is NOT
       re-evaluated — its earlier reward is reused — so the GPU budget buys distinct kernels,
       not re-measurements (the same "an unchanged tree is one more noise draw" rule the
       trajectory contract enforces).
    3. **select** — the generation is ranked by reward and its top ``elite`` seed the next.
    4. **mutate** — children are ``mutate(parent)``; a fresh random genome is injected if the
       previous generation collapsed to fewer distinct parents than the population needs.

    ``seed`` makes the whole search reproducible (a recorded run can be replayed)."""
    import random as _random
    rng = _random.Random(seed)
    cfg = config or EvolutionConfig()
    if cfg.population < 1:
        raise ValueError("population must be >= 1")
    if not (0 <= cfg.elite <= cfg.population):
        raise ValueError("elite must be in [0, population]")

    cache: dict[tuple[str, ...], float] = {}

    def _score(g: Genome) -> float:
        k = g.key()
        if k in cache:
            return cache[k]
        r = float(evaluate(g))
        cache[k] = r
        return r

    history: list[GenerationRecord] = []
    prev: list[Genome] = []

    for gen in range(cfg.generations):
        if gen == 0 or not prev:
            genomes = [random_genome(rng, length=cfg.genome_length)
                       for _ in range(cfg.population)]
        else:
            ranked = [g for g, _ in
                      GenerationRecord(gen, prev, [_score(g) for g in prev]).ranked()]
            genomes = list(ranked[:cfg.elite])
            while len(genomes) < cfg.population:
                contenders = [rng.choice(prev) for _ in range(min(3, len(prev)))]
                winner = max(contenders, key=_score)
                genomes.append(mutate(winner, rng, rate=cfg.mutation_rate))
            if len({g.key() for g in genomes}) < cfg.population // 2:
                while len({g.key() for g in genomes}) < cfg.population:
                    genomes.append(random_genome(rng, length=cfg.genome_length))

        seen: set[tuple[str, ...]] = set()
        uniq: list[Genome] = []
        for g in genomes:
            if g.key() not in seen:
                seen.add(g.key())
                uniq.append(g)
        rewards = [_score(g) for g in uniq]
        history.append(GenerationRecord(gen, uniq, rewards))
        prev = uniq

    best: Genome | None = None
    best_reward = float("-inf")
    for rec in history:
        for g, r in rec.ranked():
            if r > best_reward:
                best, best_reward = g, r
    if best is None:
        best_reward = 0.0
    return EvolutionResult(history=history, best=best, best_reward=best_reward,
                           evaluations=len(cache))


def render_evolution(result: EvolutionResult) -> str:
    """A compact digest of the search: per-generation best/mean, and the winning genome.

    One line per generation (generation, distinct candidates, best reward, mean reward) so a
    reader sees the search *progress* — a flat best across generations means the population
    converged (or never found signal), a rising best means selection + mutation are paying."""
    if not result.history:
        return "Evolution: no generations run."
    lines = ["generation | candidates | best reward | mean reward",
             "---|---|---|---"]
    for rec in result.history:
        if not rec.rewards:
            lines.append(f"{rec.index} | 0 | — | —")
            continue
        best = max(rec.rewards)
        mean = sum(rec.rewards) / len(rec.rewards)
        lines.append(f"{rec.index} | {len(rec.genomes)} | {best:+.3f} | {mean:+.3f}")
    if result.best is not None:
        lines.append("")
        lines.append(f"best: {result.best.describe()}  (reward {result.best_reward:+.3f} "
                     f"over {result.evaluations} evaluated genomes)")
    return "\n".join(lines)
