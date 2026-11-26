"""Tests for the evolutionary population search (search_loop.evolve) — no GPU.

The GPU is kept at arm's length behind ``evolve(evaluate, ...)``, so every test here injects a
plain ``evaluate`` (a fake backend) and verifies the loop's *logic*: determinism from a seed,
that a genome is measured at most once (the budget buys distinct kernels), that selection
carries the best forward and mutation explores without duplicating, and that the result and
digest report what happened. The real GPU backend changes only the evaluate body, so these pin
everything that decides which kernel gets built next.
"""

from __future__ import annotations

import random

import pytest

from forgekernel.loop.search_loop import (
    EvolutionConfig,
    Genome,
    evolve,
    mutate,
    random_genome,
    render_evolution,
)


def _rewarder():
    """A fake backend: reward rises with how many arm names sort high; deterministic in genome.

    Genome keys are tuples of arm strings, so ``hash`` is stable within a process but NOT
    across runs — do not depend on absolute values, only on ordering the loop is asked about.
    """
    def evaluate(g: Genome) -> float:
        # prefer genomes containing "retile" and "pipeline", penalise "other"
        score = 0.0
        score += 0.5 if "retile" in g.arms else 0.0
        score += 0.4 if "pipeline" in g.arms else 0.0
        score -= 0.3 if "other" in g.arms else 0.0
        score -= 0.01 * len(g.arms)  # mild parsimony pressure
        return score
    return evaluate


def test_evolve_is_deterministic_from_seed():
    a = evolve(_rewarder(), config=EvolutionConfig(population=6, generations=3), seed=42)
    b = evolve(_rewarder(), config=EvolutionConfig(population=6, generations=3), seed=42)
    assert a.best.key() == b.best.key()
    assert a.best_reward == pytest.approx(b.best_reward)
    assert a.evaluations == b.evaluations
    for ra, rb in zip(a.history, b.history):
        assert [g.key() for g in ra.genomes] == [g.key() for g in rb.genomes]


def test_evolve_different_seeds_differ():
    a = evolve(_rewarder(), config=EvolutionConfig(population=6, generations=2), seed=1)
    b = evolve(_rewarder(), config=EvolutionConfig(population=6, generations=2), seed=2)
    # overwhelmingly likely different initial populations; the point is seed actually varies it
    assert [g.key() for g in a.history[0].genomes] != [g.key() for g in b.history[0].genomes]


def test_each_genome_evaluated_at_most_once():
    calls: list[tuple[str, ...]] = []

    def evaluate(g):
        calls.append(g.key())
        return float(sum(1 for a in g.arms if a in ("retile", "pipeline")))

    res = evolve(evaluate, config=EvolutionConfig(population=8, generations=4), seed=7)
    # no key measured twice across the whole run
    assert len(calls) == len(set(calls))
    assert res.evaluations == len(calls)


def test_best_tracks_the_max_over_all_generations():
    res = evolve(_rewarder(), config=EvolutionConfig(population=10, generations=4), seed=3)
    all_pairs = [r for rec in res.history for r in rec.ranked()]
    top = max(all_pairs, key=lambda p: p[1])
    assert res.best.key() == top[0].key()
    assert res.best_reward == pytest.approx(top[1])


def test_selection_carries_elites_into_next_generation():
    res = evolve(_rewarder(), config=EvolutionConfig(population=8, elite=2, generations=2), seed=5)
    first_top = [g.key() for g, _ in res.history[0].ranked()[:2]]
    second_keys = {g.key() for g in res.history[1].genomes}
    assert set(first_top) <= second_keys


def test_improved_flag_reflects_rising_best():
    # a reward that rewards retile+pipeline: over generations the population should find one
    res = evolve(_rewarder(), config=EvolutionConfig(population=12, generations=5), seed=11)
    if res.history[0].rewards and res.history[-1].rewards:
        first_best = max(res.history[0].rewards)
        last_best = max(res.history[-1].rewards)
        assert res.improved == (res.best_reward > first_best)
        # with parsimony + retile/pipeline signal, later gens should not be worse than first
        assert last_best >= first_best - 1e-9


def test_population_size_respected_each_generation():
    res = evolve(_rewarder(), config=EvolutionConfig(population=7, generations=3), seed=9)
    for rec in res.history:
        # dedupe may shrink below population, but never exceed it
        assert len(rec.genomes) <= 7
        assert len(rec.genomes) == len(rec.rewards)
        assert len({g.key() for g in rec.genomes}) == len(rec.genomes)


def test_genome_length_fixed_when_configured():
    res = evolve(_rewarder(), config=EvolutionConfig(
        population=6, generations=2, genome_length=3), seed=4)
    for rec in res.history:
        for g in rec.genomes:
            # created genomes are length 3; mutations swap an arm (keep length), so length holds
            assert len(g.arms) == 3


def test_mutation_swaps_one_arm_and_never_duplicates():
    rng = random.Random(0)
    g = Genome(("retile", "pipeline"))
    child = mutate(g, rng, rate=1.0)
    assert len(child.arms) == len(g.arms)
    assert len(set(child.arms)) == len(child.arms)  # no duplicate
    assert child.arms != g.arms                    # at least one arm changed
    # exactly one position differs
    assert sum(a != b for a, b in zip(g.arms, child.arms)) == 1


def test_mutation_rate_zero_is_identity():
    rng = random.Random(1)
    g = Genome(("retile",))
    assert mutate(g, rng, rate=0.0).key() == g.key()


def test_random_genome_distinct_arms_and_length():
    rng = random.Random(2)
    for _ in range(50):
        g = random_genome(rng)
        assert 1 <= len(g.arms) <= 13
        assert len(set(g.arms)) == len(g.arms)
    g3 = random_genome(rng, length=3)
    assert len(g3.arms) == 3


def test_config_validation():
    with pytest.raises(ValueError):
        evolve(_rewarder(), config=EvolutionConfig(population=0))
    with pytest.raises(ValueError):
        evolve(_rewarder(), config=EvolutionConfig(population=4, elite=5))


def test_render_evolution_lists_generations_and_best():
    res = evolve(_rewarder(), config=EvolutionConfig(population=5, generations=2), seed=1)
    out = render_evolution(res)
    assert "generation" in out and "best" in out
    assert "reward" in out
    assert "best:" in out


def test_render_evolution_empty():
    from forgekernel.loop.search_loop import EvolutionResult
    out = render_evolution(EvolutionResult(history=[], best=None, best_reward=0.0, evaluations=0))
    assert "no generations" in out
