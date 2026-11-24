# Kernel components

A **component** is one bounded piece of GPU work made first-class: a developer or an agent
works inside a single directory against a fixed contract, delivers it with its own proof,
and it is dropped into another operator, model or system **by contract, not by copy**.

The goal this exists for: every carefully tuned GPU kernel becomes a unit that can be
delivered, replaced and reused independently, so the work keeps paying off in new models and
systems instead of living as lines inside one operator's `kernel.py`.

## Layout

```
components/<name>/
    component.toml      metadata: what it is, what it needs, which languages implement it
    contract.json       the DSL-neutral interface: inputs/outputs (shape+dtype over named
                        axes), invariants, and one sentence of semantics
    <language>/         one implementation per language declared in component.toml
        kernel.py         cutedsl — kernel.cu for cuda, kernel_tl.py for tilelang
        verify.json     this implementation's own test: shape, tolerance, golden statement
```

The **boundary** a contributor stays inside is exactly one component directory:
`contract.json` says what may cross it, the implementation slot says where the work lives,
and `verify.json` says how it will be judged before it leaves.

## Contract = the swap surface

`contract.json` is deliberately the same shape as an operator's judge definition
(`{"inputs": {name: {"shape", "dtype"}}, "outputs": ...}`), plus:

- `axes` — the axis names the shapes are written over,
- `invariants` — what a caller may assume without reading the code,
- `semantics` — one sentence every implementation must agree on.

Anything that satisfies a contract can stand in for anything else that does, **in any
language**. Swapping a component is choosing a different implementation of the same
contract; reusing it in a new model or system is importing it by contract, never by name.

## Commands

```
forge component new <name> [--language cutedsl,cuda,tilelang] [--summary "..."]
forge component list
forge component check [<name> | --all]
```

`check` is the static gate a component must pass to be called deliverable — the same role
`judge.check_problems` plays for an operator's check directory, one grain down. It runs no
GPU and holds no state; it checks the structure and the interface, nothing else.
