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

A port may also carry two optional physical facts the linker uses to generate the hand-off:

- `handle` — where the tensor lives: `tmem` (Blackwell tcgen05 accumulator), `reg` (Hopper
  WGMMA / consumer-Blackwell `mma.sync`) or `smem` (a ring slot). Two ports with different
  handles are *bridged by the copy the linker emits* — that is tensor import — so the kind is
  not a seam requirement. An unknown kind is `L-HANDLE`.
- `generation` — which pipeline generation the port carries (default `0`). Tensor import
  hands off one generation; a producer and consumer that disagree mean several in flight,
  refused with `L-PHASE` and the ring depth (`|Δ| + 1`) that would carry them.

Anything that satisfies a contract can stand in for anything else that does, **in any
language**. Swapping a component is choosing a different implementation of the same
contract; reusing it in a new model or system is importing it by contract, never by name.

## Commands

```
forge component new <name> [--language cutedsl,cuda,tilelang] [--summary "..."]
forge component new <name> --from-run <run>     # contract + verify lifted from the card
forge component list
forge component check [<name> | --all]
forge component plan <a> <b> ... [--edge ...] [--graph ascii|mermaid] [--format json]
forge component link <a> <b> ... [--edge FROM.OUT=TO.IN]... [--emit DIR] [--cap KEY=INT] [--ring N] [--format json]
forge component diff <before...> --then <after...>
```

`check` is the static gate a component must pass to be called deliverable — the same role
`judge.check_problems` plays for an operator's check directory, one grain down. It runs no
GPU and holds no state; it checks the structure and the interface, nothing else.

## Linking a chain into one fused kernel (simulated)

`forge component link` is a **simulated** PipeLink-style linker (`forgekernel.link`). It does
the part that does not need a GPU, which is where most link-time decisions live:

- **seam checking** — two stages link only if the output one produces matches, by name /
  dtype / shape-over-axes, the input the next consumes. A mismatch is refused with a numbered
  `L-PROTO` / `L-HANDLE` / `L-PHASE` / `L-STAGE` / `L-RES` / `L-ROLE` / `L-RING` / `L-ORDER` /
  `L-WIRE` diagnostic, rather than a kernel that hangs on the GPU. `L-HANDLE` is an unknown
  tensor location; `L-PHASE` is a generation mismatch (several in flight — needs a ring).
  Repeated codes are coalesced into one line each with every site ("L-PROTO [2 sites] …").
- **chain or DAG** — by default the pipeline is a chain (each stage's single output feeds the
  next). Pass `--edge FROM.OUT=TO.IN` (repeatable) and the linker treats it as a DAG instead:
  a stage may take several inputs, an output may fan out, and the stages are topologically
  ordered (a cycle, or a non-source input left unwired, is refused with `L-ORDER` / `L-WIRE`).
- **resource and role solving** — each stage declares its claim (`[resources]`:
  `smem_bytes`, `tmem_cols`, `regs_per_thread`, `barrier_slots`; `[roles]`: `warps`,
  `ring_min`, `ring_max`, `producer_rate`, `consumer_rate`, `expected_ms`). The linker
  solves the **peak simultaneous use** (a stage holds its buffers from start until its last
  consumer starts, so a chain of eight 64 KB stages fits a 227 KB CTA even though it sums to
  512 KB), the **warp peak** the same way, and the **ring depth** from each stage's floor and
  any producer/consumer rate mismatch (`ceil(prod/cons)` slots). It refuses a plan that
  overruns a cap, and the `L-RES` / `L-ROLE` diagnostic names the stages **live at the peak**
  (`smem_bytes peak 262144 at {a, b}`), so the fix has a lead.
- **bridge hints** — on a mismatched seam, if a component under the root could sit between
  the two stages (its input takes what the producer emits, its output gives what the consumer
  wants), the `L-PROTO` diagnostic names it: "insert X between them".
- **fused-source generation** — `--emit DIR` writes `fused_kernel.<ext>` (each stage's body
  inlined under its role, hand-offs generated from the plan; Python skeletons are
  `ast.parse`-checked). For each edge it emits the **actual move**, chosen from the two ports'
  handle kinds — `acc_tmem_to_reg` when a TMEM accumulator is imported into registers,
  `acc_reg_to_smem` to stage a ring slot, `acc_identity` when unchanged. Fusing **refuses two
  stages that define the same top-level name** (`L-PROTO`, reported with `--emit` as a
  refusal, not a traceback): inlining would let the later definition silently shadow the
  earlier one. `link-plan.json` records stages, edges, solved resources and ring depth.
- **semantic parity, on the CPU** — `forgekernel.link.run_parity` chains each stage's
  `verify.json` golden through the same wiring the linker solved and checks it against a
  reference on CPU torch. Each reading carries the stage's output shape, so a mismatch points
  at the seam; a stage with no `golden.torch` yields `parity: "incomplete"` naming the stage
  rather than passing silently. This proves the *link semantics* without a GPU; whether the
  generated CUDA is fast still needs the hardware.

`forge component plan` is a dry run of `link`: the same checks and solving, a human summary
(with `--graph ascii|mermaid`), and per-resource reasoning — no files written. `forge
component diff <before...> --then <after...>` re-links two pipelines and reports the stage /
edge / resource / ring changes, the "swap one stage, re-link the rest" check. Both accept
`--format json` for CI.

It links what stages **declare**, not the opaque code behind the declaration, so every linked
kernel still needs its numeric check and sanitizers before it is trusted on hardware — the
same caveat PipeLink's own README makes. What it buys is that contract, budget and wiring
errors surface here, instantly and with a code, instead of as a GPU hang.

