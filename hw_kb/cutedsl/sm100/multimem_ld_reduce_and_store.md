# multimem_ld_reduce / multimem_st (NVLink multicast-memory load-reduce and store)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **NVLink + NVSwitch (NVLS / multicast memory)**. Not PCIe.
>
> **Status: ⚠️ API surface verified, mechanism UNVERIFIED on hardware.**
> The module and every symbol below were read out of `cutlass.utils.distributed` in
> cutlass 4.8.0 (the version inside `lmsysorg/sglang:dev`) — file
> `.../nvidia_cutlass_dsl/dsl_packages/cutlass/utils/distributed.py`, 600 lines.
> No GPU probe in this repo has executed a multicast instruction, so this entry
> carries **no measured numbers** and must not be cited as a performance reading.

## Purpose

Let a kernel **read across the peers of a multicast (NVLink) memory object and get the
reduced value in registers**, or **write one value into all peers at once**, without a
separate collective and without an HBM round trip. This is the one place where a kernel
author can do cross-GPU reduction inside their own kernel instead of calling NCCL.

## When needed (trigger)

- You want the all-reduce of a tensor-parallel GEMM fused into its epilogue, so the
  accumulator leaves registers already summed across the TP group.
- You want to broadcast a scalar (a flag, an epoch, a running max) to every peer without
  a ring pass.
- Symptom that sends you here: an all-reduce shows up as its own kernel in the profile,
  and the gap between GEMM-end and post-reduce is a meaningful share of the step.
- **Anti-trigger**: any topology without multicast memory (plain PCIe, or a single GPU).
  Check the topology before writing a line of this — see Gotchas 1.

## Minimal runnable example

```python
# Source: cutlass/utils/distributed.py (cutlass 4.8.0). Signatures read verbatim
# from the installed module; the PTX string is the function's own default argument.
from cutlass.utils import distributed as dist

# 128-bit vector load-reduce: 8 x fp16 summed (accumulated in fp32) -> 4 x Int32 regs
r0, r1, r2, r3 = dist.multimem_ld_reduce_8xf16(mc_ptr)
# PTX it emits:
#   multimem.ld_reduce.weak.global.add.acc::f32.v4.f16x2 {$0,$1,$2,$3}, [$4];

# 128-bit vector store: write 4 x Int32 into the multicast object (lands in all peers)
dist.multimem_st(mc_ptr, r0, r1, r2, r3)
# PTX: multimem.st.weak.global.v4.f32

# dtype/width dispatch form (picks the variant for you):
regs = dist.multimem_ld_reduce(mc_ptr, dtype=cutlass.Float16, num_elements=8)
```

`mc_ptr` is a pointer into a **multicast memory object**, not an ordinary device pointer;
obtaining it is a host-side responsibility (see Gotchas 2), and this module does not help
with it.

### The full exported surface

`__all__` has 26 names. Grouped:

| Group | Names |
|---|---|
| reduce-load, 128-bit (4 regs) | `multimem_ld_reduce_8xf16`, `_4xf32`, `_8xbf16`, `_16xe4m3`, `_16xe5m2` |
| reduce-load, 64-bit (2 regs) | `multimem_ld_reduce_4xf16`, `_2xf32`, `_4xbf16`, `_8xe4m3`, `_8xe5m2` |
| reduce-load, 32-bit (1 reg) | `multimem_ld_reduce_2xf16`, `_1xf32`, `_2xbf16`, `_4xe4m3`, `_4xe5m2` |
| dispatch | `multimem_ld_reduce(mc_ptr, *, dtype, num_elements)` |
| store | `multimem_st`, `multimem_st_4xb32`, `_2xb32`, `_1xb32` |
| atomics / locks | `multimem_red_add1`, `red_add1`, `spin_lock_atom_cas_relaxed_wait`, `spin_lock_atom_cas_acquire_wait`, `spin_lock_ld_lt_relaxed_wait` |

Two observations that matter when picking a variant:

1. **The 16-bit and 8-bit reduce variants accumulate in fp32, not in the narrow type.**
   `8xf16` lowers to `add.acc::f32.v4.f16x2`; `16xe4m3` lowers to `add.acc::f16.v4.e4m3x4`.
   So the register footprint is always fp32-width, and fp8 reductions are **not** free of
   an fp16 accumulate step.
2. **Only `add` exists.** Every reduce spelling is `add`; there is no `min`/`max`/`mul`
   multicast load-reduce in this module. A reduction that is not a sum needs the
   lock + ordinary-atomic route.

## Gotchas

1. **No architecture guard anywhere in the module.** The scan for `arch`, `sm_9`,
   `sm_10`, `capability`, `support` finds nothing. On a machine without multicast memory
   the call does not fail early with a clear message — the emitted PTX is simply invalid
   for that device. Rule the topology out **before** debugging the kernel.
2. **The module takes the pointer, it does not produce one.** Registering a memory range
   as multicast is host-side work (the multicast-memory API family), and it is the step
   that usually fails first: a plain `cudaMalloc` pointer handed to these ops is not a
   multicast pointer, and the instruction has no way to tell you that.
3. **`multimem_st` writes to *all* peers.** It is a broadcast store, not a normal store
   that happens to be on a shared object. Using it where one caller was supposed to update
   only its own copy silently overwrites the peers' values.
4. **The b32 store variants take `Int32` registers only.** The reduce-load side returns
   `Int32` tuples for every dtype (fp16/bf16/fp8 included) — the caller reinterprets. Mixing
   up the reinterpret and the accumulate width is a wrong-numbers bug, not a crash.
5. **`scope` on the lock helpers is `"gpu"` or `"sys"` and means what it says.** A lock
   taken at `"gpu"` scope does not order against `"sys"`-scope traffic; the multicast
   object is shared beyond one GPU, so the `"sys"` variants are the default choice.
6. **`spin_lock_ld_lt_relaxed_wait` spins on a plain relaxed load.** It is a correct spin
   only if it lives in a loop that a scheduler cannot starve — a warp that owns the only
   progress on that SM will spin forever. Keep the waiting warps and the producing warps
   in separate roles.
7. **The first three names are deprecated or thin wrappers.** `atomicAdd` (in this module)
   is deprecated toward `cute.arch.atomic_add`; `red_add1` is the unicast sibling of
   `multimem_red_add1`. Reaching for the unicast one by mistake compiles and is fast — and
   reduces nothing across peers.

## Open questions (must be answered by a probe before this is certified)

- Does a multicast load-reduce of a **GEMM accumulator written moments earlier by the same
  kernel** observe the peer's latest value, or does it need an explicit fence? The module
  exposes no fence and the PTX is `.weak`.
- What is the actual latency/bandwidth of `multimem.ld_reduce` at 32/64/128-bit widths,
  relative to a same-size NCCL all-reduce, at this repo's scale?
- Which multicast-memory registration path do the ops require, and does it survive a
  process boundary (multi-node) or is it single-node only?

## Sources

- API symbols: `cutlass.utils.distributed` in cutlass 4.8.0 —
  `.../nvidia_cutlass_dsl/dsl_packages/cutlass/utils/distributed.py` (600 lines). Every
  signature, the `__all__` list and the default `ptx_string` arguments above were read from
  that installed file, not reconstructed.
- PTX instruction names: the `multimem.ld_reduce` / `multimem.st` / `multimem.red` spellings
  are the module's own default `ptx_string`s.
- No GPU probe in this repo has executed these paths yet; this entry records the surface,
  not a verdict.
