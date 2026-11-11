"""Minimal CTA-pair bulk load skeleton: `.cta_group::2` (Blackwell sm100a).

Source recipe: hw_kb/cutedsl/sm100/cta_pair_bulk_load_cta_group2.md
Mechanism: on a CTA pair, `cp.async.bulk.tensor` may only deliver its completion
signal to another CTA's mbarrier when the op carries `.cta_group::2`. The default
`.cta_group::1` is refused by the ISA (doing it anyway is UB).

The dangerous part is that omitting the modifier changes NOTHING ELSE: same
cute.copy shape, same descriptor, same box, same byte count. So it raises no
error, fails no compile, produces no wrong numbers -- the barrier simply never
fills and the kernel hangs. Hence the explicit construction below.

Run inside an image that ships CuTeDSL (lmsysorg/sglang:dev, cutlass 4.8.0):

    python3 cta_pair_bulk_load.py
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05


def build_pair_load_atom(dtype_x, shape, smem_one, cta_tiler):
    """Build the load atom for a CTA pair.

    `CtaGroup.TWO` is the whole mechanism: it is a single constructor argument,
    and it is what allows the completion signal to land on the peer CTA's
    mbarrier. Without it the pair's full barrier can never be filled.
    """
    x = cute.runtime.make_fake_tensor(dtype_x, shape)

    op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
    tma_atom, gX = cpasync.make_tiled_tma_atom(op, x, smem_one, cta_tiler, 1)
    return tma_atom, gX


def build_single_load_atom(dtype_x, shape, smem_one, cta_tiler):
    """The anti-trigger: a single CTA (or a pair whose two halves each own and
    issue their own MMA) must NOT set CtaGroup.TWO -- the default is correct."""
    x = cute.runtime.make_fake_tensor(dtype_x, shape)
    op = cpasync.CopyBulkTensorTileG2SOp()          # CtaGroup.ONE
    tma_atom, gX = cpasync.make_tiled_tma_atom(op, x, smem_one, cta_tiler, 1)
    return tma_atom, gX


def check_barrier_placement(barrier_is_peer: bool, cta_group_is_two: bool) -> None:
    """`Layer 1` of the recipe, as an assertion: under .cta_group::1 the mbar
    must live in the same CTA as dstMem; only .cta_group::2 may point at the
    peer's mbar."""
    if barrier_is_peer and not cta_group_is_two:
        raise AssertionError(
            "redirecting the completion to the peer under .cta_group::1 is UB; "
            "build the op with CtaGroup.TWO"
        )


def main():
    print("cutlass", getattr(cutlass, "__version__", "?"))
    print("pair op :", cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO))
    print("single  :", cpasync.CopyBulkTensorTileG2SOp())
    print("both take identical copy shape / descriptor / box / byte count")

    check_barrier_placement(barrier_is_peer=True, cta_group_is_two=True)
    try:
        check_barrier_placement(barrier_is_peer=True, cta_group_is_two=False)
    except AssertionError as exc:
        print("guard works:", exc)

    # What a full example still needs:
    #   1) one mbar allocator shared across the pair, the peer slot registered
    #   2) build_pair_load_atom(...) for every operand that the pair consumes
    #   3) expect_tx on the shared barrier = the PAIR's byte count, not one half
    #   4) each CTA issues its own load (one cute.copy moves one share, not a pair)
    #   5) teardown: both CTAs meet on the dealloc barrier before exit
    print("todo: shared mbar -> atoms -> expect_tx pair bytes -> per-CTA load -> teardown")


if __name__ == "__main__":
    main()
