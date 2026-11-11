import os, sys, torch
import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu.warp import mma as warp_mma

print("=== arch check ===")
from cutlass.base_dsl.arch import Arch
import cutlass.base_dsl as base_dsl
arch = base_dsl.BaseDSL._get_dsl().get_arch_enum()
print("arch =", arch)

print("=== construct the sm120 NVFP4 MMA op ===")
try:
    op = warp_mma.MmaMXF4NVF4Op(cutlass.Float4E2M1FN, cutlass.Float32, cutlass.Float8E4M3FN)
    print("OK op =", op)
    print("   shape_mnk =", op.shape_mnk)
    print("   sf_vec_size =", op.sf_vec_size)
except Exception as e:
    print("FAIL:", type(e).__name__, e)
    sys.exit(1)

print("=== admissible archs ===")
print(warp_mma.MmaSM120BlockScaledOp.admissible_archs)

print("=== SF smem layout helpers ===")
print("sm120_make_smem_layout_sfa:", callable(blockscaled_utils.sm120_make_smem_layout_sfa))
print("sm120_make_smem_layout_sfb:", callable(blockscaled_utils.sm120_make_smem_layout_sfb))

print("=== also try fp8 sm120 blockscaled (MXF8, shape (16,8,32)) ===")
try:
    op8 = warp_mma.MmaMXF8Op(cutlass.Float8E4M3FN, cutlass.Float32, cutlass.Float8E8M0FNU)
    print("OK MmaMXF8Op", op8.shape_mnk, "sf_vec", op8.sf_vec_size)
except Exception as e:
    print("FAIL MmaMXF8Op:", e)

print("=== tcgen05 arch gate on sm120 ===")
# The class attribute is the claim, and reading it avoids needing a real source operand
# (constructing the op without one raises a TypeError about ARGUMENTS, not about the arch --
# an earlier version of this probe printed that misleadingly).
from cutlass.cute.nvgpu import tcgen05
tc_archs = list(tcgen05.MmaMXF4NVF4Op.admissible_archs)
print("  tcgen05.MmaMXF4NVF4Op.admissible_archs =", tc_archs)
print("  sm_120a accepted by tcgen05:", arch in tc_archs)
print("  -> tcgen05 is a datacenter-Blackwell path; sm120 must use cute.nvgpu.warp.*")
