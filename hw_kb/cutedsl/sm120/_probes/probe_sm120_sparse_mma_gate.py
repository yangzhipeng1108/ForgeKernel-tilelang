"""2:4 sparse fp16 MMA on sm120: NEGATIVE RESULT -- the op cannot be tiled.

PROBE: negative. CuTeDSL has no sparse kernel for this card and, as of cutlass-dsl
4.8.0, one cannot be written: MmaF16BF16SparseOp constructs, but driving it through
make_tiled_mma raises

    AttributeError: module 'cutlass._mlir.dialects.cute_nvgpu' has no attribute
                    'MmaAtomSM80SparseType'

The cause is in the DSL itself. MmaF16BF16SparseOp._make_trait (in nvgpu/warp/mma.py,
the `_cute_nvgpu_ir.MmaAtomSM80SparseType.get(...)` call) references an MLIR atom type
that the dialect does not export. Enumerating the dialect, the only sparse atom types
present are MmaAtomSM100UMMASparseType and MmaAtomSM107UMMASparseType -- there is no
SM80 and no SM120 sparse atom type. So the op is constructible and the instruction
exists in PTX, but nothing here lowers it into an MMA atom, which means no
fragment_A, no cute.gemm, no kernel.

This is the mirror image of the TileLang fp4 situation: there the entry point exists
and produces wrong numbers; here the entry point exists and cannot be lowered at all.

Also pinned: sparse f16 MMA shapes are (16,8,16) and (16,8,32) only -- a (16,8,64)
request is rejected with exactly that message. SparseMetadataFormat has one member,
TID.

For now the only working 2:4 sparse path on this box is TileLang's T.gemm_sp
(262 TF/s at 4096^3, 1.95x dense); see hw_kb/tilelang/sm120/sparse_gemm_two_mechanisms.md.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Float16, Float32
from cutlass.cute.nvgpu.warp import SparseMetadataFormat

M = N = 128
K = 64
DT = Float16
ACC = Float32


def main():
    for shape in [(16, 8, 32), (16, 8, 16), (16, 8, 64)]:
        try:
            op = cute.nvgpu.warp.MmaF16BF16SparseOp(
                DT, ACC, shape, SparseMetadataFormat.TID)
            print(f"op {shape} constructs OK")
        except Exception as e:
            msg = str(e).splitlines()
            first = next((ln for ln in msg if "expects" in ln), type(e).__name__)
            print(f"op {shape} FAIL: {first.strip()}")


@cute.jit
def probe():
    op = cute.nvgpu.warp.MmaF16BF16SparseOp(
        DT, ACC, (16, 8, 32), SparseMetadataFormat.TID)
    tiled = cute.make_tiled_mma(op, cute.make_layout((2, 2, 1)))
    print("tiled_mma shape MNK:", tiled.shape_mnk)
    a = cute.make_tensor(
        cute.make_ptr(DT, 0, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_ordered_layout((M, K // 2), order=(1, 0)))
    b = cute.make_tensor(
        cute.make_ptr(DT, 0, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_ordered_layout((N, K), order=(1, 0)))
    thr = tiled.get_slice(0)
    fA = tiled.make_fragment_A(thr.partition_A(a))
    print("fragment_A:", fA)
    fB = tiled.make_fragment_B(thr.partition_B(b))
    print("fragment_B:", fB)


if __name__ == "__main__":
    main()
    try:
        probe()
    except Exception as e:
        print("make_tiled_mma FAIL:", type(e).__name__, e)
