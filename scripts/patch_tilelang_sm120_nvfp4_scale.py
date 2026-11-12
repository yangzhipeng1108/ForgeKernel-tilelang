"""Patch v3: packed uint32 scale word, row/kblock indexing, stage-safe view.

v1: packed uint32 but kept the word-offset formula -> cos 0.956.
v2: packed uint32 + direct (row, k_block) indexing -> cos 1.000000 at
    num_stages=1, but T.view failed under num_stages=2 because the smem scale
    buffer is then 3-D (stage, BM, BK/sf_vec) = (2, 128, 8), and the view was
    hard-coded to 2-D.

v3 keeps every leading dimension and only re-types the last one, so the same
tuple(SFA_other) + (row, col) index still lands correctly: SFA_other already
carries the stage coordinate from _scale_region_parts.

Idempotent.
"""
import sys

F = "/home/nvidia/tlvenv/lib/python3.14/site-packages/tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py"
src = open(F).read()

if "PACKED_SCALE_WORD_V3" in src:
    print("ALREADY PATCHED v3")
    sys.exit(0)

anchor = """        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)

        @T.macro
        def _load_kblock(A_local_buf, B_local_buf, SFA_local_buf, SFB_local_buf, k_block):"""
replace = """        SFA_data, SFA_other, SFA_base_m, SFA_base_k = self._scale_region_parts(SFA_buf)
        SFB_data, SFB_other, SFB_base_n, SFB_base_k = self._scale_region_parts(SFB_buf)

        # PACKED_SCALE_WORD_V3: scale_vec::4X wants one uint32 carrying four
        # e4m3 bytes per MMA operand. The smem scale tile is
        # [..., BM, BK/sf_vec] fp8, which is the same storage as
        # [..., BM, BK/(sf_vec*4)] uint32 words in row-major order, so the
        # packed word for (row, k_block) is word[row, k_block]. Reading the fp8
        # buffer directly took one byte and zero-extended it, leaving bytes 1-3
        # as 0x00. Leading dims (pipeline stage) are preserved so the
        # SFA_other prefix still carries the stage coordinate.
        SFA_word = T.view(
            SFA_data,
            list(SFA_data.shape[:-1]) + [SFA_data.shape[-1] // 4],
            "uint32",
        )
        SFB_word = T.view(
            SFB_data,
            list(SFB_data.shape[:-1]) + [SFB_data.shape[-1] // 4],
            "uint32",
        )

        @T.macro
        def _load_kblock(A_local_buf, B_local_buf, SFA_local_buf, SFB_local_buf, k_block):"""

assert anchor in src, "anchor A missing"
src = src.replace(anchor, replace, 1)

aold = """            for g in T.unroll(sfa_words):
                scale_a_word = self._tile_kmajor_scale_word(scale_m0 + (g * 32 + (qlane // 2) * 16) % self.warp_row_tiles, k_block, tile_m)
                SFA_local_buf[g] = SFA_data[
                    tuple(SFA_other) + (SFA_base_m + scale_a_word // k_blocks, SFA_base_k + scale_a_word % k_blocks)
                ]"""
anew = """            for g in T.unroll(sfa_words):
                scale_a_row = scale_m0 + (g * 32 + (qlane // 2) * 16) % self.warp_row_tiles
                SFA_local_buf[g] = SFA_word[
                    tuple(SFA_other) + (SFA_base_m + scale_a_row, SFA_base_k // 4 + k_block)
                ]"""
assert aold in src, "anchor B missing"
src = src.replace(aold, anew, 1)

bold = """            for g in T.unroll(sfb_words):
                scale_b_word = self._tile_kmajor_scale_word(scale_n0 + (g * 32 + qlane * 8) % self.warp_col_tiles, k_block, tile_n)
                SFB_local_buf[g] = SFB_data[
                    tuple(SFB_other) + (SFB_base_n + scale_b_word // k_blocks, SFB_base_k + scale_b_word % k_blocks)
                ]"""
bnew = """            for g in T.unroll(sfb_words):
                scale_b_col = scale_n0 + (g * 32 + qlane * 8) % self.warp_col_tiles
                SFB_local_buf[g] = SFB_word[
                    tuple(SFB_other) + (SFB_base_n + scale_b_col, SFB_base_k // 4 + k_block)
                ]"""
assert bold in src, "anchor C missing"
src = src.replace(bold, bnew, 1)

open(F, "w").write(src)
print("PATCHED v3: packed uint32 + row/kblock + stage-safe view")
