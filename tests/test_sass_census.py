"""Tests for the MMA operand-source census (the [VERIFY: sass-derived] channel)."""

from forgekernel.skills.sass_census import (
    census_from_text,
    parse_census_line,
    render_census_line,
)

# Both dump styles carry both operand forms: RS (A = R8) and SS (A = gdesc[...]).
CUOBJDUMP_STYLE = """\
        Function : student_kernel
/*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;   /* 0x00000a00ff017624 */
/*0010*/                   HGMMA.64x64x16.F32 R80, R8, gdesc[UR6], !UPT ;  /* 0x... */
/*0020*/                   HGMMA.64x64x16.F32 R80, gdesc[UR4], gdesc[UR6] ;  /* 0x... */
/*0030*/                   HGMMA.64x128x16.F32 R24, R8, gdesc[UR8] ;  /* 0x... */
/*0040*/                   BAR.SYNC 0x0 ;  /* 0x... */
"""

NCU_STYLE = """\
0x7fc07f7a1800           LDC R1, c[0x0][0x28]
0x7fc07f7a1810           QGMMA.64x64x32.F32.E4M3 R40, gdesc[UR4], gdesc[UR6]
0x7fc07f7a1820           HGMMA.64x64x16.F32 R80, R12, gdesc[UR6]
"""

# sm120 (Blackwell workstation) emits warp-level `HMMA`, not the Hopper `HGMMA`
# spelling. This is the EXACT form an RTX 6000D produces from a TileLang pipeline
# GEMM (`cuobjdump -sass`), with the A operand a register. Before `HMMA` was a
# known family this census returned mma_total=0 and rendered SASS-OPERANDS-FAIL on
# a kernel running at 133 TFLOP/s -- "no MMA seen" for a healthy tensor kernel.
SM120_HMMA_STYLE = """\
        Function : k_kernel
/*1eb0*/                   HMMA.16816.F32 R144, R56, R52, R144 ;   /* 0x000000343890723c */
/*1ec0*/                   HMMA.16816.F32 R32, R56, R54, R32 ;     /* 0x000000363820723c */
/*1ed0*/                   HMMA.16816.F32 R36, R56, R48, R36 ;     /* 0x000000303824723c */
/*1ee0*/                   HMMA.16816.F32 R40, R56, R50, R40 ;     /* 0x000000323828723c */
"""


def test_census_counts_both_forms_and_styles():
    c = census_from_text(CUOBJDUMP_STYLE)
    assert c == {"mma_total": 3, "mma_a_reg": 2, "mma_a_desc": 1, "mma_a_other": 0}
    c2 = census_from_text(NCU_STYLE)
    assert c2 == {"mma_total": 2, "mma_a_reg": 1, "mma_a_desc": 1, "mma_a_other": 0}


def test_non_mma_instructions_do_not_count():
    c = census_from_text("/*0000*/  UTMALDG.2D [UR4], [R2] ;\n/*0010*/  BAR.SYNC 0x0 ;\n")
    assert c["mma_total"] == 0


def test_sm120_warp_mma_hmma_is_counted_with_a_register_source():
    # sm120's only MMA family is HMMA; a census blind to it fails a working GEMM.
    c = census_from_text(SM120_HMMA_STYLE)
    assert c == {"mma_total": 4, "mma_a_reg": 4, "mma_a_desc": 0, "mma_a_other": 0}
    line = render_census_line(c, "k_kernel")
    assert line.startswith("SASS-OPERANDS ")
    assert "kernel=k_kernel" in line and "mma_total=4" in line and "a_reg=4" in line


def test_a_register_with_a_reuse_hint_is_still_a_register():
    # ptxas tags a re-read operand `R60.reuse`; sm120 GEMMs emit exactly this form
    # for the A operand. Classifying it as `other` understated a_reg on a fully
    # register-fed kernel (measured a_reg=8 / a_other=56 before this was fixed).
    dump = (
        "/*1d00*/  HMMA.16816.F32 R128, R60.reuse, R172, R128 ;\n"
        "/*1d90*/  HMMA.16816.F32 R100, R60, R6, R100 ;\n"
    )
    c = census_from_text(dump)
    assert c == {"mma_total": 2, "mma_a_reg": 2, "mma_a_desc": 0, "mma_a_other": 0}


def test_render_and_parse_roundtrip():
    line = render_census_line(census_from_text(CUOBJDUMP_STYLE), "student_kernel")
    fields = parse_census_line(f"noise before\n{line}\nnoise after\n")
    assert fields == {"mma_census_kernel": "student_kernel", "mma_total": 3,
                      "mma_a_reg": 2, "mma_a_desc": 1, "mma_a_other": 0}


def test_zero_mma_renders_fail_line_and_parses_to_empty():
    line = render_census_line(census_from_text("no instructions here"), "k")
    assert line.startswith("SASS-OPERANDS-FAIL")
    assert parse_census_line(line) == {}


def test_parse_takes_the_last_line():
    first = render_census_line(census_from_text(NCU_STYLE), "old")
    second = render_census_line(census_from_text(CUOBJDUMP_STYLE), "new")
    fields = parse_census_line(f"{first}\n{second}\n")
    assert fields["mma_census_kernel"] == "new"
    assert fields["mma_total"] == 3


def test_absent_line_returns_empty_so_digests_stay_unchanged():
    assert parse_census_line("plain ncu csv output, no census ran") == {}
