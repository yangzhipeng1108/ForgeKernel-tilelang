"""The arch token helpers: parsing, KB lookup, and instruction gating.

These are pure (no torch, no GPU), so they run on a CPU box and pin the rules the
loop relies on to keep a kernel's instruction set honest with its target GPU.
"""
import pytest

from forgekernel import arch


def test_parse_accepts_both_spellings_and_normalises():
    assert arch.parse("sm_86").token == "sm86"
    assert arch.parse("sm86").token == "sm86"
    a = arch.parse("sm90a")
    assert a.token == "sm90a" and a.capability == 90 and a.suffix == "a"


def test_parse_rejects_junk():
    for bad in ("", "86", "sm", "smx", None):
        with pytest.raises(ValueError):
            arch.parse(bad)


def test_sm_code_is_the_cuda_underscore_form():
    assert arch.parse("sm86").sm_code == "sm_86"
    assert arch.parse("sm100").sm_code == "sm_100"


def test_ampere_lacks_hopper_and_blackwell_families_but_has_its_own():
    lacks = set(arch.unsupported("sm86", ("wgmma", "tma", "cluster", "tcgen05", "tmem",
                                          "cp.async", "mma.sync", "ldmatrix")))
    assert lacks == {"wgmma", "tma", "cluster", "tcgen05", "tmem"}
    # the things an sm86 kernel *may* use are not reported as missing
    assert arch.supports("sm86", "cp.async") and arch.supports("sm86", "mma.sync")


def test_hopper_has_wgmma_but_not_blackwell_tcgen05():
    assert arch.supports("sm90", "wgmma") and arch.supports("sm90", "tma")
    assert not arch.supports("sm90", "tcgen05")


def test_blackwell_lacks_nothing_in_our_table():
    assert arch.unsupported("sm100", ("wgmma", "tma", "cluster", "tcgen05", "tmem")) == []


def test_unknown_feature_is_refused_not_guessed():
    with pytest.raises(KeyError):
        arch.supports("sm86", "quantum")


def test_available_arches_lists_the_created_sm86_trees(tmp_path):
    for lang in ("tilelang", "cuda", "cutedsl"):
        (tmp_path / "hw_kb" / lang / "sm86").mkdir(parents=True)
        (tmp_path / "hw_kb" / lang / "sm100").mkdir(parents=True)
    assert "sm86" in arch.available_arches(tmp_path, "tilelang")
    assert "sm86" in arch.available_arches(tmp_path, "cuda")
