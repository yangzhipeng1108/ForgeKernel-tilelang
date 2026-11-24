"""Tests for the KB entry state: an entry must say whether its interface works,
is absent, or is broken, because those three call for different actions."""

from __future__ import annotations

from pathlib import Path

import pytest

from forgekernel.loop import kb_state
from forgekernel.loop.kb_lint import lint_kb_dir, lint_text

REPO_ROOT = Path(__file__).resolve().parents[1]

WORKS_ENTRY = "> State: WORKS\n\nmeasured on the card with numbers\n"

BROKEN_ENTRY = (
    "> State: BROKEN\n\n"
    "The scale operand is fetched as one byte instead of the packed word, so\n"
    "every result is wrong; patch the view before trusting any of this.\n"
)

ABSENT_ENTRY = (
    "> State: ABSENT\n\n"
    "The keyword does not exist in this release, so there is nothing to tune.\n"
)


def test_each_state_carries_its_own_instruction():
    instructions = {s: kb_state.INSTRUCTIONS[s] for s in kb_state.STATES}
    assert len(set(instructions.values())) == len(kb_state.STATES)
    assert "do not use" in instructions[kb_state.ABSENT]
    assert "do not use" in instructions[kb_state.BROKEN]
    assert "use as written" in instructions[kb_state.WORKS]


@pytest.mark.parametrize("text,state", [
    (WORKS_ENTRY, kb_state.WORKS),
    (BROKEN_ENTRY, kb_state.BROKEN),
    (ABSENT_ENTRY, kb_state.ABSENT),
])
def test_declared_state_reads_the_header_line(text, state):
    declared = kb_state.declared_state(text)
    assert declared is not None
    assert declared.state == state
    assert declared.line == 1


def test_state_section_is_accepted():
    text = "# e\n\n## State\n\nABSENT\n\nit is not there on this target at all\n"
    declared = kb_state.declared_state(text)
    assert declared is not None and declared.state == kb_state.ABSENT


def test_state_section_with_a_bad_word_is_a_finding():
    text = "# e\n\n## State\n\nMAYBE\n\nprose that pads the length out\n"
    findings = kb_state.lint_state(text, "e.md")
    assert findings and "must open with one of" in findings[0]


def test_a_prose_mention_is_not_a_declaration():
    text = "> State: WORKS\n\nthis is not BROKEN, the wording just says so\n"
    declared = kb_state.declared_state(text)
    assert declared is not None and declared.state == kb_state.WORKS


def test_broken_and_partial_need_an_explanation():
    for state in (kb_state.BROKEN, kb_state.PARTIAL):
        bare = f"> State: {state}\n"
        findings = kb_state.lint_state(bare, "e.md")
        assert findings and "needs a concrete explanation" in findings[0], state


def test_works_needs_no_explanation():
    assert kb_state.lint_state(WORKS_ENTRY, "e.md") == []


def test_absent_needs_no_explanation():
    assert kb_state.lint_state(ABSENT_ENTRY, "e.md") == []


def test_an_entry_without_a_state_does_not_fail_the_entry_lint():
    # Recipes written before the line existed stay readable; the index check is
    # where a missing state surfaces, because that is what a reader sees first.
    assert kb_state.lint_state("# old entry\n\nprose only\n", "e.md") == []


def _write(root: Path, name: str, body: str) -> None:
    (root / name).write_text(body, encoding="utf-8")


def test_index_mark_must_match_the_entry(tmp_path):
    _write(tmp_path, "a.md", WORKS_ENTRY)
    _write(tmp_path, "b.md", ABSENT_ENTRY)
    index = (
        "| Interface | Entry | Evidence |\n"
        "|---|---|---|\n"
        "| a | `a.md` | ✅ measured |\n"
        "| b | `b.md` | ✅ measured |\n"
    )
    findings = kb_state.lint_index_states(index, str(tmp_path / "index.md"))
    assert len(findings) == 1
    assert "b.md declares ABSENT" in findings[0]


def test_index_row_without_a_mark_on_the_entry_is_reported(tmp_path):
    _write(tmp_path, "a.md", "# no state here\n")
    index = (
        "| Interface | Entry | Evidence |\n"
        "|---|---|---|\n"
        "| a | `a.md` | ✅ measured |\n"
    )
    findings = kb_state.lint_index_states(index, str(tmp_path / "index.md"))
    assert findings and "no '> State:' line" in findings[0]


def test_index_rows_naming_no_real_entry_are_skipped(tmp_path):
    index = (
        "| Interface | Entry | Evidence |\n"
        "|---|---|---|\n"
        "| gone | `gone.md` | ✅ measured |\n"
    )
    assert kb_state.lint_index_states(index, str(tmp_path / "index.md")) == []


def test_lint_kb_dir_checks_the_index(tmp_path):
    entry = tmp_path / "sm90"
    entry.mkdir()
    _write(entry, "a.md", ABSENT_ENTRY)
    _write(entry, "index.md",
           "| Interface | Entry | Evidence |\n|---|---|---|\n"
           "| a | `a.md` | ✅ measured |\n")
    findings = lint_kb_dir(tmp_path, extra_tokens=[])
    assert any("declares ABSENT" in f for f in findings)
