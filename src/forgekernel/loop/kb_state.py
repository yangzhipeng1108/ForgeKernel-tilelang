"""Machine-readable state of a KB entry: does the interface work, not exist, or lie?

The KB already records all three outcomes, but only as prose in titles and an
ad-hoc ``✅`` / ``⚠️`` column. That is enough for a human reading the index and
not enough for a student deciding what to do: "CLC compiles to nothing" and
"NVFP4 reads the wrong byte" call for opposite actions (route around it vs. patch
it), and neither is distinguishable from "measured, works" once the reader is an
agent scanning a table.

So the state is an explicit, greppable line in the entry itself, and the index
column is derived from it. The three values are not a quality ranking; they are
three different instructions:

- ``WORKS``  -- measured on hardware and ready to use as written.
- ``ABSENT`` -- the interface does not exist on this target, or exists and
  compiles to nothing. Do not use it; find another mechanism.
- ``BROKEN`` -- the interface exists and produces wrong results. Do not use it
  as shipped; apply the fix the entry records, or wait for a version bump.

``PARTIAL`` is the one value that is not an instruction but a warning: some of
the claimed surface works and some does not, and the entry says which is which.
It exists because a run can legitimately ship an entry mid-way (the MXFP4
source build is the current example) and silently filing that under WORKS would
be the worst outcome for a reader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

WORKS = "WORKS"
ABSENT = "ABSENT"
BROKEN = "BROKEN"
PARTIAL = "PARTIAL"

STATES = (WORKS, ABSENT, BROKEN, PARTIAL)

# The mark an entry wears in an index table, and the one-line instruction that
# goes with it. Kept next to the states so a new state cannot be added without
# deciding how it reads.
MARKS = {WORKS: "✅", ABSENT: "⛔", BROKEN: "🐛", PARTIAL: "⚠️"}

INSTRUCTIONS = {
    WORKS: "use as written",
    ABSENT: "do not use — this interface is not there on this target",
    BROKEN: "do not use as shipped — apply this entry's fix",
    PARTIAL: "read this entry before using — part of the surface is not ready",
}

# ``> State: WORKS`` (or ABSENT / BROKEN / PARTIAL), optionally followed by a
# one-line explanation after an em dash or hyphen. Deliberately anchored at the
# line start with a blockquote so a stray ", not BROKEN" in prose cannot match.
_STATE_LINE = re.compile(
    r"^>\s*State:\s*(" + "|".join(STATES) + r")\b[^\n]*$",
    re.MULTILINE,
)

# Just the ``> State: VALUE`` prefix, leaving the em-dash explanation on the line.
# Used when measuring a BROKEN / PARTIAL explanation: the whole-line pattern above
# would swallow the very sentence it is supposed to count.
_STATE_PREFIX = re.compile(
    r"^>\s*State:\s*(" + "|".join(STATES) + r")\b[ \t—-]*",
    re.MULTILINE,
)

# A mark in a table cell, optionally followed by text. Used to derive the index
# column, never to decide an entry's state -- the entry is the source of truth.
_MARK_CELL = re.compile(r"^\s*(" + "|".join(re.escape(m) for m in MARKS.values()) + r")")

MARK_TO_STATE = {mark: state for state, mark in MARKS.items()}


@dataclass(frozen=True)
class EntryState:
    state: str
    line: int

    @property
    def mark(self) -> str:
        return MARKS[self.state]

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS[self.state]


def declared_state(text: str) -> EntryState | None:
    """The state this entry declares, or None if it declares none.

    None is not an error for a recipe written before this existed; it is an
    error for one written after, which is what ``lint_state`` enforces at the
    points where new content enters.
    """
    match = _STATE_LINE.search(text)
    if match:
        line = text.count("\n", 0, match.start()) + 1
        return EntryState(state=match.group(1), line=line)
    section = _state_heading(text)
    if section is None:
        return None
    body = section[1].strip()
    first = body.splitlines()[0].strip() if body else ""
    word = first.strip("`*_ ").upper()
    if word in STATES:
        return EntryState(state=word, line=section[0])
    return None


def index_mark_state(cell: str) -> str | None:
    """The state an index table cell's leading mark stands for, if any."""
    match = _MARK_CELL.match(cell)
    if not match:
        return None
    return MARK_TO_STATE[match.group(1)]


def _state_heading(text: str) -> tuple[int, str] | None:
    """Locate a ``## State`` section: (line number, its body)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() in ("## state", "## status"):
            body = []
            for follow in lines[i + 1:]:
                if follow.startswith("## "):
                    break
                body.append(follow)
            return i + 1, "\n".join(body)
    return None


def lint_state(text: str, path: str = "") -> list[str]:
    """Findings for a declared state: unknown values and unexplained ones.

    An entry may declare its state in the header blockquote (``> State: BROKEN``,
    preferred) or as a ``## State`` section whose first non-empty line is the
    value. A BROKEN or PARTIAL entry must also say what is wrong within that
    section, because the mark alone tells a reader to stop and nothing about
    what to do next.
    """
    findings: list[str] = []
    declared = declared_state(text)
    if declared is None:
        section = _state_heading(text)
        if section is not None:
            body = section[1].strip()
            first = body.splitlines()[0].strip() if body else ""
            word = first.strip("`*_ ").upper()
            if word and word not in STATES:
                findings.append(
                    f"{path}:{section[0]}: '## State' must open with one of "
                    f"{', '.join(STATES)}, got {first!r}"
                )
    if declared is None:
        return findings

    if declared.state in (BROKEN, PARTIAL):
        section = _state_heading(text)
        if section is not None:
            body = section[1].strip()
            # Drop the leading state word itself; it is not an explanation.
            body = "\n".join(body.splitlines()[1:]) if body else ""
        else:
            # The header blockquote: the declared line plus any continuation
            # lines it wraps onto, so a one-sentence explanation counts.
            lines = text.splitlines()
            first = declared.line - 1
            body_lines = []
            for line in lines[first:]:
                stripped = line.strip()
                if body_lines and not stripped.startswith(">"):
                    break
                body_lines.append(stripped)
            body = "\n".join(body_lines)
            body = _STATE_PREFIX.sub("", body)
        body = body.lstrip(">").replace(MARKS[declared.state], "").strip("`*_ \n")
        if len(body.split()) < 8:
            findings.append(
                f"{path}:{declared.line}: state {declared.state} needs a concrete "
                f"explanation (>= 8 words: what is wrong, and what a reader should "
                f"do instead)"
            )
    return findings


def lint_index_states(index_text: str, path: str = "",
                      root: Path | None = None) -> list[str]:
    """Findings for an index table whose marks disagree with the entries.

    Each row is ``| Interface | `entry.md` | evidence |``. The third column is
    where the mark lives today, so a row naming an entry must open that column
    with the same mark the entry declares. An entry with no declared state is
    reported too: the index is the one place a reader looks first, and a row
    with no instruction next to it is exactly the gap this is meant to close.

    ``root`` is the directory the entry names are relative to, normally the
    directory holding the index. Rows naming a file that is not there are
    skipped, so an index may list a planned entry before it is written.
    """
    findings: list[str] = []
    base = root if root is not None else (Path(path).parent if path else Path("."))
    for number, line in enumerate(index_text.splitlines(), start=1):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        name = next((c.strip("`*_ ") for c in cells if c.strip("`*_ ").endswith(".md")), None)
        if not name:
            continue
        entry = base / name
        if not entry.is_file():
            continue
        mark_state = index_mark_state(cells[2])
        try:
            declared = declared_state(entry.read_text(encoding="utf-8"))
        except OSError:
            continue
        if declared is None:
            findings.append(
                f"{path}:{number}: {name} has no '> State:' line, so this "
                f"row cannot say what to do with it"
            )
        elif mark_state != declared.state:
            findings.append(
                f"{path}:{number}: {name} declares {declared.state} but the "
                f"row's mark reads {mark_state or 'nothing'} "
                f"(expected {declared.mark} {declared.state})"
            )
    return findings
