"""report: read a saved document's ## Summary section back.

The inverse of :mod:`watchless.summarize`. That module writes a CASE line plus
tagged bullets; this one reports which CASE a saved file actually claims, so a
corpus run can compare its expectation against the real output.

It lives beside the corpus harness because that is its only consumer: the
shipping path never reads a saved document back. ``section()`` parses our own
output format and ``parse_case()`` scores it, so both are development tools.
"""

from __future__ import annotations

import re

# A "## Name" section, from its heading to just before the next "## " heading.
# This is the document format's one boundary rule: report (reading the CASE
# line), resummarize (splitting Description/Transcript out, inserting a new
# Summary) and any future reader all need it, and three near-identical regexes
# had already drifted over case-sensitivity and trailing whitespace.
SECTION_RE = r"^##\s+{name}\s*$(.*?)(?=^##\s|\Z)"


def section(md: str, name: str) -> str | None:
    """The body of ``## <name>``, or None. Heading match is case-insensitive."""
    m = re.search(
        SECTION_RE.format(name=name), md, re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    return m.group(1).strip() if m else None


# What cli.py logs when it strips the model's classification: "[kind] B".
KIND_LOG_RE = re.compile(r"^\[kind\]\s*(A|B|BOTH)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_case(stderr: str) -> str:
    """Read the model's classification out of a run's stderr.

    The classification is internal: it is emitted as a ``[kind] A|B|BOTH``
    diagnostic and never written to the transcript, because "CASE B" is our
    vocabulary and a reader has no use for it. So it is read from the log, not
    from the .md.

    Returns "A", "B", "BOTH", or "-" when the run produced no classification --
    an older log from before this moved, or a model that ignored the prompt.
    "-" is deliberately not an error: the report's job is to show the gap, not
    to fail on it.
    """
    got = KIND_LOG_RE.search(stderr)
    return got.group(1).upper() if got else "-"
