"""Diagnostics.

All human-facing chatter goes to stderr, always. stdout carries exactly one
thing -- the path of the file that was written -- so a caller can do
``file=$(watchless URL --out-dir ~/t)`` without filtering. This split is why
``log`` exists rather than bare ``print`` calls scattered around.
"""

from __future__ import annotations

import sys

# Set from --verbose / WATCHLESS_VERBOSE. Off by default: a normal run should
# print the path and nothing else.
verbose = False


def log(message: str) -> None:
    """Progress and diagnostics. Never goes to stdout."""
    print(message, file=sys.stderr)


def debug(message: str) -> None:
    """Like :func:`log`, but only when verbose."""
    if verbose:
        print(message, file=sys.stderr)
