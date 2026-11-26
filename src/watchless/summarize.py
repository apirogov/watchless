"""Prompt and output contract: transcript in, tagged bullets out.

Assembles the prompt from title, channel, description and text, then coerces
the model's reply into the shape the rest of the program expects.

Transport -- HTTP, retries, endpoints, model names -- is :mod:`watchless.llm`.
"""

from __future__ import annotations

import re
from importlib.resources import files

from watchless import llm
from watchless.llm import SummarizeError, call, resolve
from watchless.log import debug
from watchless.settings import (
    DEFAULT_MAX_BULLETS as MAX_BULLETS,
)
from watchless.settings import (
    DEFAULT_MAX_CHARS,
    DESC_MAX_CHARS,
    SUMMARY_FLOOR_BULLETS,
    SUMMARY_TARGET_RATIO,
    SUMMARY_WORDS_PER_BULLET,
    TIMEOUT_S,
)

# The public surface is the prompt/summary pair. ``call``/``resolve``/
# ``SummarizeError`` are re-exported from :mod:`watchless.llm` for callers that
# think of "summarising" as one import; new code should import them from llm.
__all__ = [
    "DEFAULT_MAX_CHARS",
    "DESC_MAX_CHARS",
    "MAX_BULLETS",
    "PROMPT",
    "TIMEOUT_S",
    "SummarizeError",
    "budget_for",
    "build_prompt",
    "call",
    "normalize_bullets",
    "resolve",
    "summarize",
    "trim",
]

# --------------------------------------------------------------------------
# EDIT ME: the summarisation prompt. {n} is the bullet budget for THIS
# transcript and {target} the word budget (both from budget_for); {title}
# {channel} {description} {text} with the video's material.
# Keep the output contract explicit -- small local models drift otherwise.
# --------------------------------------------------------------------------
_PROMPTS = files("watchless.prompts")
# The prompt is data the model reads, kept in .md beside this module so editing
# it is not a code edit.
PROMPT = _PROMPTS.joinpath("summarize.md").read_text(encoding="utf-8")

# Appended only when scoring against the corpus: asks the model to classify the
# video so a harness can compare against an expectation, then is stripped.
KIND_PROMPT = _PROMPTS.joinpath("kind.md").read_text(encoding="utf-8")


def trim(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, bool]:
    """Keep head and tail, drop the middle. Returns (text, was_trimmed)."""
    if len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n[... middle of transcript omitted for length ...]\n\n"
        + text[-half:],
        True,
    )


def budget_for(text: str, ceiling: int = MAX_BULLETS) -> tuple[int, int]:
    """Bullet count and word target for this transcript.

    A flat ceiling ignores how much there is to summarise: it is generous on a
    short video and punishing on a long one. Scale with the transcript instead,
    but never below SUMMARY_FLOOR_BULLETS -- a 450-word stand-up set summarised
    at a literal 10% is two bullets, which is not a summary of anything.

    Returns (bullets, words) where words includes only the content bullets; the
    TL;DR and the Trust line are outside the budget.
    """
    tr = SUMMARY_TARGET_RATIO
    per = SUMMARY_WORDS_PER_BULLET
    words = int(len(text.split()) * tr)
    n = max(SUMMARY_FLOOR_BULLETS, min(ceiling, round(words / per)))
    return n, max(words, n * per // 2)


def build_prompt(
    text: str,
    max_bullets: int = MAX_BULLETS,
    max_chars: int = DEFAULT_MAX_CHARS,
    *,
    title: str | None = None,
    channel: str | None = None,
    description: str | None = None,
    classify: bool = False,
) -> tuple[str, bool]:
    """Assemble the request.

    Title, channel and description are separate labelled fields, never glued
    onto the transcript: the model must be able to tell speech from metadata.

    ``classify`` appends the benchmark's self-classification instruction. It is
    off by default because a reader never needs the model to label its own work,
    and asking costs tokens and invites the model to explain itself.
    """
    body, trimmed = trim(text, max_chars)
    desc = (description or "").strip()
    if len(desc) > DESC_MAX_CHARS:
        desc = desc[:DESC_MAX_CHARS] + "\n[... description truncated for length ...]"
    n, target = budget_for(text, max_bullets)
    template = PROMPT + KIND_PROMPT if classify else PROMPT
    return template.format(
        n=n,
        target=target,
        title=(title or "").strip() or "(none provided)",
        channel=(channel or "").strip() or "(none provided)",
        description=desc or "(none provided)",
        text=body,
    ), trimmed


# "[fact]" promises verification this tool cannot perform: every assertion the
# speaker makes is a claim. The prompt asks for [claim]; the tag is corrected
# anyway, because a model's habits outlast an instruction.
FACT_TAG_RE = re.compile(r"\[fact\]", re.IGNORECASE)

# The classification, machine-read only and never written to the file. Accepts
# the prompt's words or the bare letters a model echoes back, with an optional
# separator: "KIND: A", "**KIND** - B", "KIND A".
KIND_LINE_RE = re.compile(
    r"^kind\s*[:\-–—]?\s*(information|performance|both|a|b)\b", re.I
)
KIND_WORD = {
    "information": "A",
    "a": "A",
    "performance": "B",
    "b": "B",
    "both": "BOTH",
}
# The closing verdict. "Assessment" is the label the prompt asks for; the older
# "Trust" and the longer "How far to trust" are still recognised so a model that
# echoes either normalises instead of leaving a stray bullet.
#
# The separator is required: a content bullet can legitimately begin with the
# bare word ("- Trust between the two presenters is the real subject").
ASSESSMENT_LINE_RE = re.compile(
    r"^(?:assessment|verdict|(?:how far to )?trust)\s*[:\-–—]", re.IGNORECASE
)
ASSESSMENT_STRIP_RE = re.compile(
    r"^(?:assessment|verdict|how far to trust|trust)\s*[:\-–—]*\s*", re.IGNORECASE
)

# The TL;DR opens the summary, carries no tag, and does not count against the
# bullet budget. Several spellings, because models vary: "TL;DR:", "TLDR:",
# "**TL;DR** -".
TLDR_LINE_RE = re.compile(r"^tl\s*;?\s*dr\b\s*[:\-–—]*\s*", re.IGNORECASE)


def normalize_bullets(
    text: str, max_bullets: int = MAX_BULLETS, *, kind_out: list[str] | None = None
) -> str:
    """Force '- ' bullets, keep the TL;DR and Assessment, cap the count.

    Models drift: they emit bold headers, numbered lists, '•', and markdown
    '#' headings. Headings and one-word headings like '**Summary:**' are
    dropped rather than turned into bullets -- they are not summary points.

    The TL;DR and the closing Assessment are metadata rather than content and
    keep their own lines. Neither counts against max_bullets, which caps content
    bullets only -- otherwise the model spends its budget on content and drops
    the assessment, which is the part worth reading.

    The leading KIND line is also metadata, but of a kind the reader must never
    see. When ``kind_out`` is given it is captured there as "A" / "B" / "BOTH"
    (the benchmark's spelling) and removed from the output; without it the line
    is simply dropped.
    """
    case_line = trust_line = tldr_line = None
    bullets: list[str] = []

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        # markdown heading -> not a bullet
        if s.startswith("#"):
            continue

        # Normalise list markers and emphasis once, then test for the two
        # metadata lines: e.g. "**KIND: B**" and "- Trust: ...".
        clean = re.sub(r"^[-*•+]+\s*", "", s)
        clean = re.sub(r"^\d+[.)]\s*", "", clean)
        clean = re.sub(r"\*\*|__", "", clean).strip().strip("`").strip()

        if case_line is None and KIND_LINE_RE.match(clean):
            case_line = KIND_WORD[KIND_LINE_RE.match(clean).group(1).lower()]  # type: ignore[union-attr]
            continue
        if tldr_line is None and TLDR_LINE_RE.match(clean):
            # Kept on its own line ahead of the bullets, never tagged: it is
            # the summarizer's voice, and the bullets expand it.
            tldr_line = "**TL;DR:** " + TLDR_LINE_RE.sub("", clean, count=1)
            continue
        if ASSESSMENT_LINE_RE.match(clean):
            # Normalise the label to exactly "Trust: ..." whatever separator
            # the model used (colon, dash, em dash, bold, nothing). Top-level so
            # it always lands last, after the content bullets.
            verdict = ASSESSMENT_STRIP_RE.sub("", clean, count=1)
            trust_line = "**Assessment:** " + verdict
            continue

        s = clean

        # No bullet may be tagged a fact. See FACT_TAG_RE.
        s = FACT_TAG_RE.sub("[claim]", s)

        if not s:
            continue

        # a lone label like 'Summary:' or 'Key points' is not a bullet
        if re.fullmatch(r"[\w \-/]{1,24}:", s):
            continue
        if len(s.split()) <= 2 and s.endswith(":"):
            continue

        bullets.append(f"- {s}")

    if kind_out is not None and case_line:
        kind_out.append(case_line)
    # Both metadata lines are bolded, and both are pushed outside the bullet
    # block by a blank line. The blank line is not cosmetic: without it
    # Markdown treats a following unindented line as a LAZY CONTINUATION of the
    # final bullet -- no indentation, but still inside the list item -- so the
    # line renders as the tail of the last bullet, which is both wrong to read
    # and wrong about what it is.
    body = "\n".join(bullets[:max_bullets])
    return "\n\n".join(filter(None, [tldr_line, body, trust_line]))


def summarize(
    text: str,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_bullets: int = MAX_BULLETS,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: int = TIMEOUT_S,
    retries: int = 0,
    *,
    title: str | None = None,
    channel: str | None = None,
    description: str | None = None,
    kind_out: list[str] | None = None,
) -> tuple[str, bool]:
    """Returns (summary, was_trimmed).

    ``kind_out``, when given, receives the model's classification ("A" / "B" /
    "BOTH"). It is stripped from the returned text either way, so the caller
    that wants to score it can, and the file never shows it.
    """
    # resolve once so the preflight and the request use identical values
    _, _, resolved_model = llm.resolve(base_url, api_key, model)
    llm.preflight(base_url, api_key, resolved_model)
    prompt, trimmed = build_prompt(
        text,
        max_bullets,
        max_chars,
        title=title,
        channel=channel,
        description=description,
        # Ask for the classification only when someone is collecting it. This
        # is the one call site that knows, so the instruction travels with the
        # request rather than being baked into the shipped prompt.
        classify=kind_out is not None,
    )
    debug(f"prompt {len(prompt)} chars -> {max_bullets} bullets max")
    return normalize_bullets(
        llm.call(prompt, base_url, api_key, model, timeout=timeout, retries=retries),
        max_bullets,
        kind_out=kind_out,
    ), trimmed
