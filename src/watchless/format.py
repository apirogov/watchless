"""Document formatting: captions in, markdown out.

Pure functions only: no network, no filesystem, no globals, so the output
format is testable without touching YouTube.

The document is a set of peer sections, in a fixed order:

    # Title
    - URL: ...
    - Channel: ...
    - Track via: ...

    ## Summary      (only when a summary ran)
    ## Description  (only when the description was wanted)
    ## Transcript   (always)
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from watchless.fetch import Segment
from watchless.settings import (
    DEFAULT_MAX_SLUG,
    GAP_S,
    MAX_UNITS,
    MAX_WORDS_PER_UNIT,
)


def video_id(url: str) -> str:
    """Extract the 11-character id from any YouTube URL shape, or a bare id."""
    for pattern in (
        r"(?:v=|/shorts/|youtu\.be/|/embed/|/live/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ):
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"could not find a video id in: {url}")


def slugify(text: str, max_len: int = DEFAULT_MAX_SLUG) -> str:
    """Filesystem-safe, ASCII-ish slug. Cuts on a word boundary, never mid-word."""
    if max_len <= 0:
        return ""
    # German umlauts -> ASCII rather than dropping the character entirely.
    umlauts: dict[int, str] = {
        ord("ä"): "ae",
        ord("ö"): "oe",
        ord("ü"): "ue",
        ord("Ä"): "Ae",
        ord("Ö"): "Oe",
        ord("Ü"): "Ue",
        ord("ß"): "ss",
    }
    translated = text.translate(umlauts)
    normalised = unicodedata.normalize("NFKD", translated)
    ascii_text = normalised.encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-").lower()
    if len(slug) > max_len:
        cut = slug[:max_len]
        # Prefer a word boundary if one is reasonably close to the limit.
        if "-" in cut[max_len // 2 :]:
            cut = cut[: cut.rfind("-")]
        slug = cut.strip("-")
    return slug


def dest_path(
    out: str | None,
    out_dir: str | None,
    use_slug: bool,
    title: str | None,
    vid: str,
    max_slug: int,
) -> Path | None:
    """Where the transcript will be written. Pure: no I/O, no network.

    ``None`` means "write to stdout".
    """
    if out:
        return Path(out)
    if out_dir:
        slug = slugify(title, max_slug) if (use_slug and title) else ""
        name = f"{slug}--{vid}.txt" if slug else f"{vid}.txt"
        return Path(out_dir) / name
    return None


def fmt_ts(seconds: float) -> str:
    """Seconds -> ``m:ss``, or ``h:mm:ss`` past an hour."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _emit(start: float, sentences: Sequence[str], timestamps: bool) -> str:
    text = " ".join(sentences)
    return f"**[{fmt_ts(start)}]** {text}" if timestamps else text


def reflow(segments: Sequence[Segment], timestamps: bool = False) -> str:
    """Segments -> paragraphs split on pauses and sentence ends.

    With ``timestamps=True`` each paragraph is prefixed with the time of the
    sentence that began it, so a doubtful passage can be located.

    A unit is one segment, or one sentence where the track is punctuated:
    auto-generated tracks usually are not, so cutting on sentence ends alone
    left an entire unpunctuated transcript as a single 1600-word block. Every
    unit still carries its own time, which is what makes both the pause test
    below and the timestamps correct.

    The cap falls on segment boundaries, never inside a segment: a sentence
    that straddles segments is kept whole, so a paragraph may exceed the cap by
    one segment's worth of text.

    An earlier version grouped whole segments, so a paragraph spanning several
    segments inherited only the first one's time: hence "all 0:00".
    """
    units = _units(segments)
    if not units:
        return ""
    return _paragraphs(units, timestamps)


def _units(segments: Sequence[Segment]) -> list[tuple[float, str]]:
    """Segments -> sentences with their own timing.

    A unit is one completed sentence, assembled across segments so a sentence
    straddling two of them is kept whole. A leftover that never reaches a full
    stop is capped in words -- unpunctuated tracks never reach one, and without
    the cap the whole transcript became a single 1600-word paragraph.
    """
    units: list[tuple[float, str]] = []
    buffer: list[str] = []
    buffer_start: float | None = None

    for segment in segments:
        text = segment.text.replace("\n", " ").strip()
        if not text:
            continue
        if buffer_start is None:
            buffer_start = segment.start
        buffer.append(text)

        pieces = re.split(r"(?<=[.!?])\s+", " ".join(buffer))
        # Everything but the tail is a finished sentence.
        for piece in pieces[:-1]:
            if piece.strip():
                at = buffer_start if buffer_start is not None else segment.start
                units.append((at, piece.strip()))
            # The remainder starts somewhere inside this segment; its exact
            # time is not in the captions, so it inherits the segment's start.
            buffer_start = segment.start
        buffer = [pieces[-1].strip()] if pieces[-1].strip() else []

        if buffer and len(" ".join(buffer).split()) >= MAX_WORDS_PER_UNIT:
            at = buffer_start if buffer_start is not None else segment.start
            units.append((at, " ".join(buffer)))
            buffer, buffer_start = [], None

    if buffer:
        units.append(
            (buffer_start if buffer_start is not None else 0.0, " ".join(buffer))
        )
    return units


def _paragraphs(units: Sequence[tuple[float, str]], timestamps: bool) -> str:
    """Units -> the rendered body, breaking on a real pause or at the cap."""
    out: list[str] = []
    paragraph: list[str] = []
    paragraph_start = units[0][0]
    previous = units[0][0]

    for start, unit in units:
        if paragraph and start - previous > GAP_S * 2:
            out.append(_emit(paragraph_start, paragraph, timestamps))
            paragraph = []
        if not paragraph:
            paragraph_start = start
        paragraph.append(unit)
        previous = start
        if len(paragraph) >= MAX_UNITS:
            out.append(_emit(paragraph_start, paragraph, timestamps))
            paragraph = []

    if paragraph:
        out.append(_emit(paragraph_start, paragraph, timestamps))
    return "\n\n".join(out)


@dataclass(frozen=True, slots=True)
class Document:
    """The material for one rendered file.

    Grouping this beats an eight-argument ``render()`` call: two of those
    arguments were optional strings that could be silently swapped at a call
    site, which is exactly the kind of bug a dataclass prevents.
    """

    vid: str
    url: str
    segments: Sequence[Segment]
    title: str | None = None
    channel: str | None = None
    backend: str | None = None
    description: str | None = None
    summary: str | None = None
    with_header: bool = True
    timestamps: bool = False


def render(doc: Document) -> str:
    """Assemble the final markdown for ``doc``.

    Without a header there is no metadata block for the other sections to
    anchor to, so only the summary keeps a heading and the transcript is the
    bare body. That also means ``--no-header`` drops the description: the
    caller may still have fetched one (it is summary input), but there is
    nowhere to put it.
    """
    body = reflow(doc.segments, timestamps=doc.timestamps)

    if not doc.with_header:
        if doc.summary:
            return f"## Summary\n\n{doc.summary.strip()}\n\n{body}\n"
        return body + "\n"

    head = [f"# {doc.title}" if doc.title else f"# YouTube {doc.vid}"]
    meta = [f"- URL: {doc.url}"]
    if doc.channel:
        meta.append(f"- Channel: {doc.channel}")
    if doc.backend:
        meta.append(f"- Track via: {doc.backend}")

    parts = ["\n".join(head + meta)]
    if doc.summary:
        parts.append("## Summary\n\n" + doc.summary.strip())
    if doc.description:
        parts.append("## Description\n\n" + doc.description.strip())
    parts.append("## Transcript\n\n" + body)
    return "\n\n".join(parts) + "\n"
