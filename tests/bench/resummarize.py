#!/usr/bin/env python3
"""Summarise an already-saved watchless file. No YouTube, no re-fetch.

The pipeline is deliberately two steps:

    watchless.py URL              -> 0001-Title.md      (transcript + description)
    resummarize.py FILE.md     -> re-summarised in place / to stdout

That split means a prompt change costs zero YouTube calls and zero transcripts
need re-downloading. It also makes prompt iteration cheap: run it over a whole
directory of saved files and diff the results.

    python3 resummarize.py corpus/06-health-claim.md
    python3 resummarize.py corpus/            # every .md in the directory
    python3 resummarize.py corpus/ --dry-run  # show what would be sent
    python3 resummarize.py f.md --bullets 20 --show-prompt

Input format is what watchless writes:

    # Title
    - URL: ...
    - Channel: ...

    ## Summary      (optional, replaced on rewrite)
    ## Description  (optional, used as input)
    ## Transcript   (required)

--in-place replaces the existing `## Summary` section; without it the result
goes to stdout. Endpoint config comes from the environment (see
~/.config/watchless.env) or the same --llm-* flags watchless accepts.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import TypedDict

from tests.bench import report
from watchless import summarize as S


class Doc(TypedDict):
    """One saved watchless file, split into its parts."""

    title: str | None
    channel: str | None
    description: str
    transcript: str | None


def parse_doc(md: str) -> Doc:
    """Split a watchless document into title / channel / description / transcript."""
    title = channel = description = transcript = None

    m = re.search(r"^#\s+(.+)$", md, re.MULTILINE)
    if m:
        title = m.group(1).strip()
    m = re.search(r"^-\s*Channel:\s*(.+)$", md, re.MULTILINE)
    if m:
        channel = m.group(1).strip()

    description = report.section(md, "Description") or ""
    transcript = report.section(md, "Transcript")
    if description is None:
        # A --no-header file carries the description nowhere: render() drops the
        # whole metadata block, so only the Summary and the body are written.
        # There is nothing to recover, and guessing would feed the summary back
        # in as if it were source material.
        description = ""
    if not transcript and not title:
        # --no-header output: render() drops the metadata block, so a Summary
        # section has no heading after it to bound it -- the body simply
        # follows after a blank line. Falling back to the WHOLE file fed the
        # previous summary back in as if it were transcript, so every rerun
        # summarised its own output.
        transcript = re.sub(
            r"\A##\s+Summary\s*\n\n", "", md, count=1, flags=re.IGNORECASE
        )
        # the summary itself is the run of non-blank lines that follows it
        transcript = re.sub(r"\A(?:[^\n]+\n)+\n", "", transcript, count=1)
        if transcript == md:
            transcript = md  # no Summary section: the body is all
    return {
        "title": title,
        "channel": channel,
        "description": description,
        "transcript": (transcript or "").strip() or None,
    }


# What may appear after a blank line and still belong to the summary: another
# bullet, the opening TL;DR line, or the closing Trust line. Tested against the
# raw line.
#
# The TL;DR branch was added when that line was introduced. It is an
# unindented metadata line that is neither a bullet nor "Trust", so the scan
# would not treat it as part of the section. The Trust branch is unchanged:
# the leading \*{0,2} already covered the bolded "**Trust:**" form.
CONTINUES_SECTION = re.compile(
    r"^\s*(?:[-*•+]\s"
    r"|\*{0,2}tl\s*;?\s*dr\b"
    r"|\*{0,2}(?:assessment|verdict|(?:how far to )?trust)\s*[:\-–—])",
    re.IGNORECASE,
)


def rewrite(md: str, summary: str) -> str:
    """Replace the ## Summary section, leaving everything after it intact.

    The summary body is found by scanning lines, not by one pattern. It may
    contain a blank line -- the Trust line sits in its own paragraph so
    Markdown does not fold it into the last bullet -- but only the Trust line
    and continuing bullets may follow a blank. Anything else ends the section.

    Bounding the match on the next `##` heading instead is wrong for
    --no-header files: they have no heading after the summary, so the whole
    transcript sat inside the replaced region and --in-place silently deleted
    it.
    """
    block = "## Summary\n\n" + summary.strip() + "\n"
    lines = md.splitlines(keepends=True)

    head = next(
        (
            i
            for i, ln in enumerate(lines)
            if re.match(r"^##\s+Summary\s*$", ln.rstrip("\n"), re.IGNORECASE)
        ),
        None,
    )
    if head is not None:
        i = head + 1
        while i < len(lines) and not lines[i].strip():
            i += 1

        while i < len(lines):
            line = lines[i]
            if not line.strip():
                # A blank line only continues the section if what follows is
                # still part of it: a further bullet, or the Trust line.
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and CONTINUES_SECTION.match(lines[j]):
                    i = j
                    continue
                break
            if line.lstrip().startswith("#"):
                break
            i += 1

        return "".join(lines[:head]) + block + "".join(lines[i:])

    # no existing Summary section: insert before Description, else Transcript
    for name in ("Description", "Transcript"):
        m = re.search(rf"^##\s+{name}\s*$", md, re.MULTILINE | re.IGNORECASE)
        if m:
            return md[: m.start()] + block + "\n" + md[m.start() :]
    return md.rstrip() + "\n\n" + block


def one(path: Path, args: argparse.Namespace) -> tuple[str, str]:
    """-> (status, message). Never raises."""
    try:
        md = path.read_text(encoding="utf-8")
    except OSError as e:
        return "fail", f"cannot read: {e}"

    doc = parse_doc(md)
    if not doc["transcript"]:
        return "fail", "no ## Transcript section"
    if args.dry_run:
        return "dry", (
            f"title={doc['title']!r} "
            f"desc={len(doc['description'] or '')}c "
            f"text={len(doc['transcript'])}c"
        )

    # The classification is captured rather than dropped. It is not written to
    # the .md (that is our vocabulary, not the reader's), so if this command
    # discards it the corpus cannot be scored at all -- a re-summarised file
    # would report "detected: -" and the report would claim a perfect run
    # computed from zero detections.
    kind: list[str] = []
    try:
        summary, trimmed = S.summarize(
            doc["transcript"],
            base_url=args.llm_url,
            api_key=args.llm_key,
            model=args.llm_model,
            max_bullets=args.bullets,
            max_chars=args.max_chars,
            timeout=args.llm_timeout,
            title=doc["title"],
            channel=doc["channel"],
            description=doc["description"],
            kind_out=kind,
        )
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001
        return "fail", f"{type(e).__name__}: {e}"

    if not summary.strip():
        return "fail", "empty summary (check the transcript)"

    if kind:
        # Same diagnostic cli.py emits, so report.parse_case can score it.
        print(f"[kind] {kind[0]}", file=sys.stderr)
    note = " (trimmed)" if trimmed else ""
    if args.in_place:
        path.write_text(rewrite(md, summary), encoding="utf-8")
        return "ok", f"rewrote {path.name}{note}"
    print(f"\n===== {path.name} =====\n{summary}")
    return "ok", f"printed{note}"


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m tests.bench.resummarize",
        description="Re-summarise saved watchless files without touching YouTube.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Input format")[0],
    )
    ap.add_argument("path", help="a saved .md file, or a directory of them")
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="replace the file's ## Summary section instead of printing to stdout",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="parse the files and report sizes; call nothing",
    )
    ap.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the exact prompt for the first file and exit",
    )
    ap.add_argument("--pattern", default="*.md", help="glob for directories")
    ap.add_argument("--bullets", type=int, default=S.MAX_BULLETS, metavar="N")
    ap.add_argument("--max-chars", type=int, default=S.DEFAULT_MAX_CHARS, metavar="N")
    ap.add_argument("--llm-timeout", type=int, default=S.TIMEOUT_S, metavar="SEC")
    ap.add_argument("--llm-url", metavar="URL")
    ap.add_argument("--llm-key", metavar="TOKEN")
    ap.add_argument("--llm-model", metavar="NAME")
    args = ap.parse_args()

    target = Path(args.path)
    if target.is_dir():
        files = sorted(p for p in target.glob(args.pattern) if p.is_file())
        if not files:
            raise SystemExit(f"no {args.pattern} files in {target}")
    elif target.is_file():
        files = [target]
    else:
        raise SystemExit(f"not found: {target}")

    if args.show_prompt:
        doc = parse_doc(files[0].read_text(encoding="utf-8"))
        prompt, _ = S.build_prompt(
            doc["transcript"] or "",
            args.bullets,
            args.max_chars,
            title=doc["title"],
            channel=doc["channel"],
            description=doc["description"],
        )
        print(prompt)
        return

    failed = 0
    for i, p in enumerate(files, 1):
        status, msg = one(p, args)
        if status == "fail":
            failed += 1
        if status != "ok" or not args.in_place:
            print(f"[{i}/{len(files)}] {p.name}: {msg}", file=sys.stderr)
        else:
            print(f"[{i}/{len(files)}] {msg}", file=sys.stderr)

    if len(files) > 1 or failed:
        print(
            f"\n{len(files) - failed}/{len(files)} ok, {failed} failed", file=sys.stderr
        )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
