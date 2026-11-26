#!/usr/bin/env python3
"""Internal benchmark harness: run watchless over a labelled corpus.

This is a development tool for evaluating the summary prompt against a fixed,
genre-spanning set of videos -- not a command end users run. It is therefore
not registered in ``[project.scripts]``. Invoke it from the checkout:

    uv run python -m tests.bench.corpus --check        # resolve titles, spend nothing
    uv run python -m tests.bench.corpus                # run all (needs an LLM endpoint)
    uv run python -m tests.bench.corpus --only comedy-satire,craft-showcase
    uv run python -m tests.bench.corpus --force        # rerun even if output exists

The video list is the ``VIDEOS`` table below, not an input directory: each row
pins a genre the prompt contract is supposed to handle, and ``--check`` resolves
its ytsearch query to a real video id, cached in ``resolved.json``.

Output lands in tests/bench/corpus/:
    01-explainer-science.md    watchless output (summary + description + transcript)
    01-explainer-science.log   watchless stderr (model, timings, errors)
    resolved.json              query -> video id cache, so reruns stay stable
    REPORT.md                  expected vs. detected case, per video

Stdlib + this package only. The fetch runs in-process, so watchless's flags pass
straight through, e.g.:

    python3 corpus.py --bullets 20 --llm-model deepseek-v4.1-flash \
        --llm-url https://api.neuralwatt.com/v1 --llm-key "$NEURALWATT_API_KEY"

The videos are given as ytsearch queries, not hardcoded ids: --check resolves
them once and pins the result in resolved.json. If a pick is wrong (a Short, a
re-upload), replace its query with a plain URL and delete its resolved.json
entry.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.bench.report import parse_case
from watchless import settings


def _mismatch(expected: str, got: str) -> bool:
    """Does the detected classification contradict the expectation?

    "A|B" means either verdict is acceptable for this bucket -- it is a boundary
    case, not a third category. It does NOT mean anything goes: BOTH is a real
    detected value and a bucket marked A|B that comes back BOTH was misjudged.

    The old test was `expected not in (got, "A|B")`, which put the wildcard in
    the wrong place: it excused the EXPECTATION rather than widening it, so a
    bucket expecting A|B accepted A, B and BOTH alike and could never fail.
    """
    if got == "-":
        return False  # no classification produced; shown as "-", not scored
    allowed = {"A", "B"} if expected == "A|B" else {expected.upper()}
    return got.upper() not in allowed


def _kind_from(log_path: Path) -> str:
    """The model's classification, read from a previous run's log.

    It is not in the .md -- that is deliberate, the classification is our
    vocabulary and a reader does not need it. So a cached entry can only be
    scored if the log survives beside it.
    """
    try:
        return parse_case(log_path.read_text())
    except OSError:
        return "-"


# yt-dlp ships no stubs, so its option dict has no upstream type to borrow.
Options = dict[str, Any]

# Run from the checkout: `uv run python -m tests.bench.corpus`.
#
# The directory is located from __file__ rather than by walking upward for
# pyproject.toml: a wrong path would silently create an empty corpus/ elsewhere
# and every bucket would look missing.
_HERE = Path(__file__).resolve().parent  # .../tests/bench
OUT = _HERE / "corpus"
if not OUT.is_dir():  # pragma: no cover - guards a silent, confusing failure
    raise SystemExit(f"corpus data directory not found at {OUT}")
RAW = OUT / "_raw"  # where watchless writes before the file is moved


# key, expected case, why it is in the corpus, how to find it
VideoEntry = tuple[str, str, str, str]
# One row of the report: index, key, expectation, what was detected, the video
# note, seconds taken, and the outcome. A named alias because _write_report and
# the loop both build these positionally.
Row = tuple[int, str, str, str, str, int, str]
VIDEOS: list[VideoEntry] = [
    (
        "explainer-science",
        "A",
        "CASE A extraction; most evidence is shown, not spoken",
        "ytsearch1:Kurzgesagt immune system explained",
    ),
    (
        "howto-hardware",
        "A",
        "procedure + specifics; all evidence is own experience",
        "ytsearch1:Arduino WS2812B LED matrix tutorial",
    ),
    (
        "howto-software",
        "A",
        "versions, prerequisites, links usually in description",
        "ytsearch1:git rebase tutorial intermediate",
    ),
    (
        "review-hardware",
        "A",
        "verdict vs measurement; affiliate links in description",
        "ytsearch5:mechanical keyboard review comparison sound test",
    ),
    (
        "opinion-argument",
        "A",
        "fact/opinion split; anecdote as content",
        "ytsearch1:video essay remote work productivity argument",
    ),
    (
        "health-claim",
        "A",
        "high stakes; overclaim vs hedge",
        "ytsearch1:doctor explains vitamin D evidence study",
    ),
    (
        "news-commentary",
        "A",
        "contested domain; opinion delivered as fact",
        "ytsearch8:politics panel discussion analysis week",
    ),
    (
        "interview-long",
        "A",
        "conversation; longest input, exercises the trim",
        "ytsearch1:long form interview engineer podcast",
    ),
    (
        "personal-vlog",
        "A|B",
        "A/B boundary; the story IS the content",
        # Pinned to a URL: t61V427wH_Y got flagged for automated queries and
        # its captions 429 permanently. --check would just re-pick it.
        "https://www.youtube.com/watch?v=Q3p7iBOAey4",
    ),
    (
        "comedy-satire",
        "B",
        "must describe, must not summarise",
        # Same reason: M4Rr4l5mAug is flagged. Pinned so --check cannot
        # re-resolve back onto it -- it still ranks in the search results.
        "https://www.youtube.com/watch?v=Kuqrj8uBZ5U",
    ),
    (
        "music-with-captions",
        "B",
        "CASE B where a caption file DOES exist (lyrics/applause markers)",
        "ytsearch8:official music video lyrics",
    ),
    (
        "craft-showcase",
        "B",
        "the hand-thrown mugs case",
        "ytsearch8:handmade ceramic mugs collection showcase studio tour",
    ),
    (
        "making-of",
        "both",
        "the both-branch: describe one line, then extract",
        "ytsearch1:how I made this animation behind the scenes",
    ),
    (
        "german-language",
        "A",
        "language preservation; DE transcript",
        "ytsearch5:Wissenschaft Dokumentation Erklaerung deutsch Technik",
    ),
    (
        "recipe-cooking",
        "A",
        "procedure-heavy; ingredients/quantities are the content",
        "ytsearch5:recipe cooking from scratch step by step kitchen",
    ),
    (
        "game-analysis",
        "A|B",
        "performance plus analysis; mixed signal, tests CASE BOTH",
        "ytsearch5:game design analysis why this level works",
    ),
    # --- balancing pass ---------------------------------------------------
    # Before these, the corpus was A=10 / B=3 / A|B=2 / both=1, and every
    # category held exactly ONE item -- so kind and category were confounded
    # and a pass could not be attributed to either. These fill the thin cells.
    # Kind is not fully orthogonal to category (a "performance how-to" is not
    # a real genre), so the aim is coverage of the thin cells, not a grid.
    (
        "asmr-unboxing",
        "B",
        "CASE B with no argument at all: product shown, not discussed",
        "ytsearch8:asmr unboxing first look no talking",
    ),
    (
        "silent-craft",
        "B",
        "CASE B where there is almost no speech; the object is the content",
        "ytsearch8:woodworking satisfying build no talking process",
    ),
    (
        "build-log",
        "both",
        "the second both-branch; 13-making-of was the only one",
        "ytsearch5:build log from scratch timelapse construction diary",
    ),
    (
        "panel-entertainment",
        "B",
        "CASE B: multi-host panel banter, 45min, de; nothing to extract, and at "
        "156k chars it is the largest transcript in the corpus by 4x",
        "https://www.youtube.com/watch?v=e9Vv5GrSGx0",
    ),
]

# Anything shorter is a Short/clip/trailer: no usable transcript to summarise.
MIN_S, MAX_S = 5 * 60, 75 * 60

# How many search results to consider before settling on one. Rank-1 is
# routinely a Short or a trailer.
SEARCH_DEPTH = 5


def md_candidates(url: str) -> list[Path]:
    """Files the transcript may have landed in, most specific first.

    watchless is called with --out-dir corpus/_raw and picks its own filename:
    <id>.txt, or <slug>--<id>.txt when WATCHLESS_SLUG / --slug is in play. The
    run's own log records the name it chose, so this is only a fallback for
    logs written before that was recorded, or by hand. Matching on the video
    id (not a fixed filename) is what makes it work either way.
    """
    m = re.search(r"[A-Za-z0-9_-]{11}", url)
    if not m:
        return []
    return [p for p in sorted(RAW.glob(f"*{m.group(0)}*.txt")) if p.is_file()]


def search(query: str) -> list[dict[str, Any]]:
    """Top ``SEARCH_DEPTH`` search hits, as yt-dlp's flat info dicts.

    Library call, not a binary: results arrive as dicts with named keys instead
    of three tab-separated fields re-parsed from stdout. ``extract_flat`` keeps
    it to one request per query -- it lists, it does not resolve each video.
    """
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError as exc:
        raise LookupError(
            "yt-dlp is not installed (needed to resolve ytsearch queries; "
            "or put a URL in VIDEOS instead)"
        ) from exc

    opts: Options = {
        "quiet": True,
        "no_warnings": True,
        # yt-dlp's stubs say str, its code uses store_true + truthiness.
        "skip_download": True,
        "extract_flat": True,
        "playlist_items": f"1:{SEARCH_DEPTH}",
        "retries": 0,
        "extractor_retries": 0,
    }
    proxy = settings.proxy_url()
    if proxy:
        opts["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(opts) as client:
            info = client.extract_info(query, download=False)
    except Exception as exc:
        raise LookupError(str(exc)[:200]) from exc

    entries: Any = (info or {}).get("entries") or []
    return [hit for hit in entries if isinstance(hit, dict)]


def run_watchless(argv: list[str]) -> tuple[int, str, str]:
    """Run one watchless fetch in this process: ``(exit_code, stdout, stderr)``.

    Called as a function, not spawned as a subprocess. The old version ran
    ``python -m watchless.cli`` and parsed the written path out of stdout, which
    meant a second interpreter, a second import of this package, and an
    untyped argv that had to stay in sync with the CLI by hand.

    stdout/stderr are captured because the caller wants the log file and the
    path; ``cli.main`` writes both, so they are redirected here.
    """
    from watchless import cli  # noqa: PLC0415  (cycle: cli imports settings only)

    out, err = io.StringIO(), io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cli.main(argv)
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001  a fetch must not kill the run
        code = 1
        print(f"{type(exc).__name__}: {exc}", file=err)
    return code, out.getvalue(), err.getvalue()


def resolve(entry: VideoEntry, cache: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """-> (url, note). Cached in resolved.json so reruns hit the same video.

    A ytsearchN: query is resolved by listing the top N hits and taking the
    first one inside MIN_S..MAX_S seconds. Rank-1 is routinely a Short or a
    trailer, which is useless for testing a summary prompt.
    """
    key, _expected, _why, query = entry
    if key in cache:
        return cache[key]["url"], cache[key].get("note", "")

    if query.startswith(("http://", "https://")):
        cache[key] = {"url": query, "note": query}
        return query, query

    hits = search(query)
    picked: tuple[str, str, int] | None = None
    skipped: list[str] = []
    for hit in hits:
        vid = str(hit.get("id") or "")
        title = str(hit.get("title") or "")
        try:
            secs = int(float(hit.get("duration") or 0))
        except (TypeError, ValueError):
            secs = 0
        if vid and MIN_S <= secs <= MAX_S:
            picked = (vid, title, secs)
            break
        skipped.append(f"{secs // 60}:{secs % 60:02d} {title[:40]}")
    if picked is None:
        raise LookupError(
            f"no hit between {MIN_S // 60} and {MAX_S // 60} min; "
            f"rejected: {'; '.join(skipped[:4]) or 'nothing'}"
        )

    vid, title, secs = picked
    note = f"{title} [{secs // 60}:{secs % 60:02d}]"
    url = f"https://www.youtube.com/watch?v={vid}"
    cache[key] = {"url": url, "title": title, "note": note, "duration_s": secs}
    return url, note


def _reject_unknown_flags(argv: list[str]) -> None:
    """Fail on a forwarded flag that watchless's parser does not define.

    ``parse_known_args`` makes ``--bulits 20`` a silent no-op: the request for
    20 bullets disappears and the run uses the default. The forwarding channel
    stays untyped (that is what keeps the CLIs decoupled), but an unknown flag
    is a mistake worth stopping for.
    """
    from watchless.cli import build_parser  # noqa: PLC0415

    known = {opt for action in build_parser()._actions for opt in action.option_strings}
    unknown = [
        flag
        for flag in argv
        if flag.startswith("--") and flag.split("=", 1)[0] not in known
    ]
    if unknown:
        raise SystemExit(
            f"unknown option(s) for watchless: {', '.join(unknown)}. "
            f"These would be forwarded and ignored."
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="watchless-corpus",
        description="Run watchless over a genre-spanning corpus.",
        epilog="Unknown flags are passed through to watchless.py verbatim.",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="resolve the queries and print titles; run nothing",
    )
    ap.add_argument(
        "--force", action="store_true", help="rerun even when output already exists"
    )
    ap.add_argument("--only", metavar="KEYS", help="comma-separated corpus keys to run")
    ap.add_argument(
        "--timeout",
        type=int,
        default=1800,
        metavar="SEC",
        help="per-video timeout (default 1800)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=30.0,
        metavar="SEC",
        help="mean pause between videos (default 30). Most of "
        "the burst is inside a single watchless run, not "
        "between runs: see --sleep-subtitles",
    )
    ap.add_argument(
        "--sleep-jitter",
        type=float,
        default=15.0,
        metavar="SEC",
        help="randomise --sleep by +/- this much (default 15, so "
        "25-45s). A fixed gap is itself a fingerprint",
    )
    ap.add_argument(
        "--sleep-subtitles",
        type=float,
        default=5.0,
        metavar="SEC",
        help="gap inside one watchless run, before yt-dlp touches "
        "subtitles (default 5; passed straight through)",
    )
    ap.add_argument(
        "--stop-on-429",
        action="store_true",
        default=True,
        help="abort the whole run on the first rate-limit/block "
        "response instead of burning the rest of the quota "
        "(default: on; --no-stop-on-429 to disable)",
    )
    ap.add_argument(
        "--no-stop-on-429",
        dest="stop_on_429",
        action="store_false",
        help="keep going after a 429 (default is to stop)",
    )
    ap.add_argument(
        "--max-fetches",
        type=int,
        default=12,
        metavar="N",
        help="refuse to run more than N fetches in one invocation "
        "(default 12). Cached and --check entries do not "
        "count; 0 means no limit",
    )
    ap.add_argument(
        "--fetch-only",
        action="store_true",
        help="download transcripts only: no summary, no LLM calls. "
        "Inspect the text first, then use resummarize.py",
    )
    ours, passthrough = ap.parse_known_args()

    # Everything unknown is forwarded verbatim to watchless's own parser, which is
    # what keeps the two from having to be edited in lockstep. The cost is that
    # a typo in a *forwarded* flag is accepted here and only noticed (if at
    # all) deep inside a fetch, so validate the known ones now: silently
    # dropping an argument someone took the trouble to type is worse than
    # refusing it. --check makes no fetch, so it would otherwise never be seen.
    _reject_unknown_flags(passthrough)

    OUT.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)

    cache_path = OUT / "resolved.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    selected = _select(ours.only)

    run = _Run(ours=ours, passthrough=passthrough, cache=cache, cache_path=cache_path)
    for idx, entry in enumerate(VIDEOS, 1):
        if entry in selected:
            run.one(idx, entry)
        if run.blocked:
            _print_blocked(run.fetches)
            break

    _write_report(run.rows, failed=run.failed, fetches=run.fetches, blocked=run.blocked)


def _select(only: str | None) -> list[VideoEntry]:
    """The entries --only asked for, or all of them.

    Compared by identity against VIDEOS rather than by key, so a key that
    appears twice cannot be selected by name and skipped by position.
    """
    if not only:
        return list(VIDEOS)
    wanted = {k.strip() for k in only.split(",")}
    selected = [e for e in VIDEOS if e[0] in wanted]
    if not selected:
        keys = ", ".join(e[0] for e in VIDEOS)
        raise SystemExit(f"--only matched nothing (keys: {keys})")
    return selected


@dataclass
class _Run:
    """Mutable state for one pass over the corpus.

    It was seven local variables in a 102-statement main(); grouping them means
    each phase below takes one argument instead of seven, and the report can be
    written from the same object the loop filled.
    """

    ours: argparse.Namespace
    passthrough: list[str]
    cache: dict[str, dict[str, Any]]
    cache_path: Path
    rows: list[Row] = field(default_factory=list)
    failed: int = 0
    fetches: int = 0
    blocked: bool = False
    _ran_any: bool = False

    def one(self, idx: int, entry: VideoEntry) -> None:
        """Resolve, then either record what we have or fetch it."""
        key, expected, _why, _q = entry
        try:
            cached_hit = entry[0] in self.cache  # resolved before this run?
            url, note = resolve(entry, self.cache)
            self.cache_path.write_text(json.dumps(self.cache, indent=2))
        except LookupError as e:
            print(f"[{idx:02d}] {key:22} RESOLVE FAILED: {e}", flush=True)
            self.rows.append((idx, key, expected, "-", str(e), 0, "resolve"))
            self.failed += 1
            return

        md_path = OUT / f"{idx:02d}-{key}.md"
        log_path = OUT / f"{idx:02d}-{key}.log"
        print(f"[{idx:02d}] {key:22} {note}", flush=True)
        _adopt_saved(url, md_path)

        if self.ours.check:
            self.rows.append((idx, key, expected, _kind_from(log_path), note, 0, "ok"))
            # --check resolves queries and nothing else. Only a NEW resolution
            # touched YouTube, so only that one needs spacing out; sleeping
            # after cached ones would be 30s per entry for no reason.
            if not cached_hit:
                time.sleep(self.ours.sleep)
            return

        if md_path.exists() and md_path.stat().st_size and not self.ours.force:
            got = _kind_from(log_path)
            self.rows.append((idx, key, expected, got, note, 0, "cached"))
            return

        if self.ours.max_fetches and self.fetches >= self.ours.max_fetches:
            print(
                f"[{idx:02d}] {key:22} SKIPPED: --max-fetches "
                f"{self.ours.max_fetches} reached",
                flush=True,
            )
            self.rows.append((idx, key, expected, "-", "not fetched", 0, "skipped"))
            return

        self._fetch(idx, key, expected, url, note, md_path, log_path)

    def _fetch(
        self,
        idx: int,
        key: str,
        expected: str,
        url: str,
        note: str,
        md_path: Path,
        log_path: Path,
    ) -> None:
        if self._ran_any:
            # Randomised gap: a fixed interval is itself a fingerprint.
            jitter = random.uniform(-self.ours.sleep_jitter, self.ours.sleep_jitter)
            gap = max(1.0, self.ours.sleep + jitter)
            print(f"        waiting {gap:.0f}s", flush=True)
            time.sleep(gap)
        self._ran_any = True
        self.fetches += 1

        t0 = time.monotonic()
        argv = [url, "--out-dir", str(RAW), "--description"]
        if self.ours.fetch_only:
            argv.append("--no-summary")  # transcript only, zero LLM calls
        if self.ours.sleep_subtitles:
            # A gap INSIDE the run: metadata, then a pause, then subtitles.
            # --sleep only spaces out separate videos, which is not where the
            # burst is.
            argv += ["--sleep-subtitles", str(self.ours.sleep_subtitles)]
        argv += self.passthrough

        rc, out, err = run_watchless(argv)
        secs = int(time.monotonic() - t0)
        log_path.write_text(err + f"\n[elapsed {secs}s]\n[argv {' '.join(argv)}]\n")

        ok = rc == 0 and _store_output(out, md_path)
        if not ok:
            self.failed += 1
            tail = (err.strip().splitlines() or ["(no stderr)"])[-1]
            print(f"        FAILED: {tail}", flush=True)
            if self.ours.stop_on_429 and settings.looks_blocked(err):
                self.blocked = True

        self.rows.append(
            (idx, key, expected, parse_case(err), note, secs, "ok" if ok else "fail")
        )


def _adopt_saved(url: str, md_path: Path) -> None:
    """Recover an already-saved transcript that is under a different name.

    A .md from an earlier version of this script, or a hand-run, may not live
    under the expected name. Losing it silently would look like a fetch failure
    and cost another YouTube request.
    """
    if md_path.exists():
        return
    for cand in md_candidates(url):
        md_path.write_text(cand.read_text())
        print(f"        adopted {cand.name} -> {md_path.name}", flush=True)
        return


def _store_output(out: str, md_path: Path) -> bool:
    """Move the file watchless printed into the corpus, -> whether it worked."""
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return False
    src = Path(lines[-1])
    if not src.is_absolute():
        src = OUT.parent / src
    if not src.exists():
        return False
    md_path.write_text(src.read_text())
    src.unlink()
    return True


def _print_blocked(fetches: int) -> None:
    print(
        "\n! YouTube is rate-limiting or blocking this IP. The limit "
        "is IP- and volume-based (hours to days), so the remaining "
        f"entries would fail too. Stopped after {fetches} fetch(es).\n"
        "  Wait, switch network, or set WATCHLESS_PROXY. Re-running later "
        "reuses cached files, so nothing is lost.",
        flush=True,
    )


def _write_report(rows: list[Row], *, failed: int, fetches: int, blocked: bool) -> None:
    lines = [
        "# watchless corpus run",
        "",
        f"{len(rows)} entries, {failed} failed or unresolved",
        f"{fetches} fetch(es) this run"
        + (" (stopped early: rate-limited)" if blocked else ""),
        "",
        "expected = this script's guess; detected = what the summary's",
        "CASE line said. A mismatch is either a prompt bug or a wrong",
        "expectation -- read the .md before deciding which.",
        "",
        "| # | key | expected | detected | status | s | video |",
        "|---|-----|----------|----------|--------|---|-------|",
    ]
    for idx, key, expected, got, note, secs, status in rows:
        flag = " **MISMATCH**" if _mismatch(expected, got) else ""
        lines.append(
            f"| {idx:02d} | {key} | {expected} | {got}{flag} | "
            f"{status} | {secs} | {note} |"
        )
    mism = [r for r in rows if _mismatch(r[2], r[3])]
    lines += ["", f"case mismatches: {len(mism)}"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
