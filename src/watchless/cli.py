"""The ``watchless`` command: YouTube URL in, readable transcript out.

    watchless URL [-o FILE] [--lang de,en]
               [--out-dir DIR] [--slug] [--max-slug N] [--no-header]

One backend (see :mod:`watchless.fetch`). It takes the
video's ORIGINAL language unless you pass ``--lang``.

Filenames::

    default              DIR/<video-id>.txt
    --slug               DIR/<slug>--<id>.txt
    --slug --max-slug 0  DIR/<id>.txt          (slug disabled)

It never shells out to a binary: the library is imported, so a missing
dependency is an ImportError with a real message rather than a PATH mystery.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from watchless import __version__, settings, termux
from watchless import fetch as fetch_mod
from watchless.fetch import BlockedError, FetchError, Segment, Video, YtdlpBackend
from watchless.format import Document, dest_path, render, video_id
from watchless.log import log
from watchless.metadata import oembed_metadata

# Re-exported so callers (and tests) can substitute a backend through the CLI
# module without reaching into watchless.fetch. Explicit, not incidental.
__all__ = [
    "BlockedError",
    "Document",
    "FetchError",
    "Segment",
    "Video",
    "YtdlpBackend",
    "build_parser",
    "main",
    "resolve_video",
]

# Exit codes are part of the interface: termux-url-opener and corpus.py branch
# on them.
EXIT_NO_TRANSCRIPT = 2
EXIT_BLOCKED = 3
EXIT_NO_ENDPOINT = 4


def blocked_advice() -> str:
    """What to actually do about an IP block.

    A block is temporal and IP-based, not a bug, so the message names the ways
    out in the order that costs least: wait, switch network, proxy.
    """
    return (
        "YouTube is blocking this IP. Usually temporary (hours to days), and "
        "far\nmore likely on a VPN, a datacenter IP, or after repeated "
        "requests. Options:\n"
        "  - wait, or switch network (mobile data vs Wi-Fi)\n"
        "  - use a proxy:  WATCHLESS_PROXY=http://host:port\n"
        "  - transcribe the audio offline instead (RECCoon or Scrib + Parakeet)"
    )


@dataclass(slots=True)
class FetchOutcome:
    """What a backend attempt produced, or why it did not."""

    segments: list[Segment]
    track_label: str | None
    errors: list[str]
    # An IP block rather than a video with no captions. A flag, not a substring
    # test on the message: error text containing the word "blocked" would
    # otherwise send the user off to change their IP over a video that simply
    # has no subtitles.
    blocked: bool = False


# ------------------------------------------------------------------ backends


def resolve_video(vid: str, *, want_description: bool) -> tuple[Video, bool]:
    """Metadata for ``vid``, cheapest source first.

    Returns ``(video, used_ytdlp)``. oEmbed is stdlib and ~200ms and gives
    title + channel; yt-dlp is a ~1-2s import and is the only source of the
    description and the caption track list, so it is only paid for when one of
    those is actually needed.
    """
    video = Video(vid=vid)
    title, channel = oembed_metadata(vid)
    if title:
        video.title, video.channel = title, channel

    if not want_description and title:
        return video, False

    detailed = YtdlpBackend().info(vid)
    video.title = video.title or detailed.title
    video.channel = video.channel or detailed.channel
    video.lang = detailed.lang
    video.subtitles = detailed.subtitles
    video.automatic_captions = detailed.automatic_captions
    if want_description:
        video.description = detailed.description
    return video, True


def fetch_segments(video: Video, *, langs: list[str] | None) -> FetchOutcome:
    """Captions, or the reason there are none.

    A blocked error is kept verbatim in ``errors`` so the caller can tell
    "YouTube refused this IP" apart from "this video has no captions" -- the
    two need different exit codes and different advice.
    """
    impl = YtdlpBackend()
    try:
        track, segments = impl.fetch(video.vid, video, langs)
    except BlockedError as exc:
        return FetchOutcome([], None, [f"blocked ({exc})"], blocked=True)
    except FetchError as exc:
        return FetchOutcome([], None, [str(exc)])
    return FetchOutcome(segments, f"{impl.name}, {track.label}", [])


# ------------------------------------------------------------------ doctor


def _have(module: str) -> bool:
    import importlib.util  # noqa: PLC0415

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# Doctor probe timings and the point at which a model list stops being useful
# to print in full. Named because they are policy, not arithmetic.
_PROBE_TIMEOUT_S = 10
_CHAT_TIMEOUT_S = 15
_MODEL_LIST_MAX = 10


def _llm_probes() -> tuple[
    Callable[..., tuple[list[str], dict[str, Any]]],
    Callable[..., str],
    Callable[[str], str],
    Callable[..., tuple[bool, str]],
]:
    """The llm helpers the doctor needs, imported once and lazily.

    watchless.cli must stay importable without watchless.llm being exercised: a plain
    transcript fetch never touches it. Importing inside one helper keeps that
    property instead of repeating the import in five places.
    """
    from watchless.llm import (  # noqa: PLC0415
        chat_probe,
        connection_hint,
        describe_models,
        probe_models,
    )

    return probe_models, describe_models, connection_hint, chat_probe


ENV_VARS = (
    "WATCHLESS_BASE_URL",
    "WATCHLESS_API_KEY",
    "WATCHLESS_MODEL",
    "WATCHLESS_SLUG",
    "WATCHLESS_VERBOSE",
    "WATCHLESS_PROXY",
    "WATCHLESS_REASONING_EFFORT",
)


def run_doctor(args: argparse.Namespace) -> None:
    """Print exactly what this install resolves and whether it works.

    Five stages, each of which can be the last word: what is installed, what
    the config resolves to, whether the host answers at all, whether it serves
    our model, and whether a chat request works. Each prints and returns a
    verdict because the useful failure is the FIRST one, not a chat error three
    screens later that hides a wrong port.
    """
    print(f"watchless {__version__}  ({Path(__file__).resolve()})")
    print(f"python {sys.version.split()[0]}")
    _doctor_environment()

    endpoint = settings.base_url(args.llm_url)
    llm_key = settings.api_key(args.llm_key)
    llm_model = settings.model(args.llm_model)
    _doctor_resolved(args, endpoint, llm_key, llm_model)

    if not endpoint:
        print(
            "summary: DISABLED - no endpoint. "
            "Set --llm-url or export WATCHLESS_BASE_URL."
        )
        return

    if not _doctor_reachable(endpoint):
        return

    probe_models, describe_models, connection_hint, chat_probe = _llm_probes()
    ids, info = probe_models(endpoint, llm_key, timeout=_PROBE_TIMEOUT_S)
    bad_model = _doctor_models(info, ids, llm_model, describe_models, connection_hint)

    print(f"  chat: POST {endpoint} (tiny request, up to 15s) ...")
    ok, detail = chat_probe(endpoint, llm_key, llm_model, timeout=_CHAT_TIMEOUT_S)
    print(f"  chat: {detail}")
    _doctor_verdict(ok, bad_model, detail, connection_hint)


def _doctor_environment() -> None:
    print()
    print("environment (only EXPORTED vars are visible here):")
    for key in ENV_VARS:
        value = os.environ.get(key)
        if key.endswith(("KEY", "PASS")) and value:
            value = "***set***"
        print(f"  {key:24} {value or '-'}")
    print()


def _doctor_resolved(
    args: argparse.Namespace,
    endpoint: str | None,
    llm_key: str | None,
    llm_model: str | None,
) -> None:
    import shutil  # noqa: PLC0415

    print("resolved:")
    print(f"  endpoint   {endpoint or '-'}")
    print(f"  model      {llm_model or '- (field omitted)'}")
    print(f"  key        {'***set***' if llm_key else '-'}")
    print(
        f"  timeout    {args.llm_timeout}s   bullets {args.bullets}   "
        f"max_chars {args.max_chars}"
    )
    print()
    print("features:")
    features = (
        ("yt-dlp (library)", _have("yt_dlp")),
        ("yt-dlp (binary)", bool(shutil.which("yt-dlp"))),
    )
    for name, ok in features:
        print(f"  {name:24} {'yes' if ok else 'NO'}")
    print()


def _doctor_reachable(endpoint: str) -> bool:
    """Can we open a socket to the endpoint at all? -> whether to continue."""
    import socket  # noqa: PLC0415
    from urllib.parse import urlparse  # noqa: PLC0415

    parsed = urlparse(endpoint)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    print(f"summary: enabled. testing tcp {host}:{port} ...")
    try:
        with socket.create_connection((host, port), timeout=5):
            print("  tcp: connected")
    except OSError as exc:
        print(f"  tcp: FAILED - {exc}")
        print(f"  -> nothing is listening on {host}:{port} from this device.")
        print("     Common causes: server bound to 127.0.0.1 instead of 0.0.0.0;")
        print("     wrong port; Wi-Fi AP isolation; different subnet or VPN.")
        return False
    print()
    return True


def _doctor_models(
    info: dict[str, Any],
    ids: list[str],
    llm_model: str | None,
    describe_models: Callable[..., str],
    connection_hint: Callable[[str], str],
) -> bool:
    """Report /models and check our model is in it -> whether it is a bad model."""
    diagnosis = describe_models(info, ids)
    print(f"  /models: {diagnosis}")
    if not ids:
        hint = connection_hint(diagnosis)
        if hint:
            print(f"  -> {hint}")
        print()
        return False

    listed = ", ".join(ids[:_MODEL_LIST_MAX])
    suffix = " ..." if len(ids) > _MODEL_LIST_MAX else ""
    print(f"  ids: {listed}{suffix}")
    if not llm_model:
        print()
        return False
    if llm_model not in ids:
        print(
            f"  !! model {llm_model!r} is NOT served -> the summary "
            f"would fail. Use one of the ids above, or unset "
            f"WATCHLESS_MODEL."
        )
        print()
        return True
    print(f"  model {llm_model!r} is served: ok")
    print()
    return False


def _doctor_verdict(
    ok: bool, bad_model: bool, detail: str, connection_hint: Callable[[str], str]
) -> None:
    print()
    if ok and bad_model:
        print(
            "verdict: LIKELY BROKEN - the endpoint answers, but the configured "
            "model is not in its /models list. Fix WATCHLESS_MODEL first."
        )
    elif ok:
        print("verdict: READY - summaries will work.")
    else:
        hint = connection_hint(detail)
        if hint:
            print(f"  -> {hint}")
        print()
        print(
            "verdict: NOT READY - a real run would still save the transcript, "
            "without a summary."
        )


# ------------------------------------------------------------------ argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watchless",
        description="YouTube URL -> readable transcript.",
    )
    parser.add_argument("url", nargs="?", help="YouTube URL/video id")
    parser.add_argument("-o", "--out", help="write to FILE instead of stdout")
    parser.add_argument("--out-dir", help="write to DIR/<name>.txt (created if needed)")
    parser.add_argument(
        "--lang",
        help="comma-separated, e.g. de,en. Default: the video's original language",
    )

    # Three-state slug control: unset -> WATCHLESS_SLUG -> --slug / --no-slug.
    parser.set_defaults(slug=settings.env_flag("WATCHLESS_SLUG"))
    parser.add_argument(
        "--slug",
        dest="slug",
        action="store_true",
        help="use a slugified title in the filename "
        f"('<slug>--<id>.txt', max {settings.DEFAULT_MAX_SLUG} "
        "chars). Default comes from WATCHLESS_SLUG, else off",
    )
    parser.add_argument(
        "--no-slug",
        dest="slug",
        action="store_false",
        help="force '<id>.txt' even if WATCHLESS_SLUG is set",
    )
    parser.add_argument(
        "--max-slug",
        type=int,
        default=settings.DEFAULT_MAX_SLUG,
        metavar="N",
        help=f"slug length limit, 0 disables it (default {settings.DEFAULT_MAX_SLUG})",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="omit the title/URL header block from the transcript",
    )
    parser.add_argument(
        "--sleep-subtitles",
        type=float,
        default=None,
        metavar="SEC",
        help="pause between the metadata call and the caption "
        "request (default 1). Raise it if you are seeing "
        "HTTP 429. Also WATCHLESS_SLEEP_SUBTITLES",
    )

    # Both are ON unless turned off. WATCHLESS_DESCRIPTION / WATCHLESS_TIMESTAMPS are
    # still honoured, but now as an explicit opt-OUT ("0", "false", ...) rather
    # than the only way to get the feature.
    parser.set_defaults(
        description=not settings.env_flag_set_false("WATCHLESS_DESCRIPTION"),
        timestamps=not settings.env_flag_set_false("WATCHLESS_TIMESTAMPS"),
    )
    parser.add_argument(
        "--description",
        dest="description",
        action="store_true",
        help="include the video description under the metadata "
        "(default: on, unless WATCHLESS_DESCRIPTION is falsey)",
    )
    parser.add_argument(
        "--no-description",
        dest="description",
        action="store_false",
        help="omit the description entirely, and do not use it for the summary either",
    )
    parser.add_argument(
        "--timestamps",
        dest="timestamps",
        action="store_true",
        help="prefix each paragraph with its start time "
        "(default: on, unless WATCHLESS_TIMESTAMPS is falsey)",
    )
    parser.add_argument(
        "--no-timestamps",
        dest="timestamps",
        action="store_false",
        help="omit timestamps from the written file",
    )

    group = parser.add_argument_group(
        "summary (any OpenAI-compatible endpoint: llama-server, Ollama, ...)"
    )
    group.add_argument(
        "--llm-url",
        metavar="URL",
        help="base URL, e.g. http://127.0.0.1:8080/v1 (or set WATCHLESS_BASE_URL)",
    )
    group.add_argument(
        "--llm-key",
        metavar="TOKEN",
        help=(
            "bearer token (or set WATCHLESS_API_KEY); "
            "omit for llama-server without auth"
        ),
    )
    group.add_argument(
        "--llm-model",
        metavar="NAME",
        help="model name (or set WATCHLESS_MODEL). Omitted by "
        "default, which is correct for single-model servers. "
        "Required for router mode and llama-swap, which "
        "route on this field",
    )
    group.add_argument(
        "--bullets",
        type=int,
        default=settings.DEFAULT_MAX_BULLETS,
        metavar="N",
        help="max bullet points (default 12)",
    )
    group.add_argument(
        "--llm-timeout",
        type=int,
        default=settings.TIMEOUT_S,
        metavar="SEC",
        help="how long to wait for the endpoint (default 300)",
    )
    group.add_argument(
        "--max-chars",
        type=int,
        default=settings.DEFAULT_MAX_CHARS,
        metavar="N",
        help="trim transcripts longer than N chars before "
        "sending (head+tail kept; default 24000)",
    )
    group.add_argument(
        "--summary",
        action="store_true",
        help="require a summary: warn loudly and exit non-zero "
        "if no endpoint is configured",
    )
    group.add_argument(
        "--no-summary",
        action="store_true",
        help="skip the summary even if --llm-url/env is set",
    )
    parser.add_argument(
        "--version", action="version", version=f"watchless {__version__}"
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="print resolved config + endpoint reachability, then exit",
    )
    parser.add_argument(
        "--install-termux-opener",
        action="store_true",
        help="install the Android share-sheet target at ~/bin/termux-url-opener, "
        "checking for Termux and the Termux:API bits first",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --install-termux-opener: overwrite an existing opener "
        "without asking",
    )
    return parser


# ------------------------------------------------------------------ main


def _parse_langs(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _write(text: str, dest: Path | None) -> None:
    """Write the document, translating the two common failures into advice."""
    if dest is None:
        sys.stdout.write(text)
        return
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    except PermissionError as exc:
        raise SystemExit(
            f"cannot write to {exc.filename or dest}: permission denied.\n"
            f"On Android /tmp and /sdcard are not usable as a working dir.\n"
            f"Use your home or Termux's private tmp instead:\n"
            f"  --out-dir ~/transcripts\n"
            f'  --out-dir "$TMPDIR/t"'
        ) from exc
    except OSError as exc:
        raise SystemExit(
            f"cannot write to {exc.filename or dest}: {exc.strerror}"
        ) from exc
    log(f"wrote {dest.name} ({len(text.split())} words)")
    print(dest)  # stdout stays machine-parseable: the path, nothing else


def _llm_settings(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, str | None]:
    """Resolve and validate the LLM config, exiting on contradictions.

    A summary that cannot happen must never be silently skipped, and an
    endpoint that will not be used must never be silently loaded -- both are
    reported here rather than in the middle of the run.
    """
    from watchless.summarize import resolve  # noqa: PLC0415

    llm_url, llm_key, llm_model = resolve(args.llm_url, args.llm_key, args.llm_model)

    if (llm_key or llm_model) and not llm_url:
        log(
            "note: --llm-key/--llm-model given without an endpoint; "
            "set --llm-url or WATCHLESS_BASE_URL. Skipping summary."
        )
    if args.summary and not llm_url:
        log(
            "error: --summary requested but no endpoint is configured.\n"
            "  Set --llm-url, or WATCHLESS_BASE_URL. If it is in an env file,\n"
            "  the variables must be EXPORTED (add `export` in the file)."
        )
        raise SystemExit(EXIT_NO_ENDPOINT)
    if args.no_summary and llm_url:
        log("note: --no-summary given; endpoint is set but unused.")
    return llm_url, llm_key, llm_model


def _resolve_or_die(vid: str, *, want_description: bool) -> tuple[Video, bool]:
    """Metadata, or an exit code that says which kind of failure it was."""
    try:
        return resolve_video(vid, want_description=want_description)
    except FetchError as exc:
        if isinstance(exc, BlockedError):
            log(blocked_advice())
            raise SystemExit(EXIT_BLOCKED) from exc
        log(f"metadata failed: {exc}")
        raise SystemExit(EXIT_NO_TRANSCRIPT) from exc


def _segments_or_die(video: Video, *, langs: list[str] | None) -> FetchOutcome:
    """Captions, or an exit code that distinguishes a block from a miss."""
    outcome = fetch_segments(video, langs=langs)
    if outcome.segments:
        return outcome

    for error in outcome.errors:
        log(f"  ! {error}")
    if outcome.blocked:
        log(blocked_advice())
        raise SystemExit(EXIT_BLOCKED)
    log("no subtitles available -> needs ASR fallback")
    raise SystemExit(EXIT_NO_TRANSCRIPT)


def _summarize_or_degrade(
    text: str,
    *,
    video: Video,
    llm_url: str,
    llm_key: str | None,
    llm_model: str | None,
    args: argparse.Namespace,
    kind_out: list[str] | None = None,
) -> str | None:
    """Summarise, keeping the transcript if anything goes wrong.

    The transcript is the thing the user came for. A failed summary degrades
    the output; it must never discard the fetch that already succeeded.
    """
    from watchless.summarize import SummarizeError  # noqa: PLC0415
    from watchless.summarize import summarize as summarise  # noqa: PLC0415

    log(
        f"summarising via {llm_url}"
        + (f" (model {llm_model})" if llm_model else "")
        + f", timeout {args.llm_timeout}s ..."
    )
    try:
        summary, was_trimmed = summarise(
            text,
            base_url=llm_url,
            api_key=llm_key,
            model=llm_model,
            max_bullets=args.bullets,
            max_chars=args.max_chars,
            timeout=args.llm_timeout,
            title=video.title,
            channel=video.channel,
            description=video.description,
            kind_out=kind_out,
        )
    except KeyboardInterrupt:
        log("summary interrupted; keeping the transcript without it")
        return None
    except SummarizeError as exc:
        log(f"summary failed: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001  never lose the transcript
        log(f"summary failed: {exc}")
        return None

    log(
        f"summary: {len(summary.splitlines())} lines"
        + (" (transcript was trimmed)" if was_trimmed else "")
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.doctor:
        run_doctor(args)
        return
    if args.install_termux_opener:
        # Exit status rather than SystemExit: this reports "not Termux" as a
        # normal outcome, not an exception.
        raise SystemExit(termux.install_opener(force=args.force))
    if not args.url:
        parser.error(
            "a URL is required (unless you are using --doctor "
            "or --install-termux-opener)"
        )

    try:
        vid = video_id(args.url)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    llm_url, llm_key, llm_model = _llm_settings(args)
    want_summary = (not args.no_summary) and bool(llm_url)
    # Fetched unless --no-description. It is load-bearing when summarising:
    # content there, not decoration, and --no-description means "not even for
    # the summary", so this is a plain bool rather than a tri-state.
    want_desc = bool(args.description)

    video, used_ytdlp = _resolve_or_die(vid, want_description=want_desc)
    log(
        f"[meta] {'yt-dlp' if used_ytdlp else 'oEmbed'}: "
        f"title={'ok' if video.title else '-'} "
        f"lang={video.lang or '-'} "
        f"desc={'ok' if video.description else '-'}"
    )

    outcome = _segments_or_die(video, langs=_parse_langs(args.lang))

    summary = None
    kind: list[str] = []
    if want_summary:
        assert llm_url is not None  # implied by want_summary
        summary = _summarize_or_degrade(
            _transcript_text(outcome.segments),
            video=video,
            llm_url=llm_url,
            llm_key=llm_key,
            llm_model=llm_model,
            args=args,
            kind_out=kind,
        )
        # Scoring metadata, not content: the file is for a reader, stderr is for
        # diagnostics.
        if kind:
            log(f"[kind] {kind[0]}")

    document = Document(
        vid=vid,
        url=fetch_mod.url_for(vid),
        segments=outcome.segments,
        title=video.title,
        channel=video.channel,
        backend=outcome.track_label,
        # If the description was fetched, write it. It is content, not
        # decoration, so there is no reason to pay for it and then discard it.
        description=video.description,
        summary=summary,
        with_header=not args.no_header,
        timestamps=args.timestamps,
    )
    _write(
        render(document),
        dest_path(args.out, args.out_dir, args.slug, video.title, vid, args.max_slug),
    )


def _transcript_text(segments: Sequence[Segment]) -> str:
    """The summary input: the body only, never the rendered document.

    Title, channel and description travel as their own labelled fields, so
    metadata can never be mistaken for speech.

    Never carries timeline markers: the written file has them because a reader
    wants to locate a passage, the model does not, and a marker every few lines
    is noise it has to ignore.
    """
    from watchless.format import reflow  # noqa: PLC0415

    return reflow(segments, timestamps=False)


if __name__ == "__main__":
    main()
