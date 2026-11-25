"""Transcript and metadata acquisition.

One backend, ``yt_dlp``, imported as a library. It is the source of both the
description and the full track list -- including uploader dubs, which is what
makes original-language selection possible.

Everything resolves to a :class:`Track` plus segments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from watchless import settings
from watchless.log import debug

# yt-dlp ships no stubs, so its option dict has no upstream type to borrow.
# A named alias keeps call sites readable; mypy cannot check the keys.
Options = dict[str, Any]

# json3 carries word timings, is already non-overlapping, and needs no
# rolling-duplicate removal (a VTT/TTML artefact).
JSON3 = "json3"


class FetchError(RuntimeError):
    """No usable transcript. The message is shown to the user verbatim."""


class BlockedError(FetchError):
    """YouTube is rate-limiting or blocking this IP. Not retryable in-run."""


@dataclass(frozen=True, slots=True)
class Segment:
    """One caption line, in seconds."""

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class Track:
    """A caption track we can actually download."""

    lang: str
    generated: bool
    url: str

    @property
    def label(self) -> str:
        """Human-readable provenance, e.g. ``de (auto)``."""
        return f"{self.lang} ({'auto' if self.generated else 'uploader'})"


@dataclass(slots=True)
class Video:
    """Everything one metadata fetch knows about a video.

    ``subtitles`` and ``automatic_captions`` map a language code to the list of
    formats YouTube advertises. Populated by the ytdlp backend; the api backend
    only fills the scalar fields it already had.
    """

    vid: str
    title: str | None = None
    channel: str | None = None
    lang: str | None = None
    description: str | None = None
    subtitles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    automatic_captions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


# ---------------------------------------------------------------- languages


def base_lang(code: str) -> str:
    """``en-US`` -> ``en``; tolerant of ``en_US`` and empty input."""
    return (code or "").lower().replace("_", "-").split("-")[0]


def pick_by_lang(tracks: list[Track], src_lang: str) -> Track | None:
    """Best track for the video's original language, or None.

    Provenance outranks exactness. A human-written caption beats a machine
    transcription of the same language, and it does so even when the exact code
    differs: given "en" and tracks en-GB (uploader) and en (auto), the uploader
    track is the better source, because an ASR track has no punctuation, no
    speaker labels and mishears proper nouns. Testing exactness first made the
    auto track win, which is backwards.

    Only once provenance is tied does the region matter: an exact code
    (``en-GB`` for ``en-GB``) beats a base-code match (``en`` for ``en-GB``).
    """
    want = src_lang.lower().replace("_", "-")
    for generated in (False, True):  # uploader tracks first, always
        pool = [t for t in tracks if t.generated is generated]
        for match in (
            lambda t: t.lang.lower() == want,
            lambda t: base_lang(t.lang) == base_lang(want),
        ):
            hits = [t for t in pool if match(t)]
            if hits:
                return hits[0]
    return None


def _distinct_langs(tracks: list[Track]) -> list[str]:
    return sorted({track.lang for track in tracks})


def source_lang_for(vid: str, video: Video, tracks: list[Track]) -> str | None:
    """The video's own language, looked up only when it can change the answer.

    oEmbed carries no language, so a default run gets here with ``video.lang``
    unset and the track list is the only local information. Asking yt-dlp costs
    an import plus a request, so it is only done when there is an ambiguity to
    resolve: a video offering a single language has nothing to choose between.

    Returning ``None`` means "no statement available", which
    :func:`_pick_track` handles by accepting an unambiguous single language or
    refusing -- never by taking whatever came first.
    """
    if video.lang:
        return video.lang
    if len(_distinct_langs(tracks)) <= 1:
        return None
    return YtdlpBackend().info(vid).lang


def _pick_track(
    tracks: list[Track], langs: list[str] | None, src_lang: str | None
) -> Track:
    """Choose one track from everything the video offers.

    Precedence, and the reason for it:

    1. ``--lang`` wins outright, in the order given. Asking for a dub is a
       deliberate act.
    2. Otherwise the video's ORIGINAL language. Any other language is a dub or
       a translation -- someone else's words, not the speaker's. Taking "the
       first non-generated track" is what silently returned an Arabic dub for
       an English video, and an Arabic auto-dub for a Georgian vlog.
    3. A video offering exactly one language is not ambiguous, so that track is
       taken without needing to know the language.
    4. Otherwise fall back to ``WATCHLESS_LANG_FALLBACK`` (default ``en``), because
       refusing is worse than a documented, announced guess. Set the variable to
       ``none`` to make this case an error again.

    A fallback pick is announced on stderr rather than returned, so the caller
    does not have to thread a flag through: see :func:`source_lang_for`.
    """
    if not tracks:
        raise FetchError("no caption tracks at all")
    if langs:
        want = [lang.lower() for lang in langs]
        for code in want:
            hit = next(
                (t for t in tracks if t.lang.lower() == code and not t.generated),
                None,
            ) or next((t for t in tracks if t.lang.lower() == code), None)
            if hit is not None:
                return hit
        have = sorted({t.lang for t in tracks})
        raise FetchError(f"none of {want} present; have {have}")
    if src_lang:
        hit = pick_by_lang(tracks, src_lang)
        if hit is not None:
            return hit
        raise FetchError(
            f"no track in the original language ({src_lang}); have "
            f"{sorted({t.lang for t in tracks})} -- pass --lang X to take a "
            f"dub deliberately"
        )

    available = _distinct_langs(tracks)
    if len(available) == 1:
        return next((t for t in tracks if not t.generated), tracks[0])

    preferred = settings.lang_fallback()
    if preferred:
        hit = pick_by_lang(tracks, preferred)
        if hit is not None:
            debug(
                f"original language unknown; fell back to {preferred!r} "
                f"(set WATCHLESS_LANG_FALLBACK, or pass --lang, to override)"
            )
            return hit
        raise FetchError(
            f"original language unknown and no {preferred!r} track either; "
            f"have {available} -- pass --lang X"
        )

    raise FetchError(
        f"cannot tell which of {available} is the original language, and "
        f"WATCHLESS_LANG_FALLBACK is disabled -- pass --lang X to choose"
    )


# ---------------------------------------------------------------- parsing


def parse_json3(payload: str) -> list[Segment]:
    """json3 events -> segments. Empty results raise, not return []."""
    try:
        events = json.loads(payload).get("events") or []
    except json.JSONDecodeError as exc:
        raise FetchError(f"caption payload is not JSON: {exc}") from exc

    segments: list[Segment] = []
    for event in events:
        text = "".join(str(s.get("utf8", "")) for s in (event.get("segs") or []))
        if not text.strip():
            continue
        start_ms = int(event.get("tStartMs", 0) or 0)
        segments.append(
            Segment(
                start=start_ms / 1000.0,
                end=(start_ms + int(event.get("dDurationMs", 0) or 0)) / 1000.0,
                text=text,
            )
        )
    if not segments:
        raise FetchError("track was empty")
    return segments


# ---------------------------------------------------------------- ytdlp


def _tracks_from(info: dict[str, Any]) -> list[Track]:
    """Flatten advertised caption formats into downloadable tracks."""
    tracks: list[Track] = []
    for source, generated in (
        (info.get("subtitles"), False),
        (info.get("automatic_captions"), True),
    ):
        for lang, formats in (source or {}).items():
            chosen = next(
                (f for f in formats if f.get("ext") == JSON3),
                None,
            )
            if chosen is None or not chosen.get("url"):
                continue
            tracks.append(Track(lang=lang, generated=generated, url=str(chosen["url"])))
    return tracks


class YtdlpBackend:
    """yt-dlp as a library: metadata, track list and captions in one client."""

    name = "ytdlp"

    def __init__(self, *, sleep_before_subtitles: float | None = None) -> None:
        self._sleep = (
            settings.subtitle_delay()
            if sleep_before_subtitles is None
            else sleep_before_subtitles
        )

    def _client(self, **overrides: Any) -> Any:
        """A YoutubeDL configured to talk to YouTube and download nothing.

        Returns ``Any`` because every call site is a one-liner
        (``extract_info``, ``urlopen``) and the option dict is what matters.
        """
        import yt_dlp  # noqa: PLC0415  (kept lazy: a 6MB import per run)

        # Typed as yt-dlp's own option TypedDict so a wrong value type is a type
        # error here rather than a runtime surprise inside the library. That
        # name exists only in the stubs, so it must never be imported at
        # runtime -- the alias below is what makes the annotation usable.
        opts: Options = {
            "quiet": True,
            "no_warnings": True,
            # yt-dlp registers ``skip_download`` as ``action='store_true'`` and
            # only ever tests it for truthiness (``params.get('skip_download')``).
            "skip_download": True,
            "noprogress": True,
            # One attempt: retry policy belongs to the caller, not the library.
            "retries": 0,
            "extractor_retries": 0,
        }
        proxy = settings.proxy_url()
        if proxy:
            opts["proxy"] = proxy
        opts.update(overrides)
        return yt_dlp.YoutubeDL(opts)

    def info(self, vid: str) -> Video:
        """Metadata plus every advertised caption track. One request."""
        client = self._client()
        try:
            raw = client.extract_info(url_for(vid), download=False)
        except BaseException as exc:
            raise _classify(exc) from exc
        if not isinstance(raw, dict):
            raise FetchError("yt-dlp returned no video info")

        video = Video(
            vid=vid,
            title=raw.get("title") or None,
            channel=raw.get("uploader") or raw.get("channel") or None,
            description=(raw.get("description") or None),
            subtitles=raw.get("subtitles") or {},
            automatic_captions=raw.get("automatic_captions") or {},
        )
        lang = raw.get("language")
        video.lang = None if lang in (None, "", "NA") else str(lang)
        return video

    def fetch(
        self,
        _vid: str,
        video: Video,
        langs: list[str] | None = None,
    ) -> tuple[Track, list[Segment]]:
        """Pick a track and download it, in memory.

        ``_vid`` is unused: everything needed came from the ``info()`` call that
        produced ``video`` -- unless that call never happened, which is the
        common case. A plain run uses oEmbed (fast, stdlib) and only pays for
        yt-dlp when the api backend fails, so by the time we get here the track
        lists are usually absent. Fetch them rather than reporting "no caption
        tracks at all" for a video that plainly has them.
        """
        if not video.subtitles and not video.automatic_captions:
            detailed = self.info(video.vid)
            video.subtitles = detailed.subtitles
            video.automatic_captions = detailed.automatic_captions
            video.lang = video.lang or detailed.lang

        tracks = _tracks_from(
            {
                "subtitles": video.subtitles,
                "automatic_captions": video.automatic_captions,
            }
        )
        track = _pick_track(tracks, langs, video.lang)
        client = self._client()
        try:
            # The gap YouTube throttles is between metadata and the caption
            # request itself, not between separate runs.
            if self._sleep > 0:
                import time  # noqa: PLC0415

                time.sleep(self._sleep)
            with client.urlopen(track.url) as response:
                payload = response.read().decode("utf-8", "replace")
        except BaseException as exc:
            raise _classify(exc) from exc
        return track, parse_json3(payload)


# ---------------------------------------------------------------- api


# ---------------------------------------------------------------- helpers


def url_for(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


def _classify(exc: BaseException) -> FetchError:
    """Map a library exception onto this program's two failure kinds.

    Both backends signal a block with different class names and message text,
    and neither is part of a stable API, so match on the text.
    """
    text = f"{type(exc).__name__}: {exc}"
    if settings.looks_blocked(text):
        return BlockedError(text)
    return FetchError(text)
