"""Offline tests: no network, no YouTube, no LLM endpoint required.

Fetching is exercised through stub backends, never the real services. These
tests are deliberately about *behaviour* -- exit codes, precedence, document
shape, which track wins -- rather than about whether a particular subprocess
happened to be spawned, which is why the backends are now stubbed as objects.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import ClassVar
from unittest import mock

from watchless import cli as yt
from watchless import fetch as F
from watchless import format as FMT
from watchless import settings
from watchless import summarize as S
from watchless import termux as T
from watchless.fetch import BlockedError, FetchError, Segment, Track, Video
from watchless.summarize import SummarizeError

ENV = ("WATCHLESS_BASE_URL", "WATCHLESS_API_KEY", "WATCHLESS_MODEL", "WATCHLESS_SLUG")
VID = "7qt3L44LoiE"
FIXTURE = Path(__file__).parent / "fixtures" / "segments.json"


def _stub_backend(info: Callable[[object, str], Video]) -> type:
    """A YtdlpBackend stand-in covering both methods the CLI calls.

    The CLI reaches the backend twice: resolve_video calls ``info()`` for
    metadata, fetch_segments calls ``fetch()`` for captions. Tests used to
    patch only ``info`` on the real class, which left ``fetch`` going to the
    network as soon as the api backend was removed and nothing else supplied
    captions.
    """

    class Stub:
        name = "ytdlp"

        def fetch(
            self, vid: str, video: Video, langs: list[str] | None = None
        ) -> tuple[Track, list[Segment]]:
            return Track("de", True, ""), fixture_segments()

    Stub.info = info  # type: ignore[attr-defined]
    return Stub


def fixture_segments() -> list[Segment]:
    raw = json.loads(FIXTURE.read_text())["segments"]
    return [Segment(start=x["start"], end=x["end"], text=x["text"]) for x in raw]


class EnvMixin:
    """Clear the env vars that would otherwise leak in from the caller's shell."""

    # Subclasses narrow this to the vars they care about.
    VARS: ClassVar[tuple[str, ...]] = ENV

    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in self.VARS}
        for key in self.VARS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ---------------------------------------------------------------- precedence


class TestPrecedence(EnvMixin, unittest.TestCase):
    """One rule -- flag, then env, then default -- used everywhere."""

    def test_defaults(self) -> None:
        resolved = S.resolve()
        self.assertEqual(len(resolved), 3)
        self.assertEqual(resolved[0], "")
        self.assertIsNone(resolved[1])
        self.assertIsNone(resolved[2])

    def test_env_only(self) -> None:
        os.environ["WATCHLESS_BASE_URL"] = "http://env/v1"
        self.assertEqual(S.resolve()[0], "http://env/v1")

    def test_flag_beats_env(self) -> None:
        os.environ["WATCHLESS_BASE_URL"] = "http://env/v1"
        self.assertEqual(S.resolve("http://flag/v1")[0], "http://flag/v1")

    def test_explicit_empty_model_clears_env(self) -> None:
        os.environ["WATCHLESS_MODEL"] = "env-model"
        self.assertIsNone(S.resolve(None, None, "")[2])

    def test_explicit_empty_key_clears_env(self) -> None:
        os.environ["WATCHLESS_API_KEY"] = "env-key"
        self.assertIsNone(S.resolve(None, "", None)[1])

    def test_explicit_empty_url_clears_env(self) -> None:
        os.environ["WATCHLESS_BASE_URL"] = "http://env/v1"
        self.assertEqual(S.resolve("")[0], "")

    def test_empty_env_value_is_ignored(self) -> None:
        os.environ["WATCHLESS_BASE_URL"] = ""
        self.assertEqual(S.resolve()[0], "")

    def test_trailing_slash_stripped(self) -> None:
        self.assertEqual(S.resolve("http://x/v1/")[0], "http://x/v1")

    def test_slug_one_rule(self) -> None:
        """The flag already carries the env default, so None never reaches it."""
        self.assertFalse(settings.env_flag("WATCHLESS_SLUG"))
        os.environ["WATCHLESS_SLUG"] = "1"
        self.assertTrue(settings.env_flag("WATCHLESS_SLUG"))

    def test_no_duplicate_env_reads_for_llm(self) -> None:
        """Regression: env was read in two modules and could disagree."""
        source = Path(yt.__file__).read_text()
        for var in ("WATCHLESS_BASE_URL", "WATCHLESS_API_KEY", "WATCHLESS_MODEL"):
            self.assertNotIn(
                f'environ.get("{var}")',
                source,
                f"{var} is resolved in settings/summarize.resolve(); the CLI "
                f"must not read it again",
            )


# ---------------------------------------------------------------- pure format


class TestDestPath(unittest.TestCase):
    """Filename planning is pure, so it needs no network at all."""

    TITLE = "Grippeimpfung fürs Kind – sinnvoll, auch wenn es gesund ist? 🤔💉"

    def test_stdout(self) -> None:
        self.assertIsNone(FMT.dest_path(None, None, False, self.TITLE, VID, 48))

    def test_out_wins(self) -> None:
        self.assertEqual(
            FMT.dest_path("/x/y.txt", "/d", True, self.TITLE, VID, 48),
            Path("/x/y.txt"),
        )

    def test_id_only(self) -> None:
        self.assertEqual(
            FMT.dest_path(None, "/d", False, self.TITLE, VID, 48),
            Path("/d") / f"{VID}.txt",
        )

    def test_slug(self) -> None:
        path = FMT.dest_path(None, "/d", True, "Hello World", VID, 48)
        self.assertEqual(path, Path("/d") / f"hello-world--{VID}.txt")

    def test_slug_without_title_falls_back_to_id(self) -> None:
        self.assertEqual(
            FMT.dest_path(None, "/d", True, None, VID, 48), Path("/d") / f"{VID}.txt"
        )

    def test_slug_disabled_by_zero(self) -> None:
        self.assertEqual(
            FMT.dest_path(None, "/d", True, "Hello", VID, 0), Path("/d") / f"{VID}.txt"
        )

    def test_slug_length_and_boundary(self) -> None:
        slug = FMT.slugify("one two three four five six", 12)
        self.assertLessEqual(len(slug), 12)
        self.assertFalse(slug.endswith("-"))


class TestSlugify(unittest.TestCase):
    def test_umlauts(self) -> None:
        self.assertIn("ae", FMT.slugify("Käse"))
        self.assertIn("ss", FMT.slugify("Straße"))

    def test_emoji_and_punctuation_stripped(self) -> None:
        self.assertEqual(FMT.slugify("Hallo, Welt! 🤔💉"), "hallo-welt")

    def test_no_trailing_dash(self) -> None:
        self.assertFalse(FMT.slugify("abc!!!").endswith("-"))

    def test_empty(self) -> None:
        self.assertEqual(FMT.slugify(""), "")


class TestReflow(unittest.TestCase):
    def test_fixture_reflows_to_paragraphs(self) -> None:
        self.assertGreater(len(FMT.reflow(fixture_segments()).split("\n\n")), 1)

    def test_paragraph_break_on_large_gap(self) -> None:
        segs = [Segment(0.0, 1.0, "One."), Segment(10.0, 11.0, "Two.")]
        self.assertEqual(len(FMT.reflow(segs).split("\n\n")), 2)

    def test_no_break_on_small_gap(self) -> None:
        segs = [Segment(0.0, 1.0, "One."), Segment(1.2, 2.0, "Two.")]
        self.assertEqual(len(FMT.reflow(segs).split("\n\n")), 1)

    def test_empty(self) -> None:
        self.assertEqual(FMT.reflow([]), "")

    def test_unpunctuated_track_still_forms_paragraphs(self) -> None:
        """Auto tracks often carry no full stops at all.

        Cutting only on sentence ends turned a real 454-word stand-up
        transcript into ONE block, because it contained zero of them. A
        segment is the unit that always exists, so it has to cap a paragraph.
        """
        segs = [
            Segment(float(i * 2), float(i * 2 + 1), "no punctuation here at all")
            for i in range(40)
        ]
        out = FMT.reflow(segs)
        paras = out.split("\n\n")
        self.assertGreater(len(paras), 5, "unpunctuated text collapsed to one block")
        self.assertLessEqual(max(len(p.split()) for p in paras), 40)
        self.assertEqual(" ".join(out.split()), " ".join(s.text for s in segs))

    def test_pause_break_survives_sentence_splitting(self) -> None:
        """A gap must still break when the segment also splits into sentences."""
        segs = [
            Segment(0.0, 1.0, "One. Two. Three."),
            Segment(30.0, 31.0, "Four. Five. Six."),
        ]
        self.assertEqual(len(FMT.reflow(segs).split("\n\n")), 2)


class TestRender(unittest.TestCase):
    SEGS = (Segment(0.0, 1.0, "Hello world."),)

    def _doc(self, **kw: object) -> FMT.Document:
        base: dict[str, object] = {
            "vid": VID,
            "url": "u",
            "segments": self.SEGS,
            "title": "T",
            "channel": "C",
            "backend": "api, en",
        }
        base.update(kw)
        return FMT.Document(**base)  # type: ignore[arg-type]

    def test_header_present(self) -> None:
        out = FMT.render(self._doc())
        for part in ("# T", "- URL: u", "- Channel: C", "- Track via: api, en"):
            self.assertIn(part, out)

    def test_no_header(self) -> None:
        self.assertEqual(FMT.render(self._doc(with_header=False)), "Hello world.\n")


class TestVideoId(unittest.TestCase):
    def test_forms(self) -> None:
        for url in (
            f"https://www.youtube.com/watch?v={VID}",
            f"https://youtu.be/{VID}",
            f"https://www.youtube.com/shorts/{VID}",
            f"https://www.youtube.com/embed/{VID}",
            f"https://www.youtube.com/live/{VID}",
            VID,
        ):
            self.assertEqual(FMT.video_id(url), VID, url)

    def test_rejects(self) -> None:
        for junk in ("", "not a url", "https://example.com/x"):
            with self.assertRaises(ValueError):
                FMT.video_id(junk)


class TestSummaryShape(unittest.TestCase):
    def test_bullet_budget_in_prompt(self) -> None:
        # Long enough to reach the ceiling, so the ceiling is what lands.
        prompt, _ = S.build_prompt("word " * 5000, max_bullets=7)
        self.assertIn("At most 7 content bullets", prompt)

    def test_normalise_and_cap(self) -> None:
        raw = "\n".join(f"- p{i}" for i in range(20))
        self.assertEqual(len(S.normalize_bullets(raw, 5).splitlines()), 5)

    def test_trim_keeps_ends(self) -> None:
        text = "A" * 100 + "B" * 100
        trimmed, was = S.trim(text, 40)
        self.assertTrue(was)
        self.assertTrue(trimmed.startswith("A"))
        self.assertTrue(trimmed.endswith("B"))


# ---------------------------------------------------------------- failures


class TestFailureClassification(unittest.TestCase):
    """An IP block must be reported as such, not as 'no subtitles'."""

    def test_ip_block_classified(self) -> None:
        error = F._classify(Exception("blocking requests from your IP"))
        self.assertIsInstance(error, BlockedError)

    def test_rate_limit_classified(self) -> None:
        self.assertIsInstance(F._classify(RuntimeError("HTTP Error 429")), BlockedError)

    def test_genuine_miss_not_a_block(self) -> None:
        self.assertNotIsInstance(
            F._classify(RuntimeError("no subtitles")), BlockedError
        )

    def test_blocked_backend_yields_a_blocked_outcome(self) -> None:
        """fetch_segments must keep 'blocked' legible in its error text.

        There is one backend now, so "a block beats a plain failure" is a
        property of fetch_segments rather than of a helper that ranked two
        backends' errors.
        """

        class Blocked:
            name = "ytdlp"

            def fetch(
                self, vid: str, video: Video, langs: list[str] | None = None
            ) -> tuple[Track, list[Segment]]:
                raise BlockedError("429 from YouTube")

        with unittest.mock.patch.object(yt, "YtdlpBackend", Blocked):
            outcome = yt.fetch_segments(Video(vid=VID), langs=None)
        self.assertEqual(outcome.segments, [])
        self.assertTrue(any("blocked" in e for e in outcome.errors), outcome.errors)


class TestExitCodes(unittest.TestCase):
    """Exit codes are the interface: termux-url-opener and corpus branch on them."""

    def _run(
        self, exc: BaseException, argv: list[str] | None = None
    ) -> tuple[int, str]:
        class Failing:
            name = "api"

            def fetch(
                self, vid: str, video: Video, langs: list[str] | None = None
            ) -> tuple[Track, list[Segment]]:
                raise exc

        saved_meta = yt.resolve_video
        yt.resolve_video = lambda vid, *, want_description: (Video(vid=vid), False)
        stderr = io.StringIO()
        code = 0
        try:
            with (
                contextlib.redirect_stderr(stderr),
                contextlib.redirect_stdout(io.StringIO()),
                unittest.mock.patch.object(yt, "YtdlpBackend", Failing),
            ):
                try:
                    yt.main(
                        [
                            f"https://youtube.com/shorts/{VID}",
                            *(argv or []),
                        ]
                    )
                except SystemExit as exc_info:
                    code = int(exc_info.code or 0)
        finally:
            yt.resolve_video = saved_meta
        return code, stderr.getvalue()

    def test_ip_block_exits_3(self) -> None:
        code, out = self._run(BlockedError("blocking requests from your IP"))
        self.assertEqual(code, yt.EXIT_BLOCKED)
        self.assertIn("blocking this IP", out)

    def test_genuine_miss_still_exits_2(self) -> None:
        code, out = self._run(FetchError("no subtitles"))
        self.assertEqual(code, yt.EXIT_NO_TRANSCRIPT)
        self.assertIn("needs ASR fallback", out)
        self.assertNotIn("blocking this IP", out)


class TestProxyConfig(unittest.TestCase):
    VARS = ("WATCHLESS_PROXY", "WATCHLESS_WEBSHARE_USER", "WATCHLESS_WEBSHARE_PASS")

    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in self.VARS}
        for key in self.VARS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_none_when_unset(self) -> None:
        self.assertIsNone(settings.proxy_url())

    def test_generic(self) -> None:
        os.environ["WATCHLESS_PROXY"] = "http://127.0.0.1:9050"
        self.assertEqual(settings.proxy_url(), "http://127.0.0.1:9050")

    def test_empty_is_none(self) -> None:
        os.environ["WATCHLESS_PROXY"] = ""
        self.assertIsNone(settings.proxy_url())


class TestSummaryGating(unittest.TestCase):
    """A summary that cannot happen must never be silently skipped."""

    def _run(
        self, argv: list[str], env: dict[str, str] | None = None
    ) -> tuple[int, str]:
        saved_env = {
            key: os.environ.get(key)
            for key in ("WATCHLESS_BASE_URL", "WATCHLESS_API_KEY", "WATCHLESS_MODEL")
        }
        for key in saved_env:
            os.environ.pop(key, None)
        os.environ.update(env or {})

        class Ok:
            name = "ytdlp"

            def fetch(
                self, vid: str, video: Video, langs: list[str] | None = None
            ) -> tuple[Track, list[Segment]]:
                return Track("de", True, ""), fixture_segments()

        stderr = io.StringIO()
        code = 0
        try:
            with (
                contextlib.redirect_stderr(stderr),
                contextlib.redirect_stdout(io.StringIO()),
                unittest.mock.patch.object(
                    yt,
                    "resolve_video",
                    lambda vid, *, want_description: (Video(vid=vid, lang="de"), False),
                ),
                unittest.mock.patch.object(yt, "YtdlpBackend", Ok),
            ):
                try:
                    yt.main(
                        [
                            f"https://youtube.com/shorts/{VID}",
                            "--no-header",
                            *argv,
                        ]
                    )
                except SystemExit as exc_info:
                    code = int(exc_info.code or 0)
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        return code, stderr.getvalue()

    def test_summary_flag_without_endpoint_exits_4(self) -> None:
        code, out = self._run(["--summary"])
        self.assertEqual(code, yt.EXIT_NO_ENDPOINT)
        self.assertIn("EXPORTED", out)

    def test_no_summary_with_endpoint_notes_it(self) -> None:
        code, out = self._run(
            ["--no-summary"], {"WATCHLESS_BASE_URL": "http://127.0.0.1:1/v1"}
        )
        self.assertEqual(code, 0)
        self.assertIn("endpoint is set but unused", out)


# ---------------------------------------------------------------- timestamps


class TestTimestamps(unittest.TestCase):
    """Regression: continuous captions must not all stamp 0:00."""

    def test_no_timestamps_by_default(self) -> None:
        self.assertNotIn("**[", FMT.reflow(fixture_segments()))

    def test_timestamps_prefixed(self) -> None:
        self.assertRegex(
            FMT.reflow(fixture_segments(), timestamps=True), r"\*\*\[\d+:\d{2}\]\*\* "
        )

    def test_stamps_are_distinct_and_increasing(self) -> None:
        out = FMT.reflow(fixture_segments(), timestamps=True)
        stamps = [
            line.split("**")[1] for line in out.splitlines() if line.startswith("**[")
        ]
        self.assertGreater(len(stamps), 2, "need several paragraphs")
        self.assertEqual(
            len(set(stamps)), len(stamps), f"duplicate timestamps: {stamps}"
        )

        def to_seconds(stamp: str) -> int:
            parts = [int(p) for p in stamp.strip("[]").split(":")]
            total = 0
            for part in parts:
                total = total * 60 + part
            return total

        values = [to_seconds(s) for s in stamps]
        self.assertEqual(values, sorted(values), f"not monotonic: {stamps}")

    def test_continuous_captions_keep_moving_timestamps(self) -> None:
        """Segments 2s apart with no pauses: stamps must still advance."""
        words = "eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf"
        segs = [
            Segment(i * 2.0, i * 2.0 + 1.9, f"{word}.")
            for i, word in enumerate(words.split())
        ]
        out = FMT.reflow(segs, timestamps=True)
        stamps = [
            line.split("**")[1] for line in out.splitlines() if line.startswith("**[")
        ]
        self.assertGreater(len(set(stamps)), 1, f"all stamps identical: {stamps}")

    def test_never_cuts_mid_sentence(self) -> None:
        out = FMT.reflow(fixture_segments(), timestamps=True)
        for para in out.split("\n\n"):
            body = para.split("]** ", 1)[-1].strip()
            self.assertTrue(
                body.endswith((".", "!", "?")),
                f"paragraph does not end a sentence: {body[-40:]!r}",
            )

    def test_sentence_straddling_segments_is_not_split(self) -> None:
        segs = [
            Segment(0.0, 1.0, "Dies ist ein Satz der"),
            Segment(1.0, 2.0, "über zwei Segmente geht."),
        ]
        self.assertEqual(
            FMT.reflow(segs), "Dies ist ein Satz der über zwei Segmente geht."
        )

    def test_paragraph_count_matches_with_and_without_timestamps(self) -> None:
        plain = FMT.reflow(fixture_segments())
        stamped = FMT.reflow(fixture_segments(), timestamps=True)
        self.assertEqual(len(plain.split("\n\n")), len(stamped.split("\n\n")))

    def test_dense_captions_do_not_explode(self) -> None:
        segs = [Segment(i * 0.4, i * 0.4 + 0.35, f"wort{i}.") for i in range(60)]
        self.assertLess(len(FMT.reflow(segs, timestamps=True).split("\n\n")), 25)

    def test_fmt_ts(self) -> None:
        for value, want in (
            (0, "0:00"),
            (65, "1:05"),
            (3599, "59:59"),
            (3600, "1:00:00"),
            (3661, "1:01:01"),
        ):
            self.assertEqual(FMT.fmt_ts(value), want)


# ---------------------------------------------------------------- document


class TestSections(unittest.TestCase):
    SEGS = (Segment(0.0, 1.0, "Hallo Welt."), Segment(2.0, 3.0, "Zweiter Satz."))
    DESC = "Quellen:\n- RKI"
    SUMM = "- Punkt eins.\n- Punkt zwei."

    def _render(self, **kw: object) -> str:
        base: dict[str, object] = {
            "vid": "abc",
            "url": "u",
            "segments": self.SEGS,
            "title": "T",
            "channel": "C",
            "backend": "b",
        }
        base.update(kw)
        return FMT.render(FMT.Document(**base))  # type: ignore[arg-type]

    def test_transcript_is_a_section(self) -> None:
        self.assertIn("## Transcript", self._render())

    def test_section_order_summary_description_transcript(self) -> None:
        out = self._render(description=self.DESC, summary=self.SUMM)
        self.assertLess(out.index("## Summary"), out.index("## Description"))
        self.assertLess(out.index("## Description"), out.index("## Transcript"))
        self.assertLess(
            out.index("- URL:"), out.index("## Summary"), "meta block first of all"
        )

    def test_no_line_separator_before_body(self) -> None:
        """The old '---' separator is gone; body sits under a heading."""
        self.assertNotIn("\n---\n", self._render())

    def test_summary_absent_without_it(self) -> None:
        self.assertNotIn("## Summary", self._render(description=self.DESC))

    def test_description_absent_without_it(self) -> None:
        self.assertNotIn("## Description", self._render(summary=self.SUMM))

    def test_transcript_always_present_with_header(self) -> None:
        for kw in ({}, {"description": self.DESC}, {"summary": self.SUMM}):
            self.assertIn("## Transcript", self._render(**kw))

    def test_body_appears_after_transcript_heading(self) -> None:
        out = self._render(summary=self.SUMM)
        self.assertLess(out.index("## Transcript"), out.index("Hallo Welt."))

    def test_no_header_still_labels_summary(self) -> None:
        out = self._render(with_header=False, summary=self.SUMM)
        self.assertTrue(out.startswith("## Summary"))
        self.assertIn("Hallo Welt.", out)
        self.assertNotIn("## Transcript", out)

    def test_no_header_no_summary_is_bare_body(self) -> None:
        out = self._render(with_header=False)
        self.assertTrue(out.startswith("Hallo Welt."))
        self.assertNotIn("##", out)


# ---------------------------------------------------------------- metadata


class TestMetadataCost(unittest.TestCase):
    """oEmbed first for title/channel; yt-dlp exactly once, never twice.

    This used to assert that a default run never touched yt-dlp at all,
    because the api backend supplied captions. With one backend, captions
    always need yt-dlp, so the property worth pinning is the one that costs
    a real second of wall-clock: whether the metadata call and the caption
    call share a single info() or each pay for their own.
    """

    def _run(
        self,
        argv: list[str],
        oembed: tuple[str | None, str | None] = ("OT", "OC"),
        info: Video | None = None,
    ) -> tuple[dict[str, int], list[str]]:
        calls = {"ytdlp": 0, "oembed": 0}
        # A default run needs the track list to have a track to pick, and that
        # only comes from info(). With one backend the metadata call and the
        # caption call are the same call, which is the thing being counted.
        detailed = info or Video(
            vid=VID,
            title="YT",
            channel="YC",
            lang="de",
            description="DESC",
            subtitles={"de": [{"ext": "json3", "url": "http://x/de.json3"}]},
        )

        def spy_info(_self: object, vid: str) -> Video:
            calls["ytdlp"] += 1
            return detailed

        class Ok:
            """Stands in for YtdlpBackend, counting info() calls.

            Both entry points have to be on this one object: resolve_video
            calls ``YtdlpBackend().info()`` for metadata and fetch_segments
            calls ``YtdlpBackend().fetch()`` for captions. Patching only
            ``info`` on the real class left fetch() hitting the network.
            """

            name = "ytdlp"
            info = spy_info

            def fetch(
                self, vid: str, video: Video, langs: list[str] | None = None
            ) -> tuple[Track, list[Segment]]:
                # Mirrors the real backend: when resolve_video did not already
                # populate the track lists (it returns after oEmbed unless the
                # description is wanted), fetch() calls info() itself.
                if not video.subtitles and not video.automatic_captions:
                    self.info(vid)
                return Track("de", True, ""), fixture_segments()

        def oe(vid: str) -> tuple[str | None, str | None]:
            calls["oembed"] += 1
            return oembed

        directory = tempfile.mkdtemp()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
            unittest.mock.patch.object(yt, "oembed_metadata", oe),
            unittest.mock.patch.object(yt, "YtdlpBackend", Ok),
            contextlib.suppress(SystemExit),
        ):
            yt.main(
                [f"https://youtube.com/shorts/{VID}", "--out-dir", directory, *argv]
            )
        return calls, [p.name for p in Path(directory).iterdir()]

    def test_default_run_makes_one_ytdlp_call(self) -> None:
        calls, files = self._run([])
        self.assertEqual(calls["oembed"], 1)
        self.assertEqual(calls["ytdlp"], 1, "captions share the one info() call")
        self.assertEqual(files, [f"{VID}.txt"])

    def test_slug_uses_oembed_title(self) -> None:
        calls, files = self._run(["--slug"])
        self.assertEqual(calls["ytdlp"], 1)
        self.assertEqual(files, [f"ot--{VID}.txt"], "title must come from oEmbed")

    def test_slug_falls_back_to_ytdlp_when_oembed_fails(self) -> None:
        calls, files = self._run(["--slug"], oembed=(None, None))
        self.assertEqual(calls["ytdlp"], 1, "fallback must not add a second call")
        self.assertEqual(files, [f"yt--{VID}.txt"])

    def test_description_does_not_add_a_second_ytdlp_call(self) -> None:
        """The description and the track list come from the same info()."""
        calls, _ = self._run(["--description"])
        self.assertEqual(calls["ytdlp"], 1)

    def test_description_written_when_asked_for(self) -> None:
        directory = tempfile.mkdtemp()
        detailed = Video(
            vid=VID, title="T", channel="C", lang="de", description="QUELLE"
        )

        class Ok:
            name = "ytdlp"

            def fetch(
                self, vid: str, video: Video, langs: list[str] | None = None
            ) -> tuple[Track, list[Segment]]:
                return Track("de", True, ""), fixture_segments()

        with (
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
            unittest.mock.patch.object(yt, "oembed_metadata", lambda v: ("T", "C")),
            unittest.mock.patch.object(
                yt,
                "YtdlpBackend",
                _stub_backend(lambda self, v: detailed),
            ),
        ):
            yt.main(
                [
                    f"https://youtube.com/shorts/{VID}",
                    "--out-dir",
                    directory,
                    "--description",
                ]
            )
        text = next(Path(directory).iterdir()).read_text()
        self.assertIn("## Description", text)
        self.assertIn("QUELLE", text)

    def test_description_is_written_whenever_it_was_fetched(self) -> None:
        """A summarising run fetches the description, so it must write it.

        It was fetched as summary input and then discarded, which meant
        paying for it and throwing away content that is as load-bearing as
        the transcript.
        """
        detailed = Video(
            vid=VID, title="T", channel="C", lang="de", description="QUELLE"
        )

        class Ok:
            name = "ytdlp"

            def fetch(
                self, vid: str, video: Video, langs: list[str] | None = None
            ) -> tuple[Track, list[Segment]]:
                return Track("de", True, ""), fixture_segments()

        directory = tempfile.mkdtemp()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
            unittest.mock.patch.object(yt, "oembed_metadata", lambda v: ("T", "C")),
            unittest.mock.patch.object(
                yt,
                "YtdlpBackend",
                _stub_backend(lambda self, v: detailed),
            ),
            unittest.mock.patch(
                "watchless.summarize.summarize",
                return_value=("CASE A - x\n- a point", False),
            ),
        ):
            # --llm-url makes it a summarising run, so the description is
            # fetched as input; --description is NOT passed.
            yt.main(
                [
                    f"https://youtube.com/shorts/{VID}",
                    "--out-dir",
                    directory,
                    "--llm-url",
                    "http://127.0.0.1:1/v1",
                ]
            )

        text = next(Path(directory).iterdir()).read_text()
        self.assertIn("## Description", text, "fetched description was discarded")
        self.assertIn("QUELLE", text)


# ---------------------------------------------------------------- tracks


class TestOriginalLanguage(unittest.TestCase):
    """A track in another language is a dub, not the speaker's words.

    Regression: the api backend used to take "the first non-generated track",
    which returns an uploader-uploaded Arabic dub for an English video. Four
    corpus videos were silently summarized from Arabic dubs because of it.
    """

    def _tracks(self, *specs: tuple[str, bool]) -> list[Track]:
        return [
            Track(lang=code, generated=generated, url="") for code, generated in specs
        ]

    def test_arabic_dub_is_not_chosen_over_english(self) -> None:
        tracks = self._tracks(("ar", False), ("ar", True), ("en", True))
        chosen = F.pick_by_lang(tracks, "en")
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen.lang, "en")

    def test_uploader_track_preferred_over_auto_in_same_language(self) -> None:
        tracks = self._tracks(("ar", False), ("en", False), ("en", True))
        chosen = F.pick_by_lang(tracks, "en")
        assert chosen is not None
        self.assertFalse(chosen.generated)

    def test_exact_code_beats_base_code(self) -> None:
        tracks = self._tracks(("en", True), ("en-US", False))
        chosen = F.pick_by_lang(tracks, "en-US")
        assert chosen is not None
        self.assertEqual(chosen.lang, "en-US")

    def test_base_code_fallback(self) -> None:
        chosen = F.pick_by_lang(self._tracks(("en", True)), "en-GB")
        assert chosen is not None
        self.assertEqual(chosen.lang, "en")

    def test_underscore_codes_normalised(self) -> None:
        chosen = F.pick_by_lang(self._tracks(("en_US", False)), "en-US")
        assert chosen is not None
        self.assertEqual(chosen.lang, "en_US")

    def test_no_match_returns_none_not_a_dub(self) -> None:
        self.assertIsNone(
            F.pick_by_lang(self._tracks(("ar", False), ("ar", True)), "en")
        )

    def test_explicit_lang_still_takes_a_dub(self) -> None:
        """--lang ar must still work: taking a dub on purpose is legitimate."""
        video = Video(vid=VID, lang="en")
        tracks = self._tracks(("ar", False), ("en", True))
        chosen = F._pick_track(tracks, ["ar"], video.lang)
        self.assertEqual(chosen.lang, "ar")

    def test_explicit_lang_rejects_unavailable(self) -> None:
        with self.assertRaises(FetchError):
            F._pick_track(self._tracks(("de", True)), ["pt"], "de")

    def test_unknown_language_falls_back_to_english(self) -> None:
        """When the original is unknown, prefer English over an arbitrary dub.

        Regression: a Georgian vlog offering 21 auto-dubs and no uploader track
        was transcribed from the ARABIC one, purely because it came first.
        """
        chosen = F._pick_track(
            self._tracks(("ar", True), ("en", True), ("de", True)), None, None
        )
        self.assertEqual(chosen.lang, "en")

    def test_fallback_is_configurable(self) -> None:
        with unittest.mock.patch.dict(
            os.environ, {"WATCHLESS_LANG_FALLBACK": "de"}, clear=False
        ):
            chosen = F._pick_track(
                self._tracks(("ar", True), ("en", True), ("de", True)), None, None
            )
        self.assertEqual(chosen.lang, "de")

    def test_fallback_can_be_disabled(self) -> None:
        with (
            unittest.mock.patch.dict(
                os.environ, {"WATCHLESS_LANG_FALLBACK": "none"}, clear=False
            ),
            self.assertRaises(FetchError),
        ):
            F._pick_track(self._tracks(("ar", True), ("de", True)), None, None)

    def test_single_language_needs_no_language_lookup(self) -> None:
        """One language is not ambiguous, so no yt-dlp call is needed."""
        with unittest.mock.patch.object(
            F.YtdlpBackend, "info", side_effect=AssertionError("must not be called")
        ):
            chosen = F._pick_track(
                self._tracks(
                    ("de", False),
                ),
                None,
                None,
            )
        self.assertEqual(chosen.lang, "de")

    def test_source_lang_is_looked_up_only_when_ambiguous(self) -> None:
        video = F.Video(vid=VID)
        with unittest.mock.patch.object(
            F.YtdlpBackend, "info", side_effect=AssertionError("must not be called")
        ):
            got = F.source_lang_for(VID, video, self._tracks(("de", True)))
        self.assertIsNone(got)

        with unittest.mock.patch.object(
            F.YtdlpBackend,
            "info",
            lambda self, v: F.Video(vid=VID, lang="ka"),
        ):
            got = F.source_lang_for(
                VID, video, self._tracks(("ar", True), ("ka", True))
            )
        self.assertEqual(got, "ka")

    def test_no_tracks_is_an_error(self) -> None:
        with self.assertRaises(FetchError):
            F._pick_track([], None, "en")


class TestJson3(unittest.TestCase):
    """The caption payload contract: json3, already non-overlapping."""

    def test_parses_events(self) -> None:
        payload = json.dumps(
            {
                "events": [
                    {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hello"}]},
                    {"tStartMs": 1000, "dDurationMs": 500, "segs": [{"utf8": "world"}]},
                ]
            }
        )
        segs = F.parse_json3(payload)
        self.assertEqual([s.text for s in segs], ["hello", "world"])
        self.assertEqual(segs[0].start, 0.0)
        self.assertEqual(segs[0].end, 1.0)

    def test_skips_blank_events(self) -> None:
        payload = json.dumps(
            {
                "events": [
                    {"tStartMs": 0, "dDurationMs": 10, "segs": [{"utf8": "  "}]},
                    {"tStartMs": 10, "dDurationMs": 10, "segs": [{"utf8": "x"}]},
                ]
            }
        )
        self.assertEqual(len(F.parse_json3(payload)), 1)

    def test_empty_track_is_an_error(self) -> None:
        with self.assertRaises(FetchError):
            F.parse_json3(json.dumps({"events": []}))

    def test_invalid_json_is_an_error(self) -> None:
        with self.assertRaises(FetchError):
            F.parse_json3("not json")

    def test_tracks_from_info_prefers_json3(self) -> None:
        tracks = F._tracks_from(
            {
                "subtitles": {
                    "en": [{"ext": "vtt", "url": "v"}, {"ext": "json3", "url": "j"}]
                },
                "automatic_captions": {},
            }
        )
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].url, "j")
        self.assertFalse(tracks[0].generated)

    def test_tracks_from_info_skips_languages_without_json3(self) -> None:
        tracks = F._tracks_from(
            {
                "subtitles": {"en": [{"ext": "vtt", "url": "v"}]},
                "automatic_captions": {"de": [{"ext": "json3", "url": "j"}]},
            }
        )
        self.assertEqual([t.lang for t in tracks], ["de"])
        self.assertTrue(tracks[0].generated)


# ---------------------------------------------------------------- env flags


class TestEnvFlags(EnvMixin, unittest.TestCase):
    """description / timestamps / slug: env default, flag wins, --no-X clears."""

    VARS: ClassVar[tuple[str, ...]] = (
        "WATCHLESS_DESCRIPTION",
        "WATCHLESS_TIMESTAMPS",
        "WATCHLESS_SLUG",
    )

    def test_env_flag_spellings(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on", " On "):
            os.environ["WATCHLESS_TIMESTAMPS"] = value
            self.assertTrue(settings.env_flag("WATCHLESS_TIMESTAMPS"), value)
        for value in ("", "0", "false", "no", "off", "maybe"):
            os.environ["WATCHLESS_TIMESTAMPS"] = value
            self.assertFalse(settings.env_flag("WATCHLESS_TIMESTAMPS"), value)
        os.environ.pop("WATCHLESS_TIMESTAMPS", None)
        self.assertFalse(settings.env_flag("WATCHLESS_TIMESTAMPS"))

    def _render(self, argv: list[str]) -> str:
        detailed = Video(
            vid=VID, title="T", channel="C", lang="de", description="QUELLE"
        )

        class Ok:
            name = "ytdlp"

            def fetch(
                self, vid: str, video: Video, langs: list[str] | None = None
            ) -> tuple[Track, list[Segment]]:
                return Track("de", True, ""), fixture_segments()

        directory = tempfile.mkdtemp()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
            unittest.mock.patch.object(yt, "oembed_metadata", lambda v: ("T", "C")),
            unittest.mock.patch.object(
                yt,
                "YtdlpBackend",
                _stub_backend(lambda self, v: detailed),
            ),
            contextlib.suppress(SystemExit),
        ):
            yt.main(
                [f"https://youtube.com/shorts/{VID}", "--out-dir", directory, *argv]
            )
        files = list(Path(directory).iterdir())
        return files[0].read_text() if files else ""

    def test_env_enables_description(self) -> None:
        # The env default is baked into the parser, so it is read at build time.
        with unittest.mock.patch.dict(os.environ, {"WATCHLESS_DESCRIPTION": "1"}):
            args = yt.build_parser().parse_args([VID])
        self.assertTrue(args.description)

    def test_env_enables_timestamps(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"WATCHLESS_TIMESTAMPS": "1"}):
            args = yt.build_parser().parse_args([VID])
        self.assertTrue(args.timestamps)

    def test_no_description_flag_beats_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"WATCHLESS_DESCRIPTION": "1"}):
            args = yt.build_parser().parse_args([VID, "--no-description"])
        self.assertFalse(args.description)

    def test_no_timestamps_flag_beats_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"WATCHLESS_TIMESTAMPS": "1"}):
            args = yt.build_parser().parse_args([VID, "--no-timestamps"])
        self.assertFalse(args.timestamps)

    def test_flags_work_without_env(self) -> None:
        text = self._render(["--description", "--timestamps"])
        self.assertIn("## Description", text)
        self.assertIn("**[", text)

    def test_on_by_default(self) -> None:
        text = self._render([])
        self.assertIn("## Description", text, "description should default ON")
        self.assertIn("**[", text, "timestamps should default ON")

    def test_no_flags_beat_defaults(self) -> None:
        text = self._render(["--no-description", "--no-timestamps"])
        self.assertNotIn("## Description", text)
        self.assertNotIn("**[", text)

    def test_env_false_turns_defaults_off(self) -> None:
        """Default-ON features must be switchable off from the env file.

        Unset has to mean "leave the default alone": reading it as falsey would
        silently disable both, which is the trap `not env_flag(...)` walks into.
        """
        for value in ("0", "false", "no", "off"):
            with unittest.mock.patch.dict(
                os.environ,
                {"WATCHLESS_DESCRIPTION": value, "WATCHLESS_TIMESTAMPS": value},
            ):
                args = yt.build_parser().parse_args([VID])
            self.assertFalse(args.description, value)
            self.assertFalse(args.timestamps, value)

    def test_unset_env_leaves_defaults_on(self) -> None:
        clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("WATCHLESS_DESCRIPTION", "WATCHLESS_TIMESTAMPS")
        }
        with unittest.mock.patch.dict(os.environ, clean, clear=True):
            args = yt.build_parser().parse_args([VID])
        self.assertTrue(args.description)
        self.assertTrue(args.timestamps)

    def test_summary_input_has_no_timestamps(self) -> None:
        """The model gets clean prose; only the written file carries markers.

        A marker every few lines is noise the model has to spend attention
        ignoring, while a reader wants it to locate a passage.
        """
        self.assertNotIn(
            "**[",
            yt._transcript_text(fixture_segments()),
            "timeline markers must never reach the summary prompt",
        )

    def test_written_file_keeps_timestamps(self) -> None:
        """Same segments: markers in the file, none in the summary input."""
        text = self._render([])
        self.assertIn("## Transcript", text)
        self.assertIn("**[", text, "the reader gets timeline markers")
        self.assertNotIn(
            "**[",
            yt._transcript_text(fixture_segments()),
            "the model must not",
        )

    def test_description_default_on_is_fetched(self) -> None:
        """Default-on description must actually reach yt-dlp, not just the parser."""
        text = self._render([])
        self.assertIn("## Description", text)
        self.assertIn("QUELLE", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSummaryFailureIsVisible(unittest.TestCase):
    """An LLM failure must name its own error and keep the transcript.

    The old code caught ``FetchError`` around the summarise call, which
    ``SummarizeError`` (a ``RuntimeError``) is not -- so that branch was
    unreachable. The outcome was the same message via the catch-all, which is
    why this was dead code rather than a bug; this pins the behaviour so
    splitting the handler cannot lose either half.
    """

    def test_summarize_error_is_reported_and_transcript_kept(self) -> None:
        detailed = Video(vid=VID, title="T", channel="C", lang="de")

        class Ok:
            name = "ytdlp"

            def fetch(
                self, vid: str, video: Video, langs: list[str] | None = None
            ) -> tuple[Track, list[Segment]]:
                return Track("de", True, ""), fixture_segments()

        directory = tempfile.mkdtemp()
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(io.StringIO()),
            unittest.mock.patch.object(yt, "oembed_metadata", lambda v: ("T", "C")),
            unittest.mock.patch.object(
                yt,
                "YtdlpBackend",
                _stub_backend(lambda self, v: detailed),
            ),
            unittest.mock.patch(
                "watchless.summarize.summarize",
                side_effect=SummarizeError("boom: endpoint said no"),
            ),
        ):
            yt.main(
                [
                    f"https://youtube.com/shorts/{VID}",
                    "--out-dir",
                    directory,
                    "--llm-url",
                    "http://127.0.0.1:1/v1",
                ]
            )

        out = stderr.getvalue()
        self.assertIn(
            "boom: endpoint said no", out, "the LLM's own message must reach the user"
        )

        written = next(Path(directory).iterdir()).read_text()
        self.assertTrue(written.strip(), "the transcript must survive")
        self.assertNotIn(
            "## Summary", written, "a failed summary must not write a Summary section"
        )


class TestBlockedIsATypeNotAWord(unittest.TestCase):
    """A block must be carried as a flag, not recovered by substring search.

    FetchOutcome used to hold only strings, so "was this an IP block?" was
    answered by looking for "blocked" inside the error text. Any video whose
    error merely mentioned the word would then be reported as a block, and the
    user told to change networks for a video that simply has no captions.
    """

    def test_error_text_containing_blocked_is_not_a_block(self) -> None:
        outcome = yt.FetchOutcome([], None, ["subtitles unavailable"])
        self.assertFalse(outcome.blocked)

    def test_blocked_flag_survives_independently_of_the_message(self) -> None:
        outcome = yt.FetchOutcome([], None, ["transient error"], blocked=True)
        self.assertTrue(outcome.blocked)
        self.assertNotIn("blocked", " ".join(outcome.errors))


class TestTrackProvenance(unittest.TestCase):
    """An uploader caption must beat an auto one, even when the code differs.

    The old order tested exactness first, so with src_lang "en" and tracks
    en-GB (uploader) + en (auto), the auto track won. That is backwards: an ASR
    track has no punctuation, no speaker labels and mishears proper nouns.
    """

    def test_uploader_track_wins_over_exact_auto_track(self) -> None:
        tracks = [
            F.Track(lang="en-GB", generated=False, url="u"),
            F.Track(lang="en", generated=True, url="a"),
        ]
        got = F.pick_by_lang(tracks, "en")
        assert got is not None
        self.assertEqual(got.lang, "en-GB")
        self.assertFalse(got.generated)

    def test_exact_code_still_wins_between_uploader_tracks(self) -> None:
        tracks = [
            F.Track(lang="en", generated=False, url="x"),
            F.Track(lang="en-GB", generated=False, url="y"),
        ]
        got = F.pick_by_lang(tracks, "en-GB")
        assert got is not None
        self.assertEqual(got.lang, "en-GB")

    def test_auto_track_is_used_when_nothing_else_matches(self) -> None:
        got = F.pick_by_lang([F.Track(lang="de", generated=True, url="a")], "de")
        assert got is not None
        self.assertTrue(got.generated)

    def test_no_matching_language(self) -> None:
        self.assertIsNone(
            F.pick_by_lang([F.Track(lang="fr", generated=False, url="z")], "en")
        )


class TestTermuxInstaller(unittest.TestCase):
    """--install-termux-opener: install, refuse, and never clobber silently.

    The failure that matters is a silent one: writing the opener somewhere the
    Termux app never reads, or replacing a script someone hand-edited.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.prefix = self.tmp / "data/data/com.termux/files/usr"
        self.bin = self.prefix / "bin"
        self.bin.mkdir(parents=True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self._env = mock.patch.dict(
            os.environ,
            {
                "PREFIX": str(self.prefix),
                "HOME": str(self.home),
                "PATH": f"{self.bin}:{os.environ['PATH']}",
            },
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def _stub(self, *names: str) -> None:
        for name in names:
            p = self.bin / name
            p.write_text("#!/bin/sh\n")
            p.chmod(0o755)

    def test_not_termux_writes_nothing(self) -> None:
        env = {"PATH": os.environ["PATH"], "HOME": str(self.home)}
        with mock.patch.dict(os.environ, env, clear=True):
            os.environ.pop("PREFIX", None)
            with mock.patch.object(T, "_TERMUX_PREFIX", "/nonexistent/termux"):
                self.assertFalse(T.in_termux())
                self.assertEqual(T.install_opener(), 1)
        self.assertFalse(T.installer_path().exists())

    def test_installs_executable_opener(self) -> None:
        self._stub("termux-info")
        self.assertEqual(T.install_opener(), 0)
        dest = T.installer_path()
        self.assertTrue(dest.is_file())
        self.assertTrue(os.access(dest, os.X_OK))
        self.assertIn("termux-url-opener", dest.name)

    def test_an_answer_of_yes_overwrites(self) -> None:
        self._stub("termux-info")
        dest = T.installer_path()
        dest.parent.mkdir(parents=True)
        dest.write_text("# old\n")
        buf = io.StringIO()
        with (
            contextlib.redirect_stdout(buf),
            mock.patch.object(sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", return_value="y"),
        ):
            rc = T.install_opener()
        self.assertEqual(rc, 0)
        self.assertNotIn("# old", dest.read_text())

    def test_missing_termux_api_is_reported_not_fatal(self) -> None:
        # termux-toast absent is a worse experience, not a broken install.
        self._stub("termux-info")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = T.install_opener()
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("termux-toast", out)
        self.assertIn("pkg install termux-api", out)
        # The app is a separate F-Droid install; no pkg command can do it.
        self.assertIn("F-Droid", out)

    def test_existing_opener_is_not_replaced_without_consent(self) -> None:
        self._stub("termux-info")
        dest = T.installer_path()
        dest.parent.mkdir(parents=True)
        dest.write_text("#!/bin/sh\n# hand-edited, do not touch\n")
        buf = io.StringIO()
        # Pin the non-tty branch: under a real terminal the installer would
        # block on input(), and a test must never depend on who ran it.
        with (
            contextlib.redirect_stdout(buf),
            mock.patch.object(sys.stdin, "isatty", return_value=False),
        ):
            rc = T.install_opener()
        self.assertEqual(rc, 0)
        self.assertIn("hand-edited", dest.read_text())
        self.assertIn("--force", buf.getvalue())

    def test_force_replaces(self) -> None:
        self._stub("termux-info")
        dest = T.installer_path()
        dest.parent.mkdir(parents=True)
        dest.write_text("# old\n")
        self.assertEqual(T.install_opener(force=True), 0)
        self.assertNotIn("# old", dest.read_text())

    def test_prompt_is_bundled_with_the_package(self) -> None:
        # Without package-data the wheel installs fine and this raises at first
        # use, long after the regression.
        text = files("watchless.data").joinpath("termux-url-opener").read_text("utf-8")
        self.assertIn("termux-url-opener", text)
        self.assertTrue(text.startswith("#!"))
