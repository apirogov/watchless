"""Tests for the saved-file side: corpus.py's case reader, resummarize.py's
document parse and rewrite.

These cover two bugs that silently produced wrong output rather than crashing:

  * report.parse_case() scanned the whole summary for the bare word "both",
    so "CASE A -- a review where both keyboards were measured" was reported as
    CASE BOTH, hiding exactly the mismatches REPORT.md exists to show.
  * resummarize.rewrite() bounded the old Summary section on the next "## "
    heading, which a --no-header file does not have, so --in-place deleted the
    transcript.
"""

import json
import re
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from typing import ClassVar

from tests.bench import corpus as C
from tests.bench import report as RP
from tests.bench import resummarize as R
from watchless import fetch as F
from watchless import settings
from watchless import summarize as S


class TestParseCase(unittest.TestCase):
    """report.parse_case reads the classification from a run's log.

    It used to parse the ## Summary section. That moved: the classification is
    the benchmark's vocabulary, not something a reader asked to see, so it is a
    ``[kind]`` stderr diagnostic now and the .md never contains it.
    """

    def test_reads_the_kind_diagnostic(self) -> None:
        for line, want in (
            ("[kind] A", "A"),
            ("[kind] B", "B"),
            ("[kind] BOTH", "BOTH"),
            ("[kind] both", "BOTH"),
        ):
            self.assertEqual(RP.parse_case(line + "\n"), want, line)

    def test_found_amongst_other_log_lines(self) -> None:
        log = (
            "[meta] yt-dlp: title=ok lang=en desc=ok\n"
            "[kind] B\n"
            "wrote thing.txt (400 words)\n"
        )
        self.assertEqual(RP.parse_case(log), "B")

    def test_missing_or_unrecognised(self) -> None:
        self.assertEqual(RP.parse_case(""), "-")
        self.assertEqual(RP.parse_case("[meta] nothing here\n"), "-")
        self.assertEqual(RP.parse_case("[kind] C\n"), "-")

    def test_old_style_case_line_in_a_file_is_not_read(self) -> None:
        """The old location must stay dead, or the split leaks back."""
        self.assertEqual(RP.parse_case("## Summary\n\nCASE A — explainer\n"), "-")


class TestParseDoc(unittest.TestCase):
    """resummarize.parse_doc: pick out the real transcript, never the summary."""

    def test_normal_document(self) -> None:
        md = (
            "# Title\n- Channel: C\n\n## Summary\n\nold\n\n## Description\n\n"
            "desc\n\n## Transcript\n\nbody\n"
        )
        d = R.parse_doc(md)
        self.assertEqual(d["transcript"], "body")
        self.assertEqual(d["description"], "desc")
        self.assertEqual(d["title"], "Title")
        self.assertEqual(d["channel"], "C")

    def test_no_header_with_summary_excludes_the_summary(self) -> None:
        # regression: the previous summary was fed back in as the transcript,
        # so every rerun summarised its own output
        md = (
            "## Summary\n\nCASE A - x\n- [fact] a\n- [fact] b\n\n"
            "Real transcript one. Two.\n"
        )
        self.assertEqual(R.parse_doc(md)["transcript"], "Real transcript one. Two.")

    def test_no_header_without_summary_is_all_body(self) -> None:
        md = "Just body text. Second sentence.\n"
        self.assertEqual(
            R.parse_doc(md)["transcript"], "Just body text. Second sentence."
        )

    def test_no_header_description_is_empty_not_none(self) -> None:
        self.assertEqual(R.parse_doc("## Transcript\n\nbody\n")["description"], "")


class TestRewrite(unittest.TestCase):
    """resummarize.rewrite: replace the summary, keep everything else."""

    def test_no_header_keeps_the_transcript(self) -> None:
        # regression: this returned '## Summary\n\nnew\n- b\n\n' and nothing else
        md = "## Summary\n\nold summary\n- a\n\nReal transcript one. Two.\n"
        out = R.rewrite(md, "new summary\n- b")
        self.assertIn("Real transcript one. Two.", out)
        self.assertNotIn("old summary", out)
        self.assertIn("new summary", out)
        self.assertEqual(out.count("## Summary"), 1)

    def test_normal_document_keeps_description_and_transcript(self) -> None:
        md = (
            "# T\n- Channel: C\n\n## Summary\n\nold\n- a\n\n## Description\n\n"
            "desc here\n\n## Transcript\n\nbody text\n"
        )
        out = R.rewrite(md, "new\n- b")
        self.assertIn("desc here", out)
        self.assertIn("body text", out)
        self.assertNotIn("old", out.replace("new", ""))

    def test_blank_line_before_trust_is_replaced_not_orphaned(self) -> None:
        """Trust now sits after a blank line, and the old one must still go.

        Bounding the body on contiguous lines stopped at that blank: the
        bullets were replaced and the previous Trust line stayed behind under
        the new one, so a re-summarised file ended up with two.
        """
        md = (
            "## Summary\n\n- [fact] a\n- [fact] b\n\nTrust: old verdict.\n\n"
            "## Description\n\ndesc\n\n## Transcript\n\nbody\n"
        )
        out = R.rewrite(md, "- [fact] z\n\nTrust: new verdict.")
        self.assertNotIn("old verdict", out, out)
        self.assertEqual(out.count("Trust:"), 1, out)
        self.assertIn("desc", out)
        self.assertIn("body", out)

    def test_multi_paragraph_summary_does_not_eat_the_description(self) -> None:
        """Allowing the blank line must not let the match run past the summary."""
        md = "## Summary\n\n- a\n\n- b\n\nTrust: x\n\n## Transcript\n\nbody\n"
        out = R.rewrite(md, "- z")
        self.assertIn("body", out, out)
        self.assertNotIn("Trust: x", out)

    def test_inserts_before_description_when_missing(self) -> None:
        md = "# T\n\n## Description\n\ndesc\n\n## Transcript\n\nbody\n"
        out = R.rewrite(md, "sum\n- b")
        self.assertLess(out.index("## Summary"), out.index("## Description"))

    def test_inserts_before_transcript_when_missing(self) -> None:
        md = "# T\n\n## Transcript\n\nbody\n"
        out = R.rewrite(md, "sum\n- b")
        self.assertLess(out.index("## Summary"), out.index("## Transcript"))

    def test_round_trips(self) -> None:
        """Rewriting a rewritten file must not grow or lose anything."""
        md = "## Summary\n\nold\n- a\n\nBody one. Two.\n"
        once = R.rewrite(md, "s1\n- b")
        twice = R.rewrite(once, "s2\n- c")
        self.assertEqual(once.count("## Summary"), 1)
        self.assertEqual(twice.count("## Summary"), 1)
        self.assertIn("Body one. Two.", twice)
        self.assertEqual(R.parse_doc(twice)["transcript"], "Body one. Two.")


class TestYtdlpBackend(unittest.TestCase):
    """The yt-dlp backend used to shell out and pick a track by filename.

    It now works in memory: the track is chosen from YouTube's own track list
    and its content is fetched from the URL that list gave. That removes the
    scratch directory, the argv surface, and the bug where ``--lang en,de``
    returned German because "de" sorts before "en".
    """

    INFO: ClassVar[dict[str, object]] = {
        "title": "T",
        "uploader": "C",
        "language": "de",
        "description": "D",
        "subtitles": {
            "de": [{"ext": "json3", "url": "https://example.invalid/de"}],
        },
        "automatic_captions": {
            "en": [{"ext": "json3", "url": "https://example.invalid/en"}],
        },
    }

    def _info_video(self) -> F.Video:
        backend = F.YtdlpBackend(sleep_before_subtitles=0)
        with unittest.mock.patch.object(backend, "_client") as client:
            client.return_value.extract_info.return_value = dict(self.INFO)
            return backend.info("abc123")

    def test_info_maps_fields(self) -> None:
        video = self._info_video()
        self.assertEqual(video.title, "T")
        self.assertEqual(video.channel, "C")
        self.assertEqual(video.lang, "de")
        self.assertEqual(video.description, "D")

    def test_info_keeps_the_track_lists(self) -> None:
        video = self._info_video()
        self.assertIn("de", video.subtitles)
        self.assertIn("en", video.automatic_captions)

    def test_fetch_reads_the_chosen_track_in_memory(self) -> None:
        video = F.Video(
            vid="abc123",
            lang="de",
            subtitles={"de": [{"ext": "json3", "url": "https://example.invalid/de"}]},
        )
        payload = json.dumps(
            {
                "events": [
                    {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hello"}]}
                ]
            }
        )
        backend = F.YtdlpBackend(sleep_before_subtitles=0)
        response = unittest.mock.MagicMock()
        response.read.return_value = payload.encode()
        response.__enter__.return_value = response
        with unittest.mock.patch.object(backend, "_client") as client:
            client.return_value.urlopen.return_value = response
            track, segments = backend.fetch("abc123", video)

        self.assertEqual(track.label, "de (uploader)")
        self.assertEqual([s.text for s in segments], ["hello"])

    def test_no_scratch_directory_is_ever_created(self) -> None:
        """Regression: every yt-dlp run used to leak a temp dir."""
        scratch = Path(tempfile.gettempdir())
        before = set(scratch.glob("watchless-*"))
        self._info_video()
        self.assertEqual(set(scratch.glob("watchless-*")) - before, set())


class TestLangOrder(unittest.TestCase):
    """``--lang en,de`` must honour the order given, not alphabetical order."""

    def _tracks(self, *codes: str) -> list[F.Track]:
        return [F.Track(lang=c, generated=True, url="") for c in codes]

    def test_requested_order_wins(self) -> None:
        self.assertEqual(
            F._pick_track(self._tracks("de", "en"), ["en", "de"], "de").lang, "en"
        )
        self.assertEqual(
            F._pick_track(self._tracks("de", "en"), ["de", "en"], "de").lang, "de"
        )

    def test_uploader_beats_auto_for_the_same_language(self) -> None:
        tracks = [F.Track("en", True, ""), F.Track("en", False, "")]
        chosen = F._pick_track(tracks, ["en"], None)
        self.assertFalse(chosen.generated)


if __name__ == "__main__":
    unittest.main(verbosity=2)


OPENER = Path(__file__).parent.parent / "src/watchless/data/termux-url-opener"


def _opener_text() -> str:
    return OPENER.read_text(encoding="utf-8")


class TestEnvVarCoverage(unittest.TestCase):
    """Every WATCHLESS_* the code reads must be documented in the README table.

    The termux opener's hardcoded export list drifted: four variables worked
    from a terminal and were silently dropped from the share sheet, because
    nothing connected "what the code reads" to "what the wrapper forwards".
    This is that connection.
    """

    def _read(self) -> set[str]:
        src = Path(__file__).parent.parent / "src" / "watchless"
        sources = list(src.glob("*.py"))
        names: set[str] = set()
        for path in sources:
            names.update(re.findall(r"(WATCHLESS_[A-Z_]+)", path.read_text()))
        return names

    def test_every_env_var_is_in_the_readme(self) -> None:
        readme = (Path(__file__).parent.parent / "README.md").read_text()
        missing = sorted(n for n in self._read() if n not in readme)
        self.assertEqual(missing, [], f"undocumented env vars: {missing}")

    def test_opener_does_not_hardcode_a_list(self) -> None:
        """It must source the env file under `set -a`, not re-export a list."""
        opener = _opener_text()
        self.assertNotIn("for _v in WATCHLESS_", opener)
        self.assertIn("set -a", opener)


class TestNoRuntimeStubImports(unittest.TestCase):
    """Stub-only names must never be imported at runtime.

    mypy was happy to accept ``from yt_dlp import _YoutubeDLOptions`` because
    the stubs declare it, but yt-dlp does not export it at runtime: every
    ytdlp fetch died with ImportError while the suite stayed green, because
    the tests stub the client and never import the real module's options.
    """

    def test_types_yt_dlp_names_are_guarded(self) -> None:
        offenders: list[str] = []
        root = Path(__file__).parent.parent / "src" / "watchless"
        for path in root.glob("*.py"):
            text = path.read_text()
            if "_YoutubeDLOptions" not in text:
                continue
            # Every mention must sit under a TYPE_CHECKING guard.
            for line_no, line in enumerate(text.splitlines(), 1):
                if "_YoutubeDLOptions" not in line or line.lstrip().startswith("#"):
                    continue
                before = "\n".join(text.splitlines()[:line_no])
                if "if TYPE_CHECKING:" not in before:
                    offenders.append(f"{path.name}:{line_no}")
        self.assertEqual(
            offenders,
            [],
            f"_YoutubeDLOptions imported outside TYPE_CHECKING: {offenders}",
        )

    def test_ytdlp_backend_imports_for_real(self) -> None:
        """Importing the module must not require the stubs."""
        backend = F.YtdlpBackend()
        # Building a client is the operation that used to raise ImportError.
        client = backend._client()
        self.assertIsNotNone(client)


class TestOpenerResolvesTheCommand(unittest.TestCase):
    """The opener must not hardcode a path that the package layout can delete.

    Regression: the uv migration removed watchless.py, but the opener kept
    ``PY="$HOME/bin/watchless.py"`` and bailed at its existence guard, so every
    share-sheet run failed while the terminal kept working. The env-var
    coverage test above missed it because it only looks at WATCHLESS_* names.
    """

    def _opener(self) -> str:
        return _opener_text()

    def test_no_hardcoded_script_path(self) -> None:
        """No code line may name a .py path; comments explaining why are fine."""
        offenders = [
            line
            for line in self._opener().splitlines()
            if "watchless.py" in line and not line.lstrip().startswith("#")
        ]
        self.assertEqual(offenders, [], f"hardcoded script path: {offenders}")

    def test_resolves_a_command_with_fallbacks(self) -> None:
        text = self._opener()
        self.assertIn(".local/bin/watchless", text, "tier 2: the uv tool install dir")
        self.assertNotIn(
            ".venv/bin/watchless",
            text,
            "the venv layout is gone: `uv tool install` owns the command now",
        )
        self.assertIn("command -v watchless", text, "tier 2: on PATH")
        self.assertIn("python3 -m watchless.cli", text, "tier 3: importable module")

    def test_missing_command_is_reported_not_silent(self) -> None:
        self.assertIn("watchless not found", self._opener())


class TestSingleSourceOfConfig(unittest.TestCase):
    """A default must be *assigned* in one place only.

    Regression: DEFAULT_MAX_BULLETS lived in settings.py while resummarize took
    its default from summarize.MAX_BULLETS and cli from
    settings.DEFAULT_MAX_BULLETS. Both looked authoritative; changing one
    changed only one entry point.
    """

    TUNABLES = (
        "TIMEOUT_S",
        "DEFAULT_MAX_CHARS",
        "DEFAULT_MAX_BULLETS",
        "DESC_MAX_CHARS",
    )

    def _assignments(self) -> dict[str, list[str]]:
        """Which module *assigns* each tunable (imports and re-exports excluded)."""
        src = Path(__file__).parent.parent / "src" / "watchless"
        found: dict[str, list[str]] = {}
        for py in sorted(src.glob("*.py")):
            for line in py.read_text().splitlines():
                if not line or line[0].isspace() or line.startswith("#"):
                    continue  # only module-level, non-comment lines
                for name in self.TUNABLES:
                    if line.startswith(f"{name} =") or line.startswith(f"{name}="):
                        found.setdefault(name, []).append(py.name)
        return found

    def test_each_default_is_assigned_once(self) -> None:
        for name, where in self._assignments().items():
            self.assertEqual(where, ["settings.py"], f"{name} assigned in {where}")

    def test_no_module_shadows_a_settings_name(self) -> None:
        """A module-local knob must not reuse a name settings.py already owns.

        metadata.py had its own TIMEOUT_S = 15 (oEmbed) while settings.py owns
        TIMEOUT_S = 300 (the LLM). Different knobs, same name, so a reader
        could not tell which was which -- now OEMBED_TIMEOUT_S.
        """
        src = Path(__file__).parent.parent / "src" / "watchless"
        owned = set(self.TUNABLES)
        offenders: list[str] = []
        for py in sorted(src.glob("*.py")):
            if py.name == "settings.py":
                continue
            for line in py.read_text().splitlines():
                if not line or line[0].isspace():
                    continue
                for name in owned:
                    if line.startswith(f"{name} =") or line.startswith(f"{name}="):
                        offenders.append(f"{py.name}: {line.strip()}")
        self.assertEqual(offenders, [], f"shadowed settings names: {offenders}")

    def test_every_tunable_is_declared(self) -> None:
        self.assertEqual(
            sorted(self._assignments()), sorted(self.TUNABLES), "a tunable vanished"
        )

    def test_both_entry_points_agree(self) -> None:
        self.assertEqual(S.MAX_BULLETS, settings.DEFAULT_MAX_BULLETS)
        self.assertEqual(S.DEFAULT_MAX_CHARS, settings.DEFAULT_MAX_CHARS)
        self.assertEqual(S.DESC_MAX_CHARS, settings.DESC_MAX_CHARS)
        self.assertEqual(S.TIMEOUT_S, settings.TIMEOUT_S)


class TestCorpusForwarding(unittest.TestCase):
    """The corpus harness forwards unknown flags to watchless -- but not silently.

    Internal benchmark harness (no console script), but the bug was real:
    parse_known_args exists so the two parsers need not be edited in lockstep,
    and the failure mode is a typo -- --bulits 20 was accepted, dropped, and the
    run quietly used the default with no fetch ever validating it.
    """

    def test_unknown_flag_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            C._reject_unknown_flags(["--bulits", "20"])
        self.assertIn("--bulits", str(caught.exception))

    def test_known_flag_is_forwarded(self) -> None:
        C._reject_unknown_flags(["--bullets", "20", "--no-summary"])

    def test_flag_with_value_is_checked_on_its_name_only(self) -> None:
        C._reject_unknown_flags(["--llm-url=http://x/v1"])


class TestCorpusLocation(unittest.TestCase):
    """The bench lives under tests/ and must look in the right place.

    A wrong OUT path does not raise -- it makes every bucket look missing, which
    reads as "the corpus is gone" rather than "the path is wrong".
    """

    def test_data_directory_sits_beside_the_harness(self) -> None:
        self.assertEqual(C.OUT, Path(C.__file__).resolve().parent / "corpus")
        self.assertTrue(C.OUT.is_dir())

    def test_declared_buckets_resolve_to_data_or_are_known_missing(self) -> None:
        present = {p.stem.split("-", 1)[1] for p in C.OUT.glob("*.md") if "-" in p.stem}
        declared = {key for key, _e, _w, _q in C.VIDEOS}
        # Every file on disk belongs to a declared bucket, and the only buckets
        # without data are the three that cannot be fetched (no captions).
        self.assertEqual(present - declared, set())
        self.assertEqual(
            declared - present, {"music-with-captions", "asmr-unboxing", "silent-craft"}
        )


class TestCorpusScoring(unittest.TestCase):
    """A bucket's expectation has to be able to fail, or the report is theatre."""

    def test_exact_expectation_rejects_other_verdicts(self) -> None:
        self.assertFalse(C._mismatch("A", "A"))
        self.assertTrue(C._mismatch("A", "B"))
        self.assertTrue(C._mismatch("A", "BOTH"))

    def test_a_or_b_accepts_a_b_or_both_verdict(self) -> None:
        """A|B is a boundary case, not a wildcard.

        The previous test put "A|B" inside the expectation check, which excused
        the expectation instead of widening it: every A|B bucket accepted any
        verdict, so personal-vlog and game-analysis could never fail.
        """
        self.assertFalse(C._mismatch("A|B", "A"))
        self.assertFalse(C._mismatch("A|B", "B"))
        self.assertTrue(C._mismatch("A|B", "BOTH"))

    def test_both_expectation_is_its_own_value(self) -> None:
        self.assertFalse(C._mismatch("both", "BOTH"))
        self.assertTrue(C._mismatch("both", "A"))

    def test_no_detection_is_not_a_mismatch(self) -> None:
        # "-" means no classification was produced. That is a gap to show, not
        # a wrong answer to flag.
        self.assertFalse(C._mismatch("A", "-"))
