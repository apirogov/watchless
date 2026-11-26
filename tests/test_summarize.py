"""Tests for the summarize module against a local mock OpenAI-compatible
server."""

import contextlib
import io
import json
import os
import re
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

from watchless import llm
from watchless import summarize as S

LAST: dict[str, Any] = {}
FORCED: dict[str, Any] = {"body": None, "tries": 0, "models": None}
MODE: dict[str, str] = {"v": "ok"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a: object) -> None:
        pass

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            ids = FORCED.get("models")
            if ids is None:
                self._send(404, b'{"error":"no route"}')
                return
            body = json.dumps(
                {"object": "list", "data": [{"id": m} for m in ids]}
            ).encode()
            self._send(200, body)
        else:
            self._send(404, b"{}")

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        LAST["body"] = json.loads(self.rfile.read(n))
        LAST["auth"] = self.headers.get("Authorization")
        LAST["path"] = self.path

        if MODE["v"] == "fail_then_ok":
            FORCED["tries"] += 1
            if FORCED["tries"] == 1:
                self._send(500, b'{"error":"transient"}')
                return

        if FORCED["body"] is not None:
            self._send(200, FORCED["body"])
            return

        if MODE["v"] == "500":
            self._send(500, b'{"error":"boom"}')
        elif MODE["v"] == "429":
            self._send(429, b'{"error":"rate limited"}')
        elif MODE["v"] == "400":
            self._send(400, b'{"error":"bad request"}')
        elif MODE["v"] == "badjson":
            self._send(200, b"<html>not json</html>")
        elif MODE["v"] == "shape":
            self._send(200, b'{"unexpected":true}')
        else:
            self._send(
                200,
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "- Erster Punkt\n- Zweiter Punkt"
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode(),
            )


class Base(unittest.TestCase):
    # The server handle is genuinely per-class, not per-instance: one server
    # for the whole TestCase, started in setUpClass.
    srv: ClassVar[HTTPServer]
    port: ClassVar[int]
    url: ClassVar[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.srv = HTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.srv.server_address[1]
        cls.url = f"http://127.0.0.1:{cls.port}/v1"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.srv.shutdown()

    def setUp(self) -> None:
        MODE["v"] = "ok"
        LAST.clear()
        FORCED["body"] = None
        FORCED["tries"] = 0
        FORCED["models"] = None


def forced_choice(message: dict[str, Any], finish: str = "stop") -> None:
    FORCED["body"] = json.dumps(
        {"choices": [{"message": message, "finish_reason": finish}]}
    ).encode()


class TestNormalize(Base):
    def test_strips_headings_and_labels(self) -> None:
        messy = (
            "**Summary:**\n\n1. First point here\n• Second\n* Third\n"
            "## A heading\n- Already correct\n2) Ninth\n- Last"
        )
        out = S.normalize_bullets(messy, 12).splitlines()
        self.assertNotIn("- *Summary:**", out)
        self.assertNotIn("- A heading", out)
        self.assertIn("- First point here", out)
        self.assertIn("- Second", out)
        self.assertIn("- Already correct", out)

    def test_caps_count(self) -> None:
        many = "\n".join(f"- point {i}" for i in range(30))
        self.assertEqual(len(S.normalize_bullets(many, 12).splitlines()), 12)
        self.assertEqual(len(S.normalize_bullets(many, 3).splitlines()), 3)

    def test_empty_and_whitespace(self) -> None:
        self.assertEqual(S.normalize_bullets(""), "")
        self.assertEqual(S.normalize_bullets("\n\n   \n"), "")


# ---------------------------------------------------------------- metadata


class TestMetadataLines(unittest.TestCase):
    """The Trust line survives normalisation uncounted; KIND is stripped."""

    def test_trust_is_a_separate_markdown_block(self) -> None:
        """A blank line must precede Trust, or Markdown folds it into a bullet.

        A line that follows a list item without a blank line is a lazy
        continuation: it renders INSIDE that bullet. 'Trust: ...' then reads as
        the tail of the last fact, which is both wrong to read and wrong about
        what it is.
        """
        out = S.normalize_bullets("- [fact] a\n- [fact] b\nTrust: check it.")
        self.assertIn("\n\n**Assessment:** check it.", out, out)
        self.assertNotIn("\nTrust:", out.replace("\n\nTrust:", ""), out)

    def test_no_trailing_blank_when_there_is_no_trust_line(self) -> None:
        self.assertEqual(S.normalize_bullets("- a\n- b"), "- a\n- b")

    def test_trust_kept_outside_the_budget(self) -> None:
        raw = (
            "KIND: information\n"
            + "\n".join(f"- [claim] p{i}" for i in range(1, 16))
            + "\nTrust: solid; nothing to check."
        )
        lines = S.normalize_bullets(raw, 12).splitlines()
        self.assertEqual(lines[0], "- [claim] p1", "KIND must not survive")
        self.assertEqual(lines[-1], "**Assessment:** solid; nothing to check.")
        self.assertEqual(
            sum(1 for ln in lines if ln.startswith("- ")),
            12,
            "metadata must not eat a content slot",
        )

    def test_kind_is_captured_and_removed(self) -> None:
        """The classification is for the benchmark; the file must not show it."""
        for raw, want in (
            ("KIND: information", "A"),
            ("**KIND** - performance", "B"),
            ("- KIND: both", "BOTH"),
        ):
            kind: list[str] = []
            out = S.normalize_bullets(raw + "\n- x", kind_out=kind)
            self.assertNotIn("KIND", out, raw)
            self.assertNotIn("kind", out.lower(), raw)
            self.assertEqual(out.splitlines()[0], "- x")
            self.assertEqual(kind, [want], raw)

    def test_kind_dropped_even_without_a_collector(self) -> None:
        out = S.normalize_bullets("KIND: both\n- x")
        self.assertEqual(out, "- x")

    def test_kind_is_not_a_bullet_even_though_it_has_no_dash(self) -> None:
        for raw in ("**KIND B**", "__KIND: A__", "- KIND - BOTH"):
            out = S.normalize_bullets(raw + "\n- x")
            self.assertEqual(out.splitlines()[0], "- x", raw)

    def test_trust_phrasings_normalised(self) -> None:
        for raw in (
            "How far to trust: nothing.",
            "- Trust: nothing.",
            "**Trust** — nothing.",
        ):
            last = S.normalize_bullets("- x\n" + raw).splitlines()[-1]
            self.assertTrue(last.startswith("**Assessment:**"), last)

    def test_a_bullet_mentioning_kind_is_not_the_kind_line(self) -> None:
        out = S.normalize_bullets("- [claim] the kind of steel matters\n- x")
        self.assertEqual(out.splitlines()[0], "- [claim] the kind of steel matters")

    def test_bullet_starting_with_trust_stays_a_bullet(self) -> None:
        # "Trust" needs a separator to be the label; a content bullet that
        # merely begins with the word used to be swallowed as the Trust line.
        out = S.normalize_bullets(
            "Trust: high for the method\n"
            "- [fact] the study found X\n"
            "- Trust between the two presenters is the subject\n"
        )
        lines = out.splitlines()
        self.assertIn("- Trust between the two presenters is the subject", lines)
        self.assertEqual(lines[-1], "**Assessment:** high for the method")

    def test_old_trust_label_migrates_to_assessment(self) -> None:
        """A cached model or an older file still says "Trust:".

        The label changed because "Trust" could not carry delivery style or
        datedness. The old spelling must normalise to the new one, not sit in
        the output as a stray bullet.
        """
        for raw in ("Trust: old.", "How far to trust - old.", "Verdict: old."):
            last = S.normalize_bullets(f"- [claim] x\n{raw}").splitlines()[-1]
            self.assertTrue(last.startswith("**Assessment:**"), last)
            self.assertNotIn("Trust", last)

    def test_fact_tag_is_rewritten_to_claim(self) -> None:
        """Nothing here verifies anything, so no bullet may claim to be a fact.

        The prompt asks for [claim]; a model's habit is stronger than an
        instruction, so the tag is corrected rather than trusted.
        """
        out = S.normalize_bullets(
            "- [fact] the sky is blue\n- [FACT] water is wet\n- [opinion] a view"
        )
        self.assertNotIn("[fact]", out.lower())
        self.assertEqual(out.splitlines()[0], "- [claim] the sky is blue")
        self.assertEqual(out.splitlines()[1], "- [claim] water is wet")
        self.assertEqual(out.splitlines()[2], "- [opinion] a view")

    def test_tldr_leads_and_is_untagged(self) -> None:
        """The TL;DR is metadata: first line, no tag, outside the budget."""
        out = S.normalize_bullets(
            "- [claim] a point\nTL;DR: the whole video in one line.\n"
            "- [claim] another\nTrust: check it."
        )
        lines = [ln for ln in out.splitlines() if ln.strip()]
        self.assertEqual(lines[0], "**TL;DR:** the whole video in one line.")
        # A blank line separates the TL;DR from the bullet block, so the first
        # bullet is not line 1 of the file.
        self.assertTrue(lines[1].startswith("- [claim] a point"))
        self.assertIn("**TL;DR:** the whole video in one line.\n\n- ", out)

    def test_tldr_does_not_consume_a_bullet(self) -> None:
        bullets = "\n".join(f"- [claim] p{i}" for i in range(1, 4))
        out = S.normalize_bullets(f"TL;DR: x\n{bullets}", max_bullets=3)
        self.assertEqual(out.count("- [claim]"), 3)

    def test_tldr_spelling_variants_are_normalised(self) -> None:
        for raw in ("TLDR: x", "tl;dr - x", "**TL;DR:** x"):
            out = S.normalize_bullets(f"{raw}\n- [claim] y")
            self.assertEqual(out.splitlines()[0], "**TL;DR:** x", raw)

    def test_overview_tag_survives(self) -> None:
        """[overview] is the summarizer's own voice; it must reach the reader.

        The other three tags are the video talking. A describing line carries
        the summarizer's observation, so tagging it [claim] would attribute
        words to someone who never said them.
        """
        out = S.normalize_bullets(
            "- [overview] This is a stand-up set.\n- [claim] the routine is about X\n"
            "- [overview] The transcript is a fragment."
        )
        self.assertEqual(out.count("[overview]"), 2)
        self.assertIn("- [overview] The transcript is a fragment.", out)

    def test_fact_inside_prose_is_not_rewritten(self) -> None:
        """Only the tag is a promise; the word is ordinary English elsewhere."""
        out = S.normalize_bullets("- [claim] the fact of the matter is contested")
        self.assertIn("the fact of the matter", out)

    def test_no_metadata_is_fine(self) -> None:
        self.assertEqual(S.normalize_bullets("- a\n- b"), "- a\n- b")


class TestTrim(Base):
    def test_short_untouched(self) -> None:
        txt, trimmed = S.trim("hello", 24000)
        self.assertFalse(trimmed)
        self.assertEqual(txt, "hello")

    def test_long_keeps_head_and_tail(self) -> None:
        txt, trimmed = S.trim("A" * 2000 + "MIDDLE" * 100 + "Z" * 2000, 1000)
        self.assertTrue(trimmed)
        self.assertTrue(txt.startswith("A"))
        self.assertTrue(txt.endswith("Z"))
        self.assertIn("omitted", txt)
        self.assertNotIn("MIDDLE", txt)


class TestCall(Base):
    def test_payload_and_auth(self) -> None:
        llm.call("PROMPT", base_url=self.url, api_key="sekrit", model="m1")
        self.assertEqual(LAST["path"], "/v1/chat/completions")
        self.assertEqual(LAST["auth"], "Bearer sekrit")
        self.assertEqual(LAST["body"]["model"], "m1")
        self.assertEqual(LAST["body"]["messages"][0]["content"], "PROMPT")
        self.assertFalse(LAST["body"]["stream"])

    def test_no_auth_header_without_key(self) -> None:
        llm.call("P", base_url=self.url, model="m1")
        self.assertIsNone(LAST["auth"])

    def test_bare_host_gets_suffix(self) -> None:
        llm.call("P", base_url=f"http://127.0.0.1:{self.port}/v1", model="m1")
        self.assertEqual(LAST["path"], "/v1/chat/completions")

    def test_http_500_raises_after_retry(self) -> None:
        MODE["v"] = "500"
        with self.assertRaises(llm.SummarizeError) as cm:
            llm.call("P", base_url=self.url, model="m1")
        self.assertIn("500", str(cm.exception))

    def test_http_400_not_retried(self) -> None:
        MODE["v"] = "400"
        FORCED["tries"] = 0
        with self.assertRaises(llm.SummarizeError):
            llm.call("P", base_url=self.url, model="m1", retries=3)
        # non-transient: the server saw exactly one attempt
        self.assertEqual(FORCED["tries"], 0)

    def test_non_json_raises(self) -> None:
        MODE["v"] = "badjson"
        with self.assertRaises(llm.SummarizeError):
            llm.call("P", base_url=self.url, model="m1")

    def test_bad_shape_raises(self) -> None:
        MODE["v"] = "shape"
        with self.assertRaises(llm.SummarizeError):
            llm.call("P", base_url=self.url, model="m1")

    def test_unreachable_host_raises(self) -> None:
        with self.assertRaises(llm.SummarizeError) as cm:
            llm.call(
                "P", base_url="http://127.0.0.1:1/v1", model="m1", timeout=2, retries=1
            )
        self.assertIn("cannot reach", str(cm.exception))

    def test_default_is_a_single_attempt(self) -> None:
        """retries defaults to 0, so one 5xx ends the run immediately.

        The default used to be 2, which made --llm-timeout 300 wait 600s before
        giving up -- long enough to look hung on a phone.
        """
        MODE["v"] = "500"
        FORCED["tries"] = 0
        with self.assertRaises(llm.SummarizeError):
            llm.call("P", base_url=self.url, model="m1")
        self.assertEqual(FORCED["tries"], 0)


class TestRobustness(Base):
    def test_content_parts_shape(self) -> None:
        forced_choice(
            {
                "content": [
                    {"type": "text", "text": "- part one\n"},
                    {"type": "text", "text": "- part two"},
                ]
            }
        )
        out = S.normalize_bullets(S.call("P", base_url=self.url, model="m1"))
        self.assertEqual(out, "- part one\n- part two")

    def test_reasoning_content_fallback(self) -> None:
        forced_choice({"content": None, "reasoning_content": "- only reasoning"})
        out = S.normalize_bullets(S.call("P", base_url=self.url, model="m1"))
        self.assertEqual(out, "- only reasoning")

    def test_empty_content_raises(self) -> None:
        forced_choice({"content": "   "})
        with self.assertRaises(S.SummarizeError) as cm:
            S.call("P", base_url=self.url, model="m1")
        self.assertIn("empty assistant message", str(cm.exception))

    def test_transient_500_is_retried_then_succeeds(self) -> None:
        MODE["v"] = "fail_then_ok"
        out = S.call("P", base_url=self.url, model="m1", retries=1)
        self.assertIn("Erster Punkt", out)
        self.assertEqual(FORCED["tries"], 2)

    def test_length_finish_reason_warns(self) -> None:
        forced_choice({"content": "- cut off"}, finish="length")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            S.call("P", base_url=self.url, model="m1")
        self.assertIn("token limit", err.getvalue())


class TestThinEvidence(unittest.TestCase):
    """ "Not enough to work with" is an outcome, not a failure."""

    def test_prompt_names_thin_evidence_as_a_valid_outcome(self) -> None:
        prompt, _ = S.build_prompt("word " * 500)
        self.assertIn("THIN EVIDENCE", prompt)
        self.assertIn("is a real and common outcome, not a", prompt)

    def test_thin_output_is_not_padded_to_the_budget(self) -> None:
        # The instruction has to survive a long transcript, since budget_for
        # asks for many bullets; thin evidence must override that.
        prompt, _ = S.build_prompt("word " * 5000)
        self.assertIn("Do not pad to reach the bullet budget", prompt)

    def test_absent_and_broken_transcripts_are_distinguished(self) -> None:
        prompt, _ = S.build_prompt("word " * 500)
        self.assertIn("BROKEN transcript", prompt)
        self.assertIn("ABSENT or empty one", prompt)

    def test_tldr_only_output_is_still_valid(self) -> None:
        # A one-line summary must survive normalisation rather than being
        # rejected for having no bullets.
        out = S.normalize_bullets(
            "**TL;DR:** nothing to summarise.\n\n**Assessment:** nothing to check."
        )
        self.assertIn("**TL;DR:** nothing to summarise.", out)
        self.assertIn("**Assessment:** nothing to check.", out)


class TestBudget(unittest.TestCase):
    """Budget scales with transcript length, with a floor.

    A flat ceiling compressed a 3,816-word vlog and a 569-word making-of
    equally, which made the ratio six times worse on the short file.
    """

    def test_long_transcript_gets_more_bullets_than_short(self) -> None:
        short, _ = S.budget_for("w " * 600)
        long, _ = S.budget_for("w " * 3600)
        self.assertLess(short, long)

    def test_floor_stops_a_short_video_being_wrecked(self) -> None:
        # 454 words at a literal 10% is under two bullets.
        n, _ = S.budget_for("w " * 454)
        self.assertGreaterEqual(n, 4)

    def test_ceiling_still_applies(self) -> None:
        n, _ = S.budget_for("w " * 50_000, ceiling=12)
        self.assertEqual(n, 12)

    def test_word_target_is_in_the_prompt(self) -> None:
        p, _ = S.build_prompt("w " * 1000)
        self.assertIn("about 100 words", p)


class TestBuildPrompt(unittest.TestCase):
    """Title/description are separate labelled fields, never glued to {text}."""

    def test_metadata_in_own_fields(self) -> None:
        p, trimmed = S.build_prompt(
            "SPEECH_ONLY",
            6,
            title="My Title",
            channel="My Channel",
            description="https://example.org/paper",
        )
        self.assertFalse(trimmed)
        self.assertIn("TITLE: My Title", p)
        self.assertIn("CHANNEL: My Channel", p)
        self.assertIn("https://example.org/paper", p)
        self.assertIn("SPEECH_ONLY", p)
        # the transcript block holds the speech and nothing else
        body = p.split("<<<", 1)[1]
        self.assertIn("SPEECH_ONLY", body)
        self.assertNotIn("My Title", body)

    def test_absent_values_are_marked(self) -> None:
        p, _ = S.build_prompt("t", 12)
        self.assertIn("TITLE: (none provided)", p)
        self.assertIn("CHANNEL: (none provided)", p)
        self.assertIn("(none provided)", p)

    def test_long_description_is_truncated_and_marked(self) -> None:
        # Assert the description was cut to its own cap, not that the whole
        # prompt fits a fixed number: the prompt text is edited often, and a
        # magic total failed every time a rule was added to it.
        p, _ = S.build_prompt("t", 12, description="D" * 10000)
        self.assertIn("description truncated", p)
        self.assertLess(p.count("D" * 10), 10000 // 10)

    def test_budget_is_derived_from_length_not_copied(self) -> None:
        # --bullets is a ceiling. A short transcript is not worth 7 bullets and
        # the prompt says so, instead of asking for padding.
        short, _ = S.build_prompt("word " * 100, 7)
        long, _ = S.build_prompt("word " * 5000, 7)
        self.assertIn("At most 4 content bullets", short)
        self.assertIn("At most 7 content bullets", long)


class TestSummarize(Base):
    def test_end_to_end(self) -> None:
        bullets, trimmed = S.summarize("some transcript", base_url=self.url, model="m1")
        self.assertFalse(trimmed)
        self.assertEqual(bullets, "- Erster Punkt\n- Zweiter Punkt")

    def test_prompt_contains_budget_and_transcript(self) -> None:
        S.summarize(
            "UNIQUE_MARKER " * 5000, base_url=self.url, model="m1", max_bullets=7
        )
        sent = LAST["body"]["messages"][0]["content"]
        self.assertIn("At most 7 content bullets", sent)
        self.assertIn("UNIQUE_MARKER", sent)

    def test_metadata_reaches_the_request(self) -> None:
        S.summarize(
            "SPEECH",
            base_url=self.url,
            model="m1",
            title="T",
            channel="C",
            description="DESC_MARKER",
        )
        sent = LAST["body"]["messages"][0]["content"]
        self.assertIn("TITLE: T", sent)
        self.assertIn("CHANNEL: C", sent)
        self.assertIn("DESC_MARKER", sent)

    def test_env_config(self) -> None:
        os.environ["WATCHLESS_BASE_URL"] = self.url
        os.environ["WATCHLESS_API_KEY"] = "envkey"
        try:
            S.call("P", model="m1")  # model has no env fallback in this test
            self.assertEqual(LAST["auth"], "Bearer envkey")
        finally:
            del os.environ["WATCHLESS_BASE_URL"], os.environ["WATCHLESS_API_KEY"]

    def test_cli_key_overrides_env(self) -> None:
        os.environ["WATCHLESS_API_KEY"] = "envkey"
        try:
            S.call("P", base_url=self.url, model="m1", api_key="clikey")
            self.assertEqual(LAST["auth"], "Bearer clikey")
        finally:
            del os.environ["WATCHLESS_API_KEY"]

    def test_missing_url_raises(self) -> None:
        with self.assertRaises(S.SummarizeError) as cm:
            S.call("P")
        self.assertIn("no endpoint", str(cm.exception))


class TestModelField(Base):
    """model is required, not optional.

    It used to be omitted unless set, on the theory that a single-model
    llama-server would pick for you. Hosted endpoints reject the body outright
    -- llama.cpp itself answers 422 "Field required at 'model'" -- so the
    default was wrong on every API but the one it was written against.
    """

    def test_missing_model_is_refused_before_any_request(self) -> None:
        LAST.clear()
        for absent in (None, ""):
            with self.assertRaises(llm.SummarizeError) as cm:
                llm.call("P", base_url=self.url, model=absent)
            self.assertIn("no model", str(cm.exception))
        self.assertIsNone(LAST.get("body"), "a request went out without a model")

    def test_error_names_both_ways_to_supply_it(self) -> None:
        with self.assertRaises(llm.SummarizeError) as cm:
            llm.call("P", base_url=self.url)  # no model anywhere -> must raise
        msg = str(cm.exception)
        self.assertIn("--llm-model", msg)
        self.assertIn("WATCHLESS_MODEL", msg)

    def test_model_sent_when_given(self) -> None:
        llm.call("P", base_url=self.url, model="qwen3-8b")
        self.assertEqual(LAST["body"]["model"], "qwen3-8b")

    def test_env_model_used(self) -> None:
        os.environ["WATCHLESS_MODEL"] = "from-env"
        try:
            llm.call("P", base_url=self.url)  # no model: must come from env
            self.assertEqual(LAST["body"]["model"], "from-env")
        finally:
            del os.environ["WATCHLESS_MODEL"]

    def test_no_default_model_name_is_invented(self) -> None:
        """Regression: 'local' was hard-coded and would 404 on a real server."""
        with self.assertRaises(llm.SummarizeError):
            llm.call("P", base_url=self.url, model=None)
        self.assertIsNone(LAST.get("body"))

    def test_preflight_rejects_unknown_model(self) -> None:
        FORCED["models"] = ["qwen3-8b", "gemma-3-4b"]
        with self.assertRaises(llm.SummarizeError) as cm:
            S.summarize("text", base_url=self.url, model="local")
        msg = str(cm.exception)
        self.assertIn("'local' is not served", msg)
        self.assertIn("qwen3-8b", msg)

    def test_preflight_accepts_known_model(self) -> None:
        FORCED["models"] = ["qwen3-8b"]
        bullets, _ = S.summarize("text", base_url=self.url, model="qwen3-8b")
        self.assertTrue(bullets.startswith("- "))

    def test_preflight_skipped_when_no_models_endpoint(self) -> None:
        FORCED["models"] = None  # 404 on /models
        bullets, _ = S.summarize("text", base_url=self.url, model="anything")
        self.assertTrue(bullets.startswith("- "))


class TestProbeModels(Base):
    def test_ids_and_info_on_success(self) -> None:
        FORCED["models"] = ["a", "b"]
        ids, info = llm.probe_models(self.url)
        self.assertEqual(ids, ["a", "b"])
        self.assertTrue(info["reachable"])
        self.assertEqual(info["status"], 200)

    def test_404_is_reported_not_hidden(self) -> None:
        FORCED["models"] = None  # handler 404s /models
        ids, info = llm.probe_models(self.url)
        self.assertEqual(ids, [])
        self.assertTrue(info["reachable"], "server answered, so reachable")
        self.assertEqual(info["status"], 404)

    def test_unreachable_reports_error(self) -> None:
        ids, info = llm.probe_models("http://127.0.0.1:1/v1", timeout=2)
        self.assertEqual(ids, [])
        self.assertFalse(info["reachable"])
        self.assertIsNone(info["status"])
        self.assertIn("refused", info["error"].lower())

    def test_list_models_still_returns_empty_on_failure(self) -> None:
        self.assertEqual(llm.list_models("http://127.0.0.1:1/v1", timeout=2), [])


class TestDescribeModels(unittest.TestCase):
    """describe_models must not call a 200-with-ids a failure."""

    def test_ok_with_ids(self) -> None:
        info = {
            "status": 200,
            "reachable": True,
            "error": None,
            "body_head": '{"data":[{"id":"m"}]}',
        }
        self.assertEqual(llm.describe_models(info, ["m"]), "ok, 1 model(s)")

    def test_http_404(self) -> None:
        info = {"status": 404, "reachable": True, "error": None, "body_head": "{}"}
        self.assertIn("HTTP 404", llm.describe_models(info, []))

    def test_connection_error(self) -> None:
        info = {
            "status": None,
            "reachable": False,
            "error": "URLError: refused",
            "body_head": None,
        }
        self.assertIn("refused", llm.describe_models(info, []))

    def test_no_status_no_error(self) -> None:
        self.assertEqual(llm.describe_models({}, []), "no HTTP response")


class TestConnectionHint(unittest.TestCase):
    def test_refused_mentions_binding(self) -> None:
        h = llm.connection_hint("cannot connect: [Errno 111] Connection refused")
        self.assertIn("0.0.0.0", h)

    def test_timeout(self) -> None:
        self.assertIn("did not answer", llm.connection_hint("timed out after 5s"))

    def test_auth(self) -> None:
        self.assertIn("API_KEY", llm.connection_hint("HTTP 401 Unauthorized: nope"))

    def test_unknown(self) -> None:
        self.assertEqual(llm.connection_hint("something else"), "")


class TestChatProbe(Base):
    def test_ok_against_mock(self) -> None:
        ok, detail = llm.chat_probe(self.url, model="m1")
        self.assertTrue(ok, detail)
        self.assertIn("OK", detail)

    def test_refused(self) -> None:
        ok, detail = llm.chat_probe("http://127.0.0.1:1/v1", model="m1", timeout=2)
        self.assertFalse(ok)
        self.assertIn("cannot connect", detail)

    def test_500(self) -> None:
        MODE["v"] = "500"
        ok, detail = llm.chat_probe(self.url, model="m1", timeout=10)
        self.assertFalse(ok)
        self.assertIn("HTTP 500", detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPromptAsData(unittest.TestCase):
    """The prompt is a .md file, and that has two failure modes worth testing."""

    def test_prompt_loads_and_is_not_empty(self) -> None:
        # A wheel that forgot package-data ships fine and fails at the first
        # summarize. This test is the tripwire for that.
        self.assertGreater(len(S.PROMPT), 5000)
        self.assertIn("PART 2", S.PROMPT)

    def test_every_placeholder_has_a_keyword(self) -> None:
        """A typo'd or renamed placeholder must fail here, not at runtime.

        PROMPT.format(**...) raises KeyError at request time, which for a user
        means a broken summary rather than a visibly wrong edit.
        """
        named = set(re.findall(r"\{([a-z_]+)\}", S.PROMPT))
        supplied = {"n", "target", "title", "channel", "description", "text"}
        self.assertEqual(named - supplied, set(), "placeholder with no keyword")
        self.assertEqual(supplied - named, set(), "keyword never placeholdered")
        # And it really formats, with the documented keys.
        S.build_prompt("word " * 100, 8, 4000, title="t", channel="c", description="d")

    def test_bench_classification_is_not_in_the_shipped_prompt(self) -> None:
        """A reader never needs the model to label its own work.

        The KIND instruction exists so the corpus bench can score a video
        against an expectation. Shipped, it costs tokens and invites the model
        to explain itself.
        """
        self.assertNotIn("KIND", S.PROMPT)
        self.assertIn("KIND", S.KIND_PROMPT)

    def test_classify_flag_appends_the_bench_instruction(self) -> None:
        plain, _ = S.build_prompt("word " * 100)
        scored, _ = S.build_prompt("word " * 100, classify=True)
        self.assertNotIn("KIND", plain)
        self.assertIn("KIND", scored)
        self.assertTrue(scored.startswith(plain.rstrip()))

    def test_shipped_prompt_does_not_name_a_superseded_label(self) -> None:
        # The opening paragraph said 'a final "Trust:" line' for several commits
        # after the label became Assessment, because prose in a docstring is not
        # read by anything.
        self.assertNotIn('"Trust:"', S.PROMPT)
        self.assertIn("Assessment", S.PROMPT)
