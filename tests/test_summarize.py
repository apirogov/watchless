"""Tests for the summarize module against a local mock OpenAI-compatible
server."""

import contextlib
import io
import json
import os
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest import mock

import pytest

from watchless import llm
from watchless import settings as St
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


@pytest.fixture(scope="session")
def mock_url() -> Iterator[str]:
    """One HTTP server for the whole session: it is stateless but costs a port."""
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


@pytest.fixture(autouse=True)
def reset_mock() -> None:
    """Each test starts from the same canned state."""
    MODE["v"] = "ok"
    LAST.clear()
    FORCED["body"] = None
    FORCED["tries"] = 0
    FORCED["models"] = None


def forced_choice(message: dict[str, Any], finish: str = "stop") -> None:
    FORCED["body"] = json.dumps(
        {"choices": [{"message": message, "finish_reason": finish}]}
    ).encode()


class TestNormalize:
    def test_strips_headings_and_labels(self) -> None:
        messy = (
            "**Summary:**\n\n1. First point here\n• Second\n* Third\n"
            "## A heading\n- Already correct\n2) Ninth\n- Last"
        )
        out = S.normalize_bullets(messy, 12).splitlines()
        assert "- *Summary:**" not in out
        assert "- A heading" not in out
        assert "- First point here" in out
        assert "- Second" in out
        assert "- Already correct" in out

    def test_caps_count(self) -> None:
        many = "\n".join(f"- point {i}" for i in range(30))
        assert len(S.normalize_bullets(many, 12).splitlines()) == 12
        assert len(S.normalize_bullets(many, 3).splitlines()) == 3

    def test_empty_and_whitespace(self) -> None:
        assert S.normalize_bullets("") == ""
        assert S.normalize_bullets("\n\n   \n") == ""


# ---------------------------------------------------------------- metadata


class TestMetadataLines:
    """The Trust line survives normalisation uncounted; KIND is stripped."""

    def test_trust_is_a_separate_markdown_block(self) -> None:
        """A blank line must precede Trust, or Markdown folds it into a bullet.

        A line that follows a list item without a blank line is a lazy
        continuation: it renders INSIDE that bullet. 'Trust: ...' then reads as
        the tail of the last fact, which is both wrong to read and wrong about
        what it is.
        """
        out = S.normalize_bullets("- [fact] a\n- [fact] b\nTrust: check it.")
        assert "\n\n**Assessment:** check it." in out, out
        assert "\nTrust:" not in out.replace("\n\nTrust:", ""), out

    def test_no_trailing_blank_when_there_is_no_trust_line(self) -> None:
        assert S.normalize_bullets("- a\n- b") == "- a\n- b"

    def test_trust_kept_outside_the_budget(self) -> None:
        raw = (
            "KIND: information\n"
            + "\n".join(f"- [claim] p{i}" for i in range(1, 16))
            + "\nTrust: solid; nothing to check."
        )
        lines = S.normalize_bullets(raw, 12).splitlines()
        assert lines[0] == "- [claim] p1", "KIND must not survive"
        assert lines[-1] == "**Assessment:** solid; nothing to check."
        assert sum(1 for ln in lines if ln.startswith("- ")) == 12, (
            "metadata must not eat a content slot"
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
            assert "KIND" not in out, raw
            assert "kind" not in out.lower(), raw
            assert out.splitlines()[0] == "- x"
            assert kind == [want], raw

    def test_kind_dropped_even_without_a_collector(self) -> None:
        out = S.normalize_bullets("KIND: both\n- x")
        assert out == "- x"

    def test_kind_is_not_a_bullet_even_though_it_has_no_dash(self) -> None:
        for raw in ("**KIND B**", "__KIND: A__", "- KIND - BOTH"):
            out = S.normalize_bullets(raw + "\n- x")
            assert out.splitlines()[0] == "- x", raw

    def test_trust_phrasings_normalised(self) -> None:
        for raw in (
            "How far to trust: nothing.",
            "- Trust: nothing.",
            "**Trust** — nothing.",
        ):
            last = S.normalize_bullets("- x\n" + raw).splitlines()[-1]
            assert last.startswith("**Assessment:**"), last

    def test_a_bullet_mentioning_kind_is_not_the_kind_line(self) -> None:
        out = S.normalize_bullets("- [claim] the kind of steel matters\n- x")
        assert out.splitlines()[0] == "- [claim] the kind of steel matters"

    def test_bullet_starting_with_trust_stays_a_bullet(self) -> None:
        # "Trust" needs a separator to be the label; a content bullet that
        # merely begins with the word used to be swallowed as the Trust line.
        out = S.normalize_bullets(
            "Trust: high for the method\n"
            "- [fact] the study found X\n"
            "- Trust between the two presenters is the subject\n"
        )
        lines = out.splitlines()
        assert "- Trust between the two presenters is the subject" in lines
        assert lines[-1] == "**Assessment:** high for the method"

    def test_old_trust_label_migrates_to_assessment(self) -> None:
        """A cached model or an older file still says "Trust:".

        The label changed because "Trust" could not carry delivery style or
        datedness. The old spelling must normalise to the new one, not sit in
        the output as a stray bullet.
        """
        for raw in ("Trust: old.", "How far to trust - old.", "Verdict: old."):
            last = S.normalize_bullets(f"- [claim] x\n{raw}").splitlines()[-1]
            assert last.startswith("**Assessment:**"), last
            assert "Trust" not in last

    def test_fact_tag_is_rewritten_to_claim(self) -> None:
        """Nothing here verifies anything, so no bullet may claim to be a fact.

        The prompt asks for [claim]; a model's habit is stronger than an
        instruction, so the tag is corrected rather than trusted.
        """
        out = S.normalize_bullets(
            "- [fact] the sky is blue\n- [FACT] water is wet\n- [opinion] a view"
        )
        assert "[fact]" not in out.lower()
        assert out.splitlines()[0] == "- [claim] the sky is blue"
        assert out.splitlines()[1] == "- [claim] water is wet"
        assert out.splitlines()[2] == "- [opinion] a view"

    def test_tldr_leads_and_is_untagged(self) -> None:
        """The TL;DR is metadata: first line, no tag, outside the budget."""
        out = S.normalize_bullets(
            "- [claim] a point\nTL;DR: the whole video in one line.\n"
            "- [claim] another\nTrust: check it."
        )
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert lines[0] == "**TL;DR:** the whole video in one line."
        # A blank line separates the TL;DR from the bullet block, so the first
        # bullet is not line 1 of the file.
        assert lines[1].startswith("- [claim] a point")
        assert "**TL;DR:** the whole video in one line.\n\n- " in out

    def test_tldr_does_not_consume_a_bullet(self) -> None:
        bullets = "\n".join(f"- [claim] p{i}" for i in range(1, 4))
        out = S.normalize_bullets(f"TL;DR: x\n{bullets}", max_bullets=3)
        assert out.count("- [claim]") == 3

    def test_tldr_spelling_variants_are_normalised(self) -> None:
        for raw in ("TLDR: x", "tl;dr - x", "**TL;DR:** x"):
            out = S.normalize_bullets(f"{raw}\n- [claim] y")
            assert out.splitlines()[0] == "**TL;DR:** x", raw

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
        assert out.count("[overview]") == 2
        assert "- [overview] The transcript is a fragment." in out

    def test_fact_inside_prose_is_not_rewritten(self) -> None:
        """Only the tag is a promise; the word is ordinary English elsewhere."""
        out = S.normalize_bullets("- [claim] the fact of the matter is contested")
        assert "the fact of the matter" in out

    def test_no_metadata_is_fine(self) -> None:
        assert S.normalize_bullets("- a\n- b") == "- a\n- b"


class TestBudgetChars:
    """The transcript budget comes from the endpoint, not a fixed constant."""

    def test_half_the_context_window(self) -> None:
        # 1,048,560-token window -> ~524k tokens -> 4 chars each
        assert S.budget_chars(1_048_560) == 524_280 * 4

    def test_unknown_context_falls_back_to_the_constant(self) -> None:
        assert S.budget_chars(None) == St.DEFAULT_MAX_CHARS

    def test_zero_context_is_treated_as_unknown(self) -> None:
        assert S.budget_chars(0) == St.DEFAULT_MAX_CHARS


class TestRefusesRatherThanTruncates:
    """An over-budget transcript is refused before anything is sent.

    Cutting a hole in the middle made the model write about a document it had
    not been shown, and report the hole as a defect in the video's captions.
    """

    def test_over_budget_raises(self) -> None:
        with pytest.raises(S.TranscriptTooLong):
            S.build_prompt("x" * 200_000, 12, budget=24_000)

    def test_nothing_is_sent_when_it_does_not_fit(self, mock_url: str) -> None:
        sent: list[object] = []
        original = llm.call
        llm.call = lambda *a, **k: sent.append(a) or "unreachable"  # type: ignore[assignment]
        try:
            with pytest.raises(S.TranscriptTooLong):
                S.summarize("x" * 5_000_000, base_url=mock_url, model="m1")
        finally:
            llm.call = original  # type: ignore[assignment]
        assert sent == []

    def test_error_says_nothing_was_sent(self) -> None:
        with pytest.raises(S.TranscriptTooLong) as cm:
            S.build_prompt("x" * 200_000, 12, budget=24_000)
        assert "Nothing was sent" in str(cm.value)

    def test_within_budget_is_untouched(self) -> None:
        text = "w " * 1000
        prompt = S.build_prompt(text, 12, budget=24_000)
        assert text.strip() in prompt


class TestCall:
    def test_payload_and_auth(self, mock_url: str) -> None:
        llm.call("PROMPT", base_url=mock_url, api_key="sekrit", model="m1")
        assert LAST["path"] == "/v1/chat/completions"
        assert LAST["auth"] == "Bearer sekrit"
        assert LAST["body"]["model"] == "m1"
        assert LAST["body"]["messages"][0]["content"] == "PROMPT"
        assert not LAST["body"]["stream"]

    def test_no_auth_header_without_key(self, mock_url: str) -> None:
        llm.call("P", base_url=mock_url, model="m1")
        assert LAST["auth"] is None

    def test_bare_host_gets_suffix(self, mock_url: str) -> None:
        llm.call("P", base_url=mock_url, model="m1")
        assert LAST["path"] == "/v1/chat/completions"

    def test_http_500_raises_after_retry(self, mock_url: str) -> None:
        MODE["v"] = "500"
        with pytest.raises(llm.SummarizeError) as cm:
            llm.call("P", base_url=mock_url, model="m1")
        assert "500" in str(cm.value)

    def test_http_400_not_retried(self, mock_url: str) -> None:
        MODE["v"] = "400"
        FORCED["tries"] = 0
        with pytest.raises(llm.SummarizeError):
            llm.call("P", base_url=mock_url, model="m1", retries=3)
        # non-transient: the server saw exactly one attempt
        assert FORCED["tries"] == 0

    def test_non_json_raises(self, mock_url: str) -> None:
        MODE["v"] = "badjson"
        with pytest.raises(llm.SummarizeError):
            llm.call("P", base_url=mock_url, model="m1")

    def test_bad_shape_raises(self, mock_url: str) -> None:
        MODE["v"] = "shape"
        with pytest.raises(llm.SummarizeError):
            llm.call("P", base_url=mock_url, model="m1")

    def test_unreachable_host_raises(self) -> None:
        with pytest.raises(llm.SummarizeError) as cm:
            llm.call(
                "P", base_url="http://127.0.0.1:1/v1", model="m1", timeout=2, retries=1
            )
        assert "cannot reach" in str(cm.value)

    def test_default_is_a_single_attempt(self, mock_url: str) -> None:
        """retries defaults to 0, so one 5xx ends the run immediately.

        The default used to be 2, which made --llm-timeout 300 wait 600s before
        giving up -- long enough to look hung on a phone.
        """
        MODE["v"] = "500"
        FORCED["tries"] = 0
        with pytest.raises(llm.SummarizeError):
            llm.call("P", base_url=mock_url, model="m1")
        assert FORCED["tries"] == 0


class TestRobustness:
    def test_content_parts_shape(self, mock_url: str) -> None:
        forced_choice(
            {
                "content": [
                    {"type": "text", "text": "- part one\n"},
                    {"type": "text", "text": "- part two"},
                ]
            }
        )
        out = S.normalize_bullets(S.call("P", base_url=mock_url, model="m1"))
        assert out == "- part one\n- part two"

    def test_reasoning_content_fallback(self, mock_url: str) -> None:
        forced_choice({"content": None, "reasoning_content": "- only reasoning"})
        out = S.normalize_bullets(S.call("P", base_url=mock_url, model="m1"))
        assert out == "- only reasoning"

    def test_reasoning_fallback(self, mock_url: str) -> None:
        """Some servers name the field ``reasoning``, not ``reasoning_content``.

        A real endpoint returns content: null with the thinking under
        ``reasoning``. Reading only ``reasoning_content`` turned that into
        "empty assistant message" and threw away the only explanation.
        """
        forced_choice({"content": None, "reasoning": "- thinking only"})
        out = S.normalize_bullets(S.call("P", base_url=mock_url, model="m1"))
        assert out == "- thinking only"

    def test_empty_content_raises(self, mock_url: str) -> None:
        forced_choice({"content": "   "})
        with pytest.raises(S.SummarizeError) as cm:
            S.call("P", base_url=mock_url, model="m1")
        assert "empty assistant message" in str(cm.value)

    def test_transient_500_is_retried_then_succeeds(self, mock_url: str) -> None:
        MODE["v"] = "fail_then_ok"
        out = S.call("P", base_url=mock_url, model="m1", retries=1)
        assert "Erster Punkt" in out
        assert FORCED["tries"] == 2

    def test_length_finish_reason_warns(self, mock_url: str) -> None:
        forced_choice({"content": "- cut off"}, finish="length")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            S.call("P", base_url=mock_url, model="m1")
        assert "token limit" in err.getvalue()


class TestThinEvidence:
    """ "Not enough to work with" is an outcome, not a failure."""

    def test_prompt_names_thin_evidence_as_a_valid_outcome(self) -> None:
        prompt = S.build_prompt("word " * 500)
        assert "THIN EVIDENCE" in prompt
        assert "is a real and common outcome, not a" in prompt

    def test_thin_output_is_not_padded_to_the_budget(self) -> None:
        # The instruction has to survive a long transcript, since budget_for
        # asks for many bullets; thin evidence must override that.
        prompt = S.build_prompt("word " * 5000, budget=10**6)
        assert "Do not pad to reach the bullet budget" in prompt

    def test_absent_and_broken_transcripts_are_distinguished(self) -> None:
        prompt = S.build_prompt("word " * 500)
        assert "BROKEN transcript" in prompt
        assert "ABSENT or empty one" in prompt

    def test_tldr_only_output_is_still_valid(self) -> None:
        # A one-line summary must survive normalisation rather than being
        # rejected for having no bullets.
        out = S.normalize_bullets(
            "**TL;DR:** nothing to summarise.\n\n**Assessment:** nothing to check."
        )
        assert "**TL;DR:** nothing to summarise." in out
        assert "**Assessment:** nothing to check." in out


class TestBudget:
    """Budget scales with transcript length, with a floor.

    A flat ceiling compressed a 3,816-word vlog and a 569-word making-of
    equally, which made the ratio six times worse on the short file.
    """

    def test_long_transcript_gets_more_bullets_than_short(self) -> None:
        short, _ = S.budget_for("w " * 600)
        long, _ = S.budget_for("w " * 3600)
        assert short < long

    def test_floor_stops_a_short_video_being_wrecked(self) -> None:
        # 454 words at a literal 10% is under two bullets.
        n, _ = S.budget_for("w " * 454)
        assert n >= 4

    def test_ceiling_still_applies(self) -> None:
        n, _ = S.budget_for("w " * 50_000, ceiling=12)
        assert n == 12

    def test_word_cap_scales_with_bullets_not_transcript_length(self) -> None:
        """12 bullets is a fixed ask, so it gets a fixed word budget.

        A 26k-word transcript used to ask for 2,678 words -- an essay about a
        document whose middle had been trimmed away. It now asks for
        12 x SUMMARY_MAX_WORDS_PER_BULLET.
        """
        _, cap_26k = S.budget_for("w " * 26_788)
        _, cap_10k = S.budget_for("w " * 10_000)
        _, cap_80k = S.budget_for("w " * 80_000)
        assert cap_26k == 12 * St.SUMMARY_MAX_WORDS_PER_BULLET
        assert cap_10k == cap_26k
        assert cap_80k == cap_26k

    def test_word_cap_follows_a_reduced_bullet_count(self) -> None:
        n, cap = S.budget_for("w " * 26_788, ceiling=6)
        assert n == 6
        assert cap == 6 * St.SUMMARY_MAX_WORDS_PER_BULLET

    def test_short_transcript_budget_is_unchanged_by_the_cap(self) -> None:
        """The cap bites only where the ratio exceeds it; a short video keeps
        the word target it already had."""
        _, cap = S.budget_for("w " * 3_600)
        assert cap == 360
        assert cap < 12 * St.SUMMARY_MAX_WORDS_PER_BULLET

    def test_word_cap_exceeds_a_typical_bullet(self) -> None:
        """Median corpus bullet is ~30 words, p90 38. The cap has to leave room
        for a long one or it would force clipped bullets."""
        assert St.SUMMARY_MAX_WORDS_PER_BULLET >= 38

    def test_word_target_is_in_the_prompt(self) -> None:
        p = S.build_prompt("w " * 1000)
        assert "about 100 words" in p


class TestBuildPrompt:
    """Title/description are separate labelled fields, never glued to {text}."""

    def test_metadata_in_own_fields(self) -> None:
        p = S.build_prompt(
            "SPEECH_ONLY",
            6,
            title="My Title",
            channel="My Channel",
            description="https://example.org/paper",
        )
        assert "TITLE: My Title" in p
        assert "CHANNEL: My Channel" in p
        assert "https://example.org/paper" in p
        assert "SPEECH_ONLY" in p
        # the transcript block holds the speech and nothing else
        body = p.split("<<<", 1)[1]
        assert "SPEECH_ONLY" in body
        assert "My Title" not in body

    def test_absent_values_are_marked(self) -> None:
        p = S.build_prompt("t", 12)
        assert "TITLE: (none provided)" in p
        assert "CHANNEL: (none provided)" in p
        assert "(none provided)" in p

    def test_long_description_is_truncated_and_marked(self) -> None:
        # Assert the description was cut to its own cap, not that the whole
        # prompt fits a fixed number: the prompt text is edited often, and a
        # magic total failed every time a rule was added to it.
        p = S.build_prompt("t", 12, description="D" * 10000)
        assert "description truncated" in p
        assert p.count("D" * 10) < 10000 // 10

    def test_budget_is_derived_from_length_not_copied(self) -> None:
        # --bullets is a ceiling. A short transcript is not worth 7 bullets and
        # the prompt says so, instead of asking for padding.
        short = S.build_prompt("word " * 100, 7)
        long = S.build_prompt("word " * 5000, 7, 10**6)
        assert "At most 4 content bullets" in short
        assert "At most 7 content bullets" in long


class TestSummarize:
    def test_end_to_end(self, mock_url: str) -> None:
        bullets = S.summarize("some transcript", base_url=mock_url, model="m1")
        assert bullets == "- Erster Punkt\n- Zweiter Punkt"

    def test_prompt_contains_budget_and_transcript(self, mock_url: str) -> None:
        # A wide context window, as a real endpoint advertises, so the whole
        # transcript is sent rather than refused.
        with mock.patch.object(S.llm, "context_limit", return_value=1_000_000):
            S.summarize(
                "UNIQUE_MARKER " * 5000, base_url=mock_url, model="m1", max_bullets=7
            )
        sent = LAST["body"]["messages"][0]["content"]
        assert "At most 7 content bullets" in sent
        assert "UNIQUE_MARKER" in sent

    def test_metadata_reaches_the_request(self, mock_url: str) -> None:
        S.summarize(
            "SPEECH",
            base_url=mock_url,
            model="m1",
            title="T",
            channel="C",
            description="DESC_MARKER",
        )
        sent = LAST["body"]["messages"][0]["content"]
        assert "TITLE: T" in sent
        assert "CHANNEL: C" in sent
        assert "DESC_MARKER" in sent

    def test_env_config(self, mock_url: str) -> None:
        os.environ["WATCHLESS_BASE_URL"] = mock_url
        os.environ["WATCHLESS_API_KEY"] = "envkey"
        try:
            S.call("P", model="m1")  # model has no env fallback in this test
            assert LAST["auth"] == "Bearer envkey"
        finally:
            del os.environ["WATCHLESS_BASE_URL"], os.environ["WATCHLESS_API_KEY"]

    def test_cli_key_overrides_env(self, mock_url: str) -> None:
        os.environ["WATCHLESS_API_KEY"] = "envkey"
        try:
            S.call("P", base_url=mock_url, model="m1", api_key="clikey")
            assert LAST["auth"] == "Bearer clikey"
        finally:
            del os.environ["WATCHLESS_API_KEY"]

    def test_missing_url_raises(self) -> None:
        with pytest.raises(S.SummarizeError) as cm:
            S.call("P")
        assert "no endpoint" in str(cm.value)


class TestModelField:
    """model is required, not optional.

    It used to be omitted unless set, on the theory that a single-model
    llama-server would pick for you. Hosted endpoints reject the body outright
    -- llama.cpp itself answers 422 "Field required at 'model'" -- so the
    default was wrong on every API but the one it was written against.
    """

    def test_missing_model_is_refused_before_any_request(self, mock_url: str) -> None:
        LAST.clear()
        for absent in (None, ""):
            with pytest.raises(llm.SummarizeError) as cm:
                llm.call("P", base_url=mock_url, model=absent)
            assert "no model" in str(cm.value)
        assert LAST.get("body") is None, "a request went out without a model"

    def test_error_names_both_ways_to_supply_it(self, mock_url: str) -> None:
        with pytest.raises(llm.SummarizeError) as cm:
            llm.call("P", base_url=mock_url)  # no model anywhere -> must raise
        msg = str(cm.value)
        assert "--llm-model" in msg
        assert "WATCHLESS_MODEL" in msg

    def test_model_sent_when_given(self, mock_url: str) -> None:
        llm.call("P", base_url=mock_url, model="qwen3-8b")
        assert LAST["body"]["model"] == "qwen3-8b"

    def test_env_model_used(self, mock_url: str) -> None:
        os.environ["WATCHLESS_MODEL"] = "from-env"
        try:
            llm.call("P", base_url=mock_url)  # no model: must come from env
            assert LAST["body"]["model"] == "from-env"
        finally:
            del os.environ["WATCHLESS_MODEL"]

    def test_no_default_model_name_is_invented(self, mock_url: str) -> None:
        """Regression: 'local' was hard-coded and would 404 on a real server."""
        with pytest.raises(llm.SummarizeError):
            llm.call("P", base_url=mock_url, model=None)
        assert LAST.get("body") is None

    def test_preflight_rejects_unknown_model(self, mock_url: str) -> None:
        FORCED["models"] = ["qwen3-8b", "gemma-3-4b"]
        with pytest.raises(llm.SummarizeError) as cm:
            S.summarize("text", base_url=mock_url, model="local")
        msg = str(cm.value)
        assert "'local' is not served" in msg
        assert "qwen3-8b" in msg

    def test_preflight_accepts_known_model(self, mock_url: str) -> None:
        FORCED["models"] = ["qwen3-8b"]
        bullets = S.summarize("text", base_url=mock_url, model="qwen3-8b")
        assert bullets.startswith("- ")

    def test_preflight_skipped_when_no_models_endpoint(self, mock_url: str) -> None:
        FORCED["models"] = None  # 404 on /models
        bullets = S.summarize("text", base_url=mock_url, model="anything")
        assert bullets.startswith("- ")


class TestProbeModels:
    def test_ids_and_info_on_success(self, mock_url: str) -> None:
        FORCED["models"] = ["a", "b"]
        ids, info = llm.probe_models(mock_url)
        assert ids == ["a", "b"]
        assert info["reachable"]
        assert info["status"] == 200

    def test_404_is_reported_not_hidden(self, mock_url: str) -> None:
        FORCED["models"] = None  # handler 404s /models
        ids, info = llm.probe_models(mock_url)
        assert ids == []
        assert info["reachable"], "server answered, so reachable"
        assert info["status"] == 404

    def test_unreachable_reports_error(self) -> None:
        ids, info = llm.probe_models("http://127.0.0.1:1/v1", timeout=2)
        assert ids == []
        assert not info["reachable"]
        assert info["status"] is None
        assert "refused" in info["error"].lower()

    def test_list_models_still_returns_empty_on_failure(self) -> None:
        assert llm.list_models("http://127.0.0.1:1/v1", timeout=2) == []


class TestDescribeModels:
    """describe_models must not call a 200-with-ids a failure."""

    def test_ok_with_ids(self) -> None:
        info = {
            "status": 200,
            "reachable": True,
            "error": None,
            "body_head": '{"data":[{"id":"m"}]}',
        }
        assert llm.describe_models(info, ["m"]) == "ok, 1 model(s)"

    def test_http_404(self) -> None:
        info = {"status": 404, "reachable": True, "error": None, "body_head": "{}"}
        assert "HTTP 404" in llm.describe_models(info, [])

    def test_connection_error(self) -> None:
        info = {
            "status": None,
            "reachable": False,
            "error": "URLError: refused",
            "body_head": None,
        }
        assert "refused" in llm.describe_models(info, [])

    def test_no_status_no_error(self) -> None:
        assert llm.describe_models({}, []) == "no HTTP response"


class TestConnectionHint:
    def test_refused_mentions_binding(self) -> None:
        h = llm.connection_hint("cannot connect: [Errno 111] Connection refused")
        assert "0.0.0.0" in h

    def test_timeout(self) -> None:
        assert "did not answer" in llm.connection_hint("timed out after 5s")

    def test_auth(self) -> None:
        assert "API_KEY" in llm.connection_hint("HTTP 401 Unauthorized: nope")

    def test_unknown(self) -> None:
        assert llm.connection_hint("something else") == ""


class TestChatProbe:
    def test_ok_against_mock(self, mock_url: str) -> None:
        ok, detail = llm.chat_probe(mock_url, model="m1")
        assert ok, detail
        assert "OK" in detail

    def test_refused(self) -> None:
        ok, detail = llm.chat_probe("http://127.0.0.1:1/v1", model="m1", timeout=2)
        assert not ok
        assert "cannot connect" in detail

    def test_500(self, mock_url: str) -> None:
        MODE["v"] = "500"
        ok, detail = llm.chat_probe(mock_url, model="m1", timeout=10)
        assert not ok
        assert "HTTP 500" in detail


class TestPromptAsData:
    """The prompt is a .md file, and that has two failure modes worth testing."""

    def test_prompt_loads_and_is_not_empty(self) -> None:
        # A wheel that forgot package-data ships fine and fails at the first
        # summarize. This test is the tripwire for that.
        assert len(S.PROMPT) > 5000
        assert "PART 2" in S.PROMPT

    def test_every_placeholder_has_a_keyword(self) -> None:
        """A typo'd or renamed placeholder must fail here, not at runtime.

        PROMPT.format(**...) raises KeyError at request time, which for a user
        means a broken summary rather than a visibly wrong edit.
        """
        named = set(re.findall(r"\{([a-z_]+)\}", S.PROMPT))
        supplied = {"n", "target", "title", "channel", "description", "text"}
        assert named - supplied == set(), "placeholder with no keyword"
        assert supplied - named == set(), "keyword never placeholdered"
        # And it really formats, with the documented keys.
        S.build_prompt("word " * 100, 8, 10**6, title="t", channel="c", description="d")

    def test_bench_classification_is_not_in_the_shipped_prompt(self) -> None:
        """A reader never needs the model to label its own work.

        The KIND instruction exists so the corpus bench can score a video
        against an expectation. Shipped, it costs tokens and invites the model
        to explain itself.
        """
        assert "KIND" not in S.PROMPT
        assert "KIND" in S.KIND_PROMPT

    def test_classify_flag_appends_the_bench_instruction(self) -> None:
        plain = S.build_prompt("word " * 100)
        scored = S.build_prompt("word " * 100, classify=True)
        assert "KIND" not in plain
        assert "KIND" in scored
        assert scored.startswith(plain.rstrip())

    def test_shipped_prompt_does_not_name_a_superseded_label(self) -> None:
        # The opening paragraph said 'a final "Trust:" line' for several commits
        # after the label became Assessment, because prose in a docstring is not
        # read by anything.
        assert '"Trust:"' not in S.PROMPT
        assert "Assessment" in S.PROMPT
