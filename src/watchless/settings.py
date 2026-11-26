"""Every environment variable and config default, in one place.

One precedence rule -- explicit flag > environment > default -- resolved here
rather than at each call site, so the value that is validated is the value that
gets used.

Nothing here reads ``sys.argv``: the CLI resolves flags and passes values in.
The env fallback exists for standalone/library use.
"""

from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------------ defaults

# Declared once. Everything that needs a default imports it from here.
TIMEOUT_S = 300  # local models on weak hardware are slow
RETRIES = 0  # attempts beyond the first; the timeout applies to each
DEFAULT_MAX_CHARS = 24_000  # ~6k tokens; trimmed from the middle if longer
DEFAULT_MAX_BULLETS = 12
# Budget shaping. A fixed bullet ceiling compresses a 3,800-word transcript far
# harder than a 570-word one: measured over the corpus, five files sat at the
# 12-bullet cap while their transcripts differed 6x in length (13 words per
# bullet vs 318). The budget scales with the transcript instead, and the floor
# stops a short video being summarised into uselessness -- at floor 4 the
# worst-case short-file ratio is 27%, at 6 it is 41%, which is worse than the
# uncapped behaviour it replaces.
SUMMARY_TARGET_RATIO = 0.10  # of transcript words
SUMMARY_WORDS_PER_BULLET = 31  # measured median over 122 corpus bullets
SUMMARY_FLOOR_BULLETS = 4
# Distinct from DEFAULT_MAX_BULLETS: the prompt's --bullets is the count to ask
# for; this is the ceiling before working memory is considered.
MAX_BULLETS = DEFAULT_MAX_BULLETS
DESC_MAX_CHARS = 4_000  # descriptions are mostly boilerplate; head-biased
DEFAULT_MAX_SLUG = 48

# Paragraph shaping for the rendered transcript. MAX_UNITS counts segments,
# not sentences: auto-generated tracks are often unpunctuated, so a segment is
# the only unit that always exists to cap a paragraph with.
GAP_S = 1.2
MAX_UNITS = 4
# Ceiling for one unit when the track has no sentence ends to cut on. Generous
# on purpose: it must sit above a normal spoken sentence (20-30 words) so a
# punctuated track is never cut mid-sentence by it.
MAX_WORDS_PER_UNIT = 30

# Substrings that mean "YouTube is rate-limiting or blocking this IP", matched
# case-insensitively against a backend's error text. Both backends use
# different class names for the same condition, and neither is a stable API.
BLOCK_SIGNS = (
    "429",
    "too many requests",
    "blocking this ip",
    "blocking requests from your ip",
    "ipblocked",
    "requestblocked",
    "sign in to confirm",
)


def looks_blocked(text: str | None) -> bool:
    """True when ``text`` carries one of :data:`BLOCK_SIGNS`.

    The one matcher for both callers: :func:`watchless.fetch._classify` classifies
    a library exception, and the corpus tool has only a log line to inspect.
    They were two separate lists before -- identical in content, so a new
    signature would have had to be added in two places to take effect.
    """
    low = (text or "").lower()
    return any(sign in low for sign in BLOCK_SIGNS)


# ------------------------------------------------------------------ precedence


def pick(arg: str | None, env_name: str, default: str | None = None) -> str | None:
    """The one precedence rule used for every env-backed string setting.

    ``arg`` not None  -> use it verbatim (an explicit empty string means
                         "empty", NOT "fall back to the env").
    ``arg`` is None   -> env, if set and non-empty.
    otherwise         -> ``default``.
    """
    if arg is not None:
        return arg if arg != "" else default
    env = os.environ.get(env_name)
    if env is not None and env != "":
        return env
    return default


def env_flag(name: str) -> bool:
    """True if an env var is set to a truthy spelling. Empty/unset -> False."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def env_flag_set_false(name: str) -> bool:
    """True only if the variable is explicitly falsey ("0", "false", "no", "off").

    For features that default to ON. Unset must mean "leave the default alone",
    which is why this is not `not env_flag(...)` -- that would read unset as
    falsey and silently turn the feature off.
    """
    return os.environ.get(name, "").strip().lower() in ("0", "false", "no", "off")


# ------------------------------------------------------------------ LLM config


def base_url(arg: str | None = None) -> str:
    return pick(arg, "WATCHLESS_BASE_URL", "") or ""


def api_key(arg: str | None = None) -> str | None:
    return pick(arg, "WATCHLESS_API_KEY", None)


def model(arg: str | None = None) -> str | None:
    return pick(arg, "WATCHLESS_MODEL", None)


def reasoning_effort() -> str:
    """``WATCHLESS_REASONING_EFFORT``, or "" to use the server default.

    Hosted reasoning models reason at a high effort by default, which costs
    latency and tokens on every call. Not part of the OpenAI spec, so the field
    is only sent when explicitly asked for.
    """
    return os.environ.get("WATCHLESS_REASONING_EFFORT", "").strip()


# ------------------------------------------------------------------ language

# Used when the video's own language cannot be determined. English because it
# is the most common source language on YouTube, so it is the least wrong
# guess -- but it is still a guess, so it is overridable and it is announced.
DEFAULT_LANG_FALLBACK = "en"


def lang_fallback() -> str | None:
    """``WATCHLESS_LANG_FALLBACK``: the language to take when the original is unknown.

    Set it to ``""`` or ``"none"`` to disable guessing entirely and make an
    unresolvable language an error instead.
    """
    raw = os.environ.get("WATCHLESS_LANG_FALLBACK")
    if raw is None:
        return DEFAULT_LANG_FALLBACK
    text = raw.strip().lower()
    if text in ("", "none", "off", "no"):
        return None
    return text


# ------------------------------------------------------------------ network


def proxy_url() -> str | None:
    """Proxy for yt-dlp's own requests."""
    return os.environ.get("WATCHLESS_PROXY") or None


def subtitle_delay() -> float:
    """Seconds to wait before the caption request, not before the next run."""
    raw = os.environ.get("WATCHLESS_SLEEP_SUBTITLES")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 1.0


# ------------------------------------------------------------------ paths


def state_dir() -> Path:
    """Where generated corpora and reports belong, honouring XDG."""
    root = os.environ.get("XDG_STATE_HOME")
    if root:
        return Path(root) / "watchless"
    return Path("~/.local/state").expanduser() / "watchless"
