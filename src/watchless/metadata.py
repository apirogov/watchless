"""Cheap metadata that does not need a caption fetch.

YouTube's oEmbed endpoint is stdlib-only, needs no API key, and returns the
title and channel in ~200ms. It has no description field, so anything richer
comes from :class:`watchless.fetch.YtdlpBackend`. Calling oEmbed first means a
plain run never pays the ~1-2s import cost of yt-dlp.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

OEMBED = "https://www.youtube.com/oembed"
# Qualified on purpose: this is not the LLM timeout. oEmbed is a fast,
# unauthenticated metadata endpoint; 15s is generous. A bare TIMEOUT_S here
# read as if it were the same knob settings.TIMEOUT_S owns.
OEMBED_TIMEOUT_S = 15


def oembed_metadata(vid: str) -> tuple[str | None, str | None]:
    """``(title, channel)`` from oEmbed. Both ``None`` if unavailable.

    Never raises: metadata is a nicety, and a video whose transcript is
    available should still be transcribed if oEmbed is having a bad day.
    """
    url = f"{OEMBED}?format=json&url=https://www.youtube.com/watch?v={vid}"
    request = urllib.request.Request(url, headers={"User-Agent": "watchless"})
    try:
        with urllib.request.urlopen(request, timeout=OEMBED_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    title = payload.get("title")
    channel = payload.get("author_name")
    return (
        str(title) if title else None,
        str(channel) if channel else None,
    )
