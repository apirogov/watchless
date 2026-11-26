"""Transport: talk to an OpenAI-compatible chat endpoint. Stdlib only.

Works with any host exposing POST /v1/chat/completions -- llama-server, Ollama,
llama.cpp, LocalAI, LM Studio, vLLM, a hosted API. Takes a prompt string,
returns the assistant text; it knows nothing about transcripts or bullets.

Config precedence: CLI flags > environment > defaults.

    WATCHLESS_BASE_URL          e.g. http://192.168.1.10:8080/v1
    WATCHLESS_API_KEY           bearer token; unset for llama-server
    WATCHLESS_MODEL             required; the request is rejected without it
    WATCHLESS_REASONING_EFFORT  unset omits the field
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from watchless import settings
from watchless.log import debug
from watchless.settings import TIMEOUT_S

__all__ = [
    "SummarizeError",
    "call",
    "chat_probe",
    "connection_hint",
    "describe_models",
    "list_models",
    "preflight",
    "probe_models",
    "reasoning_effort",
    "resolve",
]


class SummarizeError(RuntimeError):
    pass


class _Transient(SummarizeError):
    """Retryable: connection failure, timeout, 5xx, 429."""


def resolve(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Single source of truth for LLM config precedence: arg > env > default.

    Delegates to :func:`watchless.settings.pick`, the one precedence rule in the
    program. Callers resolve once and pass concrete values down, so the
    /models preflight and the request can never disagree.

    Passing ``""`` explicitly clears a setting (e.g. ``--llm-model ""`` omits
    the field even when ``WATCHLESS_MODEL`` is set).
    """
    base = settings.pick(base_url, "WATCHLESS_BASE_URL", "") or ""
    key = settings.pick(api_key, "WATCHLESS_API_KEY", None)
    mdl = settings.pick(model, "WATCHLESS_MODEL", None)
    return base.rstrip("/"), key, mdl


def _endpoint(base_url: str | None, api_key: str | None) -> tuple[str, str | None]:
    """Resolve the base URL to .../chat/completions. No model needed."""
    base, key, _ = resolve(base_url, api_key, None)
    if not base:
        raise SummarizeError("no endpoint: pass --llm-url or set WATCHLESS_BASE_URL")
    if not base.endswith("/chat/completions"):
        base = base + "/chat/completions"  # accept ".../v1" and a bare host
    return base, key


def _config(
    base_url: str | None, api_key: str | None, model: str | None
) -> tuple[str, str | None, str]:
    """As :func:`_endpoint`, and additionally require a model."""
    base, key = _endpoint(base_url, api_key)
    mdl = resolve(base_url, api_key, model)[2]
    if not mdl:
        # Required, not optional: the field is mandatory in the request, and
        # guessing a name would only move the failure to the server's 422.
        raise SummarizeError(
            "no model: pass --llm-model or set WATCHLESS_MODEL "
            "(list the ids with: curl -s <endpoint>/models)"
        )
    return base, key, mdl


def _with_model(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """Add the required ``model`` field, plus ``reasoning_effort`` if set.

    ``model`` is required by the OpenAI spec; hosting endpoints reject a body
    without it (llama.cpp answers 422 "Field required at 'model'").

    ``reasoning_effort`` is not in the spec, so it is sent only when set --
    servers that do not know the field must not receive it.
    """
    payload["model"] = model
    effort = reasoning_effort()
    if effort:
        payload["reasoning_effort"] = effort
    return payload


def _chat_request(
    url: str,
    key: str | None,
    payload: dict[str, Any],
) -> urllib.request.Request:
    """A JSON POST with auth. The one place a chat request is constructed.

    ``chat_probe`` and ``_post_once`` differ only in payload and in how they
    report failure; building the request twice meant the auth header and
    content type had to be kept in sync by hand.
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    return req


def reasoning_effort() -> str:
    """``WATCHLESS_REASONING_EFFORT``, or "" to use the server default.

    Hosted reasoning models (DeepSeek, GLM, Kimi, Qwen) reason at a high effort
    by default, which costs latency and tokens on every call. Not part of the
    OpenAI spec, so the field is only sent when explicitly asked for; servers
    that do not know it never receive it.
    """
    return settings.reasoning_effort()


# --------------------------------------------------------------------- probes


def probe_models(
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 10,
) -> tuple[list[str], dict[str, Any]]:
    """GET {base}/models. Returns (ids, diagnosis) where ids may be empty.

    The diagnosis separates causes that would otherwise all read as "no answer":
      reachable: the HTTP layer answered at all
      status:    HTTP status, if the server replied
      error:     exception text, if it did not
    """
    url, key = _endpoint(base_url, api_key)
    root = url.rsplit("/chat/completions", 1)[0]
    endpoint = f"{root}/models"
    debug(f"GET {endpoint} (preflight)")
    req = urllib.request.Request(endpoint, method="GET")
    if key:
        req.add_header("Authorization", f"Bearer {key}")

    info: dict[str, Any] = {
        "endpoint": endpoint,
        "reachable": False,
        "status": None,
        "error": None,
        "body_head": None,
    }
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            info["reachable"] = True
            info["status"] = r.status
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        info["reachable"] = True  # the server answered, with an error
        info["status"] = e.code
        info["body_head"] = e.read().decode("utf-8", "replace")[:200]
        return [], info
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
        return [], info

    info["body_head"] = raw[:200]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], info
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], info
    return [i["id"] for i in items if isinstance(i, dict) and "id" in i], info


def list_models(
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 10,
) -> list[str]:
    """GET {base}/models -> available ids. Empty list if unsupported."""
    ids, info = probe_models(base_url, api_key, timeout)
    if not ids:
        debug(f"/models: {describe_models(info, ids)}")
    return ids


def describe_models(info: dict[str, Any], ids: list[str] | None = None) -> str:
    """One readable line from a probe_models() diagnosis.

    Pass the ids too: a 200 with a valid list is success, and must not be
    described as "no usable model list".
    """
    if ids:
        return f"ok, {len(ids)} model(s)"
    error = info.get("error")
    if error:
        return str(error)
    status = info.get("status")
    if status:
        return (
            f"HTTP {status} - no usable model list "
            f"({(info.get('body_head') or '')[:60]!r})"
        )
    return "no HTTP response"


def connection_hint(detail: str) -> str:
    """Plain-language next step for a probe failure."""
    d = detail.lower()
    if "connection refused" in d:
        return (
            "nothing is listening there. Is the server running, on that "
            "host:port, and bound to 0.0.0.0 rather than 127.0.0.1?"
        )
    if "timed out" in d:
        return (
            "host did not answer in time; wrong IP, firewall, or not on this network?"
        )
    if "name or service not known" in d or "nodename" in d:
        return "hostname did not resolve; use an IP address."
    if "network is unreachable" in d or "no route" in d:
        return "no route to that address; are you on the same network/VPN?"
    if "http 401" in d or "http 403" in d:
        return "server reachable but rejected the request; check WATCHLESS_API_KEY."
    if "http 404" in d:
        return "server reachable but has no such route; check the /v1 suffix."
    if "non-json" in d:
        return "something answered, but it is not an OpenAI-compatible API."
    return ""


def chat_probe(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: int = 15,
) -> tuple[bool, str]:
    """One tiny chat request. Returns (ok, detail). Never raises.

    /models is optional in the OpenAI spec; the chat endpoint is authoritative.

    max_tokens is generous because a reasoning model burns tokens thinking
    before it emits content; a tight budget returns an empty message that looks
    like a broken endpoint.
    """
    url, key, mdl = _config(base_url, api_key, model)
    payload = _with_model(
        {
            "messages": [{"role": "user", "content": "say ok"}],
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
        },
        mdl,
    )
    req = _chat_request(url, key, payload)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")[:160]
        return False, f"HTTP {e.code} {e.reason}: {text}"
    except urllib.error.URLError as e:
        return False, f"cannot connect: {getattr(e, 'reason', e)}"
    except TimeoutError:
        return False, f"timed out after {timeout}s (model may still be loading)"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    try:
        reply = _extract_content(body, url)
    except SummarizeError as e:
        return False, str(e)[:160]
    return True, f"OK - replied {reply[:60]!r}"


def preflight(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> None:
    """Fail early and clearly if the requested model isn't served.

    A warning-free no-op only when the server does not expose /models
    (llama-server, Ollama and friends all do).
    """
    available = list_models(base_url, api_key)
    if not available:
        return  # server has no /models; let chat decide
    if model not in available:
        raise SummarizeError(
            f"model {model!r} is not served by this endpoint. "
            f"Available: {', '.join(available[:10])}"
            + (" ..." if len(available) > 10 else "")
        )
    debug(f"model {model!r} confirmed by /models")


# ------------------------------------------------------------------- requests


def call(
    prompt: str,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: int = TIMEOUT_S,
    retries: int = 0,
) -> str:
    """POST the prompt, retrying transient errors up to ``retries`` extra times.

    ``timeout`` is per attempt, so the worst case is ``(retries + 1) * timeout``
    plus the sleeps between attempts. The default of 0 means one attempt: a
    timeout is the budget being spent, and a second identical wait rarely helps
    a model that is still loading.
    """
    import time  # noqa: PLC0415

    last: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.monotonic()
        try:
            out = _post_once(prompt, base_url, api_key, model, timeout)
            debug(f"ok in {time.monotonic() - t0:.1f}s")
            return out
        except _Transient as e:
            last = e
            if attempt >= retries:
                break
            from watchless.log import log  # noqa: PLC0415

            log(f"warning: {e}; retrying ({attempt + 2}/{retries + 1})")
            time.sleep(1.5 * (attempt + 1))
        except KeyboardInterrupt:
            # Ctrl-C during a blocking connect: not a crash, just stop.
            from watchless.log import log  # noqa: PLC0415

            log("warning: interrupted while waiting for the LLM endpoint")
            raise
    raise SummarizeError(str(last)) from last


def _post_once(
    prompt: str,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: int = TIMEOUT_S,
) -> str:
    url, key, mdl = _config(base_url, api_key, model)
    debug(f"POST {url} model={mdl} auth={'yes' if key else 'no'} timeout={timeout}s")
    payload = _with_model(
        {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": False,
        },
        mdl,
    )
    req = _chat_request(url, key, payload)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        msg = f"HTTP {e.code} from {url}: {detail}"
        if e.code >= 500 or e.code == 429:
            raise _Transient(msg) from e
        raise SummarizeError(msg) from e
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        raise _Transient(
            f"cannot reach {url}: {reason}. Is the endpoint up and reachable "
            f"from this device (right host/port, same network, not bound to "
            f"127.0.0.1 on the server)?"
        ) from e
    except TimeoutError as e:
        raise _Transient(
            f"no answer after {timeout}s from {url}. The model may still be "
            f"loading, or too slow for this timeout -- try --llm-timeout 900"
        ) from e

    return _extract_content(raw, url)


def _extract_content(raw: str, url: str = "") -> str:
    """Pull the assistant text out of a chat-completions response.

    Handles the three shapes seen in the wild:
      content: str                      (standard)
      content: [{type: text, text: ..}] (content parts)
      content: null + reasoning_content (reasoning models)
    """
    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        msg = choice["message"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise SummarizeError(
            f"unexpected response shape from {url}: {raw[:200]!r}"
        ) from e

    content = msg.get("content")
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    if not content:
        content = msg.get("reasoning_content") or ""
    if not isinstance(content, str) or not content.strip():
        raise SummarizeError(f"empty assistant message from {url}")

    # A length stop means the bullet list was cut off mid-sentence.
    if choice.get("finish_reason") == "length":
        from watchless.log import log  # noqa: PLC0415

        log("warning: response hit the token limit; summary may be truncated")

    return content.strip()
