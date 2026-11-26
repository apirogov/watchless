# watchless

[![PyPI version](https://img.shields.io/pypi/v/watchless)](https://pypi.org/project/watchless/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/watchless)](https://pypi.org/project/watchless/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Typed: mypy](https://img.shields.io/badge/typed-mypy-blue)](pyproject.toml)

YouTube URL in, a readable Markdown document out: the video's transcript as
prose, and — with an LLM endpoint configured — a summary of what it claims and
how far to trust it.

````markdown
# Vitamin D Toxicity Is Very Rare—But That Doesn't Mean It's Safe

- URL: https://www.youtube.com/watch?v=i2zp1qnUZlI
- Channel: Dr. Eric Berg DC
- Track via: ytdlp, en (auto)

## Summary

**TL;DR:** The video argues that vitamin D "danger" is mostly about depleted
cofactors and calcium handling, not true toxicity, and gives supplement and food
rules to avoid problems.

- [claim] True toxicity is rare and mainly hypercalcemia; the speaker says Bruce
  Hollis has never seen a case and that it needs hundreds of thousands of IU for
  months (named: Bruce Hollis; threshold no source).
- [claim] Vitamin D raises needs for magnesium, zinc, K2 and retinol; magnesium
  is said to prevent vascular calcification (cited paper), while K2/A calcium
  claims are unsourced.
- [procedure] It advises D3 over D2, avoiding synthetic D3/starch fillers (iodine
  test), magnesium glycinate, zinc with trace minerals, and MK-7 K2 at 100 mcg
  per 10,000 IU D3 (no source).
- [procedure] For high doses, it says to involve a doctor, monitor PTH, vitamin
  D, creatinine and calcium/urine calcium, drink 2.5 L fluids daily, and avoid
  excess calcium; kidney calcium deposits are the stated risk (no source).

**Assessment:** Dr. Berg sells supplements and is a chiropractor, so his
product-adjacent dosing advice has a commercial interest; the video cites Hollis,
a paper and a pharmacology text but leaves several thresholds and mechanisms
unsourced. Check the cited sources and clinician guidance before using high-dose
regimens.

## Description

What happens if you take too much vitamin D? In this video, we'll discuss the
vitamin D dangers you should know about. […] my full line of high-quality
supplements is available on Amazon — search Dr. Berg Supplements.

## Transcript

today we're going to talk about vitamin D dangers which is very different than
vitamin D toxicity they're not the same it's extremely rare to develop a vitamin
D toxicity which really involves Only One symptom that's too much calcium in the
blood I talked to Professor Bruce Hollis who is the Pioneer in vitamin D research
[…]
````

Every line of that summary is attributable. `[claim]` and `[procedure]` are the
video talking; `[overview]` is the summarizer talking. `(named: Bruce Hollis)`
marks a cited source, `(no source)` one stated without evidence — the difference
between "the video says so" and "the video shows its work". The Assessment names
the conflict of interest rather than judging the medicine.

The transcript is unedited speech-to-text: no punctuation, no speaker labels,
lowercase. It is verbatim, and the summary is what makes it readable.

---

## Install

```bash
uv tool install watchless
```

That puts `watchless` in `~/.local/bin`.

Then:

```bash
# let other Android apps read the saved .txt (Termux only)
termux-setup-storage

# optional but recommended on Termux: the Termux:API app (separate APK)
pkg install termux-api
```

That last step provides `termux-open`, which is what pops the transcript open.
Without it the file is still written, it just does not open by itself.

### Android share sheet

```bash
watchless --install-termux-opener
```

Writes `~/bin/termux-url-opener` — the fixed path the Termux app executes when
you share a link to it. It checks for Termux and the Termux:API commands first
and says what to install if anything is missing. An existing opener is not
replaced without asking; `--force` skips the question.

Termux:API is two installs: `pkg install termux-api` provides the commands, and
the Termux:API app from [F-Droid](https://f-droid.org/packages/com.termux.api/)
provides the API they call. Neither alone works. Without it the opener still
saves the transcript, just silently.

---

## Usage

```
watchless [URL] [-o FILE | --out-dir DIR]
       [--lang de,en]
       [--slug | --no-slug] [--max-slug N] [--no-header]
       [--description | --no-description] [--timestamps | --no-timestamps]
       [--sleep-subtitles SEC]
       [--llm-url URL] [--llm-key TOKEN] [--llm-model NAME]
       [--bullets N] [--llm-timeout SEC] [--llm-retries N]
       [--summary | --no-summary]
       [--install-termux-opener] [--force]
       [--doctor] [--version]
```

URL is optional only for `--doctor`, `--version` and `--install-termux-opener`.

The path of the written file goes to **stdout**; everything else goes to
**stderr**. So `f=$(watchless "$url")` works.

| flag | effect |
|---|---|
| *(none)* | `DIR/<video-id>.txt` with a header block |
| `-o FILE` / `--out FILE` | write to an exact path |
| `--out-dir DIR` | write to `DIR/<name>.txt` |
| `--slug` | `<slug>--<id>.txt`, e.g. `grippeimpfung-fuers-kind-sinnvoll-auch-wenn-es--7qt3L44LoiE.txt` |
| `--max-slug N` | slug length limit, cut on a word boundary (default 48); `0` disables the slug |
| `--no-header` | omit the title/URL/Channel block |
| `--description` | include the description as its own section (default **on**) |
| `--timestamps` | prefix each paragraph with its start time (default **on**) |
| `--lang de,en` | request specific tracks; default is the original language |
| `--sleep-subtitles SEC` | pause before the caption request (default 1); raise it if you hit 429s |
| `--llm-url URL` | enable summaries |
| `--llm-key`, `--llm-model` | auth token, model name |
| `--bullets N` | ceiling on summary lines; each bullet gets a fixed word budget |
| `--llm-timeout SEC` | how long to wait for one answer (default 300) |
| `--llm-retries N` | extra attempts on a 5xx or connection error, timeout each (default 0) |
| `--summary` | require a summary: exit 4 instead of silently skipping |
| `--no-summary` | skip it even when an endpoint is configured |
| `--doctor` | report resolved config and test the endpoint, then exit |
| `--install-termux-opener` | install the Android share-sheet target, then exit |
| `--force` | with the above: overwrite without asking |
| `--version` | print the version, then exit |

### `--lang`

No `--lang` means the video's original language. Asking for others hits
YouTube's transcode tracks, which are aggressively rate limited.

If the original cannot be determined it uses `WATCHLESS_LANG_FALLBACK`
(default `en`); `none` makes that case exit 2 instead.

### `--timestamps`

```markdown
**[0:00]** Wegen dieser Krankheit landeten in nur einem Winter 30.000 Kinder ...

**[0:08]** Die Rede ist von der Influenza bzw. der echten Grippe ...
```

Stamps are accurate to a caption line (typically ~2s).

They are for **you, not the model**: the file carries them, the summary input
never does.

---

## Summaries

Point it at any OpenAI-compatible `/v1/chat/completions` endpoint and you get a
`## Summary` section above the description and the transcript.

```bash
# local llama-server, no auth
watchless "URL" --llm-url http://192.168.1.10:8080/v1

# hosted, with a token and a model name
watchless "URL" --llm-url https://api.example.com/v1 \
    --llm-key "$TOKEN" --llm-model qwen3-8b --bullets 6
```

A summary happens exactly when an endpoint URL resolves; there is no separate
on/off switch. `--llm-model` is required alongside it, and naming a model the
endpoint does not serve fails immediately with the list of ones it does.

```
**TL;DR:** one or two sentences — the summary of the summary.

- [claim] the author asserts this (no source)
- [procedure] how to do it
- [overview] what the video is, from the summarizer

**Assessment:** what to weigh before believing it, and what to check.
```

The rules behind that output are the prompt, which is the specification:
[`src/watchless/prompts/summarize.md`](src/watchless/prompts/summarize.md). It is plain
text — edit it and re-run, no rebuild.

---

## Exit codes

| code | meaning |
|---|---|
| `0` | ok, transcript written |
| `1` | bad arguments or unrecognised URL |
| `2` | no captions exist for this video |
| `3` | YouTube is blocking this IP |
| `4` | `--summary` was passed but no endpoint is configured |

`2` is permanent for that video; `3` is about your network and will pass.

---

## Failure modes

Most failures explain themselves: a block prints what to do (wait, switch
network, set `WATCHLESS_PROXY`), and a summary that cannot reach its endpoint
says so and still writes the transcript. Three are worth knowing in advance.

**YouTube is blocking this IP** (exit 3). Blocks run from hours to days and are
much likelier on a VPN or a datacenter IP. Retrying later is free; nothing is
lost. The request burst is *inside* one run — metadata, track list, subtitles —
so `--sleep-subtitles` is the lever that matters, not spacing between videos.
Do not add cookies: authenticating around a block risks a permanent account ban.

**No captions exist** (exit 2). Permanent for that video, and the tool does no
speech recognition. Transcribe the audio separately (`yt-dlp -x`, then whisper or
an on-device model) and you have the text either way.

**The summary is refused as too long.** How much of a video fits is the
endpoint's context window, read from `GET /models`. A transcript past half of it
is refused rather than cut: the error says so, nothing is sent, and the
transcript is still written. Use a model with a larger window, or transcribe in
parts.

**No summary appears.** Run `--doctor`: it reports what was resolved and why the
endpoint did or did not answer. A large model cold-loading can exceed the 300s
default; raise `--llm-timeout`. The timeout is per attempt, so the total wait is
`(1 + --llm-retries) * --llm-timeout` — with the default `--llm-retries 0` that
is simply the timeout you asked for.

---

## Configuration

Config is read from env vars, and every env var has a flag. A flag wins even
when empty (`--llm-model ""` clears `WATCHLESS_MODEL`); an empty env var is ignored.

```bash
export WATCHLESS_DEST="$HOME/storage/shared/transcripts"
export WATCHLESS_SLUG=1
```

| env var | flag |
|---|---|
| `WATCHLESS_BASE_URL` | `--llm-url` |
| `WATCHLESS_API_KEY` | `--llm-key` |
| `WATCHLESS_MODEL` | `--llm-model` (required once an endpoint is set) |
| `WATCHLESS_SLUG` | `--slug` / `--no-slug` |
| `WATCHLESS_DESCRIPTION` | `--description` / `--no-description` |
| `WATCHLESS_TIMESTAMPS` | `--timestamps` / `--no-timestamps` |
| `WATCHLESS_SLEEP_SUBTITLES` | `--sleep-subtitles` |
| `WATCHLESS_VERBOSE` | progress on stderr |
| `WATCHLESS_PROXY` | proxy for yt-dlp |
| `WATCHLESS_REASONING_EFFORT` | sent as `reasoning_effort`; unset omits it |
| `WATCHLESS_LANG_FALLBACK` | language when the original is unknown (default `en`) |
| `WATCHLESS_DEST` | output directory, used by the share sheet |
| `WATCHLESS_BULLETS` | `--bullets`, used by the share sheet |
| `WATCHLESS_SLUG_OVERRIDE` | `1` / `0` for `--slug` / `--no-slug`, used by the share sheet |

Booleans are true for `1`, `true`, `yes`, `on`. Everything else is off.

On Termux the share sheet sources `~/.config/watchless.env`, so put them there:

```bash
export WATCHLESS_DEST="$HOME/storage/shared/transcripts"
export WATCHLESS_BASE_URL=http://192.168.1.10:8080/v1
export WATCHLESS_MODEL=qwen3-8b
```

**`export` is required.** A bare `WATCHLESS_MODEL=...` stays a shell variable and
the Python child never sees it.

