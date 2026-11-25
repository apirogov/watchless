# Tests

Everything here runs **offline**: no YouTube, no LLM endpoint, no network of any
kind.

| file | what it covers |
|---|---|
| `test_offline.py` | config precedence, filename planning, slugify, reflow, render, URL parsing, prompt shape |
| `test_summarize.py` | the HTTP client, against a local `HTTPServer` on 127.0.0.1 |
| `test_docs.py` | that the package and this README stay consistent |
| `mock_llm.py` | standalone mock endpoint for manual poking |
| `fixtures/segments.json` | a captured transcript, so reflow tests need no fetch |
| `bench/` | prompt tuning against the recorded corpus |

```bash
uv run python -m unittest discover -s tests -t . -v
```

The real fetch path (`yt-dlp`) is not covered: it is network-bound, rate-limited
and changes without notice. Exercise it manually against one URL after changing
`fetch.py`.
