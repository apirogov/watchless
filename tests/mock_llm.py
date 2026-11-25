"""Minimal mock OpenAI-compatible /v1/chat/completions server for testing.

Records the last request so the test can assert on what was actually sent.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

LAST: dict[str, Any] = {"body": None, "auth": None, "path": None}
MODE = {"value": "ok"}

REPLY = """- Die STIKO empfiehlt die Grippeimpfung nur für Risikokinder ab 6 Monaten.
- Vier von fünf hospitalisierten Kindern waren zuvor kerngesund.
- Nur jedes zehnte Risikokind wird tatsächlich geimpft.
- Ab zwei Jahren ist die Impfung als Nasenspray möglich.
- Österreich und England impfen alle Kinder, Deutschland nicht.
- Der Schutz liegt je nach Saison bei 20 bis 60 Prozent.
- Bei Geimpften verläuft die Grippe milder und mit weniger Komplikationen.
- Die Impfung schützt auch Risikopersonen im Umfeld.
- Für gesunde Kinder ist es eine individuelle Entscheidung.
- Bei Fragen soll man die Kinderarztpraxis konsultieren."""


class H(BaseHTTPRequestHandler):
    def log_message(self, *_a: object) -> None:  # silence
        pass

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        LAST["body"] = json.loads(raw)
        LAST["auth"] = self.headers.get("Authorization")
        LAST["path"] = self.path

        if MODE["value"] == "500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":"boom"}')
            return
        if MODE["value"] == "badjson":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"not json at all")
            return

        payload = {
            "id": "mock-1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": REPLY},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 700, "completion_tokens": 120},
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(port: int = 8765) -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", port), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    if len(sys.argv) > 2:
        MODE["value"] = sys.argv[2]
    print(f"mock listening on {port} mode={MODE['value']}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
