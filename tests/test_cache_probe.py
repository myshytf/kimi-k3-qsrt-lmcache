from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cache_probe.py"


class ProbeHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send({"data": [{"id": "Kimi-K3"}]})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        content = request["messages"][0]["content"]
        if len(content) < 20000:
            self.send_error(400, "prompt too short")
            return
        self._send(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 18103, "completion_tokens": 1},
            }
        )


class CacheProbeTests(unittest.TestCase):
    def test_probe_is_deterministic_and_reports_usage(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProbeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            summaries = []
            for _ in range(2):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--url",
                        base,
                        "--characters",
                        "24000",
                        "--timeout",
                        "2",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                summaries.append(json.loads(result.stdout))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(summaries[0]["prompt_sha256"], summaries[1]["prompt_sha256"])
        self.assertGreaterEqual(summaries[0]["prompt_characters"], 24000)
        self.assertEqual(summaries[0]["prompt_tokens"], 18103)
        self.assertEqual(summaries[0]["model"], "Kimi-K3")


if __name__ == "__main__":
    unittest.main()
