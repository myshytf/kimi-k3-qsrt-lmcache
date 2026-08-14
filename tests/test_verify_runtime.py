from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_runtime.py"


class FixtureHandler(BaseHTTPRequestHandler):
    lmcache_healthy = True
    non_cuda_contexts = 8
    gpu_contexts = 0
    gpu_context_field: str | None = "registered_gpu_ids"
    legacy_gpu_contexts: int | None = None

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "Kimi-K3"}]})
        elif self.path == "/status":
            if self.lmcache_healthy:
                payload = {
                        "health": "OK",
                        "is_healthy": True,
                        "engine_type": "MPCacheServer",
                        "registered_non_cuda_instance_ids": list(
                            range(self.non_cuda_contexts)
                        ),
                    }
                if self.gpu_context_field is not None:
                    payload[self.gpu_context_field] = list(
                        range(self.gpu_contexts)
                    )
                if self.legacy_gpu_contexts is not None:
                    payload["registered_kv_cache_ids"] = list(
                        range(self.legacy_gpu_contexts)
                    )
                self._json(200, payload)
            else:
                self._json(503, {"health": "ERROR"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        if self.path == "/v1/chat/completions" and request.get("model") == "Kimi-K3":
            self._json(
                200,
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": "pong"}}
                    ]
                },
            )
        else:
            self._json(400, {"error": "bad request"})


class RuntimeVerificationTests(unittest.TestCase):
    def _run(
        self,
        lmcache_healthy: bool,
        *,
        non_cuda_contexts: int = 8,
        gpu_contexts: int = 0,
        gpu_context_field: str | None = "registered_gpu_ids",
        legacy_gpu_contexts: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        handler = type(
            "ConfiguredFixtureHandler",
            (FixtureHandler,),
            {
                "lmcache_healthy": lmcache_healthy,
                "non_cuda_contexts": non_cuda_contexts,
                "gpu_contexts": gpu_contexts,
                "gpu_context_field": gpu_context_field,
                "legacy_gpu_contexts": legacy_gpu_contexts,
            },
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            return subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--vllm-url",
                    base,
                    "--lmcache-url",
                    base,
                    "--timeout",
                    "2",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_models_chat_and_lmcache_status_pass(self) -> None:
        result = self._run(lmcache_healthy=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS models", result.stdout)
        self.assertIn("PASS chat", result.stdout)
        self.assertIn("PASS lmcache", result.stdout)

    def test_unhealthy_lmcache_fails(self) -> None:
        result = self._run(lmcache_healthy=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL lmcache", result.stdout)

    def test_missing_engine_driven_rank_fails(self) -> None:
        result = self._run(lmcache_healthy=True, non_cuda_contexts=7)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-CUDA contexts=7 expected=8", result.stdout)

    def test_unexpected_gpu_context_fails(self) -> None:
        result = self._run(lmcache_healthy=True, gpu_contexts=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU contexts=1 expected=0", result.stdout)

    def test_legacy_gpu_context_alias_still_fails(self) -> None:
        result = self._run(
            lmcache_healthy=True,
            gpu_context_field=None,
            legacy_gpu_contexts=1,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU contexts=1 expected=0", result.stdout)

    def test_conflicting_gpu_context_fields_fail_closed(self) -> None:
        result = self._run(
            lmcache_healthy=True,
            gpu_contexts=0,
            legacy_gpu_contexts=1,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU context status fields disagree", result.stdout)

    def test_omitted_zero_gpu_context_field_passes(self) -> None:
        result = self._run(lmcache_healthy=True, gpu_context_field=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("gpu_contexts=0", result.stdout)


if __name__ == "__main__":
    unittest.main()
