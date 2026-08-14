#!/usr/bin/env python3
"""Verify vLLM model discovery, chat inference, and LMCache health."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8090")
    parser.add_argument("--lmcache-url", default="http://127.0.0.1:8088")
    parser.add_argument("--model", help="Model ID (default: first /v1/models entry).")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key; never printed.",
    )
    return parser.parse_args()


def request_json(
    method: str,
    url: str,
    timeout: float,
    api_key: str | None,
    payload: dict[str, Any] | None = None,
) -> Any:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def endpoint(base: str, path: str) -> str:
    return base.rstrip("/") + path


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    failures = 0
    model = args.model

    try:
        models = request_json(
            "GET",
            endpoint(args.vllm_url, "/v1/models"),
            args.timeout,
            api_key,
        )
        entries = models.get("data") if isinstance(models, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError("response has no model entries")
        if model is None:
            model = entries[0].get("id")
        if not isinstance(model, str) or not model:
            raise ValueError("model ID is missing")
        print(f"PASS models model={model}")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as error:
        failures += 1
        print(f"FAIL models {type(error).__name__}: {error}")

    if model is not None:
        try:
            chat = request_json(
                "POST",
                endpoint(args.vllm_url, "/v1/chat/completions"),
                args.timeout,
                api_key,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with pong."}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                },
            )
            choices = chat.get("choices") if isinstance(chat, dict) else None
            if not isinstance(choices, list) or not choices:
                raise ValueError("response has no choices")
            print("PASS chat")
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as error:
            failures += 1
            print(f"FAIL chat {type(error).__name__}: {error}")

    try:
        status = request_json(
            "GET",
            endpoint(args.lmcache_url, "/status"),
            args.timeout,
            None,
        )
        if not isinstance(status, dict):
            raise ValueError("status response is not a JSON object")
        health = status.get("health")
        if isinstance(health, str) and health.lower() not in {"ok", "healthy"}:
            raise ValueError(f"reported health={health}")
        print("PASS lmcache")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as error:
        failures += 1
        print(f"FAIL lmcache {type(error).__name__}: {error}")

    if failures:
        print(f"FAILED checks={failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
