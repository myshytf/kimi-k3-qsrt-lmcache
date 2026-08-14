#!/usr/bin/env python3
"""Send a deterministic long prompt for LMCache persistence experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--model", help="Model ID (default: first /v1/models entry).")
    parser.add_argument("--characters", type=int, default=80000)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--save-prompt", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser.parse_args()


def endpoint(base: str, path: str) -> str:
    return base.rstrip("/") + path


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
        return json.loads(response.read().decode("utf-8"))


def deterministic_prompt(minimum_characters: int) -> str:
    if minimum_characters < 1:
        raise ValueError("--characters must be positive")
    lines = [
        "This is a deterministic LMCache prefix-persistence probe.\n",
        "Retain every numbered fact and answer only with OK at the end.\n",
    ]
    total_characters = sum(map(len, lines))
    index = 0
    while total_characters < minimum_characters:
        line = (
            f"cache-probe-{index:08d}: alpha beta gamma delta epsilon "
            f"zeta eta theta; value={index % 997:03d}.\n"
        )
        lines.append(line)
        total_characters += len(line)
        index += 1
    return "".join(lines)


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)

    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    else:
        prompt = deterministic_prompt(args.characters)
    if args.save_prompt:
        args.save_prompt.parent.mkdir(parents=True, exist_ok=True)
        args.save_prompt.write_text(prompt, encoding="utf-8")

    model = args.model
    if not model:
        models = request_json(
            "GET", endpoint(args.url, "/v1/models"), args.timeout, api_key
        )
        entries = models.get("data") if isinstance(models, dict) else None
        if not isinstance(entries, list) or not entries or not entries[0].get("id"):
            raise SystemExit("/v1/models returned no usable model ID")
        model = entries[0]["id"]

    started = perf_counter()
    response = request_json(
        "POST",
        endpoint(args.url, "/v1/chat/completions"),
        args.timeout,
        api_key,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        },
    )
    elapsed = perf_counter() - started
    usage = response.get("usage") if isinstance(response, dict) else None
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(usage, dict) or not isinstance(choices, list) or not choices:
        raise SystemExit("completion response is missing usage or choices")

    summary = {
        "model": model,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_characters": len(prompt),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_seconds": round(elapsed, 3),
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
