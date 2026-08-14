# Contributing

Contributions are welcome when they are tied to a reproducible image/version and include correctness evidence.

## Before opening a change

1. Open an issue describing the exact vLLM, LMCache, Python, PyTorch, CUDA, model and image versions.
2. Include the first complete traceback or the measurable problem.
3. Explain whether the change affects cache layout, DCP math, transfer protocol, staging or documentation.
4. Remove credentials, private paths, raw private prompts and model files.

## Development checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts overlays tests
bash -n launcher/serve-kimi-k3-qsrt-lmcache.sh
python3 scripts/install_overlays.py --destination /tmp/kimi-overlay-test --dry-run
docker compose --env-file examples/.env.example -f examples/compose.yml config --quiet
git diff --check
```

## Requirements for code changes

- Add a regression test that fails before the fix.
- Preserve payload and base hashes accurately; do not edit `manifest.json` by hand without recomputing them.
- Keep modified-file notices and Apache-2.0 attribution.
- Do not replace a custom native extension with Python code from a different ABI without a complete port.
- For tensor views, test shape, stride, byte count and `data_ptr()` preservation.
- For DCP changes, test every rank and record effective `tokens_per_block`.
- For cache hits, prove output correctness as well as metrics.
- For performance claims, provide the request shape, miss/hit source, warm-up state and before/after measurements.

## Pull-request evidence

A compatibility PR should include:

- exact image digest and package versions,
- original and patched file SHA-256 values,
- startup logs through API readiness,
- short inference result,
- deterministic prompt SHA and token count,
- L2-preserving restart evidence,
- external-token and per-rank load metrics,
- strict error scan,
- idle/request VRAM accounting,
- rollback result.

Do not claim support for an untested version range.
