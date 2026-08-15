# Validation guide

A running process is not sufficient proof. Validate syntax, registration, inference, persistent external reuse, output correctness, error-free operation and VRAM accounting separately.

## Gate 1: static files

From the repository:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts overlays tests
bash -n launcher/serve-kimi-k3-qsrt-lmcache.sh
```

From the deployment directory:

```bash
bash -n patchwork/serve-kimi-k3-qsrt-lmcache.sh
python3 -m py_compile \
  patchwork/lmcache/integration/vllm/kv_cache_group_edits.py
python3 -m compileall -q patchwork/lmcache patchwork/vllm
docker compose config --quiet
```

## Gate 2: LMCache and engine registration

Follow startup:

```bash
docker compose logs -f --timestamps kimik3
```

Required evidence includes:

```text
ZMQ cache server is running
LMCache MP server ready
KV cache group edits applied
Registering kv caches
Registered engine-driven context
Application startup complete
```

The edit summary must cover the model's recurrent KDA/Mamba groups and MLA groups. Verify all eight ranks register non-CUDA engine-driven contexts, not only rank 0.

Check endpoints:

```bash
curl -fsS http://127.0.0.1:8088/status | python3 -m json.tool
curl -fsS http://127.0.0.1:8090/v1/models | python3 -m json.tool
```

LMCache status schemas can vary by build. Require HTTP 200, a healthy service indication, the intended chunk size, eight registered non-CUDA contexts/IDs, and zero registered GPU contexts in engine-driven mode where those fields are exposed. Do not hard-code an undocumented field name into operational automation without inspecting the actual response.

## Gate 3: short inference

```bash
python3 scripts/verify_runtime.py \
  --vllm-url http://127.0.0.1:8090 \
  --lmcache-url http://127.0.0.1:8088
```

Expected:

```text
PASS models model=Kimi-K3
PASS chat
PASS lmcache
ALL CHECKS PASSED
```

If authentication is enabled, export the key without putting it on the command line:

```bash
export OPENAI_API_KEY='...'
python3 scripts/verify_runtime.py
```

The script never prints the key.

## Gate 4: deterministic long-prefix store

Generate and save a stable prompt:

```bash
python3 scripts/cache_probe.py \
  --url http://127.0.0.1:8090 \
  --characters 80000 \
  --save-prompt /tmp/kimi-k3-lmcache-probe.txt \
  | tee /tmp/kimi-k3-first.json
```

Check `prompt_tokens`. It must exceed one LMCache chunk (12,288 tokens). Character-to-token ratios vary; increase `--characters` if needed.

Record the printed `prompt_sha256`. The second request must use the same file, not regenerate a similar-looking prompt with a timestamp or random content.

Wait until stores are idle and L2 files have appeared:

```bash
curl -fsS http://127.0.0.1:8088/status | python3 -m json.tool
du -sh "$LMCACHE_L2_PATH"
docker compose logs --no-color kimik3 | grep -Ei 'store|l2|flush|write'
```

A short prompt below one complete chunk can correctly create zero external objects.

## Gate 5: prove persistent L2 restore

A second request in the same process can be a vLLM GPU prefix-cache hit. That does not prove LMCache L2. Clear both GPU-local prefix state and LMCache L1 while preserving the mounted L2 directory.

The simplest unambiguous test is a container restart:

```bash
cd "$DEPLOY_DIR"
docker compose stop --timeout 60
# Do not delete or change LMCACHE_L2_PATH.
docker compose up -d
```

Wait for `/v1/models`, then resend the exact saved prompt:

```bash
python3 /path/to/kimi-k3-qsrt-lmcache/scripts/cache_probe.py \
  --url http://127.0.0.1:8090 \
  --prompt-file /tmp/kimi-k3-lmcache-probe.txt \
  | tee /tmp/kimi-k3-restore.json
```

Confirm the SHA-256 values in the first and second JSON files are identical.

Inspect vLLM metrics:

```bash
curl -fsS http://127.0.0.1:8090/metrics \
  | grep -E 'external_prefix_cache|external_kv_transfer|prompt_tokens_by_source'
```

Inspect LMCache metrics and logs:

```bash
curl -fsS http://127.0.0.1:8088/metrics \
  | grep -Ei 'l2|prefetch|load|hit|store'
docker compose logs --no-color kimik3 \
  | grep -Ei 'prefetch|loading.*gpu|external.*hit|l2'
```

Evidence should show:

- external prefix tokens supplied by KV transfer,
- L2 prefetches after the restart,
- CPU-to-GPU loads on all eight ranks,
- fewer locally computed prompt tokens than total prompt tokens,
- identical model output behavior.

Validated example:

```text
prompt tokens: 18,103
external tokens: 12,288
locally computed: 5,815
external hit rate: 67.9%
L2 prefetch: 8/8 chunks
rank loads: 8/8
full miss: ~17.9 seconds
restore: ~5.95–6.24 seconds
```

The 12,288-token hit is exactly one DCP-aware LMCache chunk.

## Gate 6: output correctness

Cache hits are useful only if the output remains correct. At minimum:

1. send a deterministic salted prompt on a full miss and retain its normalized request,
2. replay it in the same process and distinguish vLLM APC from a genuine LMCache local hit,
3. restart with L2 preserved and replay the exact normalized request,
4. require an external-hit increase and compare reasoning, content, tool calls, finish reason, and usage,
5. change only the suffix and require a partial hit (`0 < hit_tokens < prompt_tokens`),
6. inspect for corruption, missing suffix attention, or cross-request leakage.

Do not rely only on throughput, fluent text, or hit counters. Hybrid cache layout mistakes can produce plausible text while diverging at the first generated token. Compare canonical parsed response fields rather than raw SSE framing.

## Gate 7: strict error scan

```bash
docker compose logs --no-color --timestamps kimik3 > /tmp/kimik3.log
grep -En 'Traceback|EngineDead|Engine core initialization failed|CUDA out of memory|OUT_OF_MEMORY|(^|[^a-z])ERROR([^a-z]|$)' /tmp/kimik3.log
```

Review every match. Some libraries log harmless text containing the word `error`, but do not dismiss an uppercase engine or LMCache error without tracing it.

Also verify lifecycle state:

```bash
docker inspect kimik3 \
  --format 'status={{.State.Status}} exit={{.State.ExitCode}} restarts={{.RestartCount}} started={{.State.StartedAt}}'
```

Expected: `running`, exit code 0 for the active state, and restart count 0.

## Gate 8: VRAM attribution and KV capacity

Capture per-process values at idle and after the long request:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

In engine-driven mode the compute-process list must contain the eight vLLM workers and no standalone LMCache server PID. Also record device-level free memory and verify startup reports at least 860,160 aggregate KV tokens for two block-rounded 420K sequences. Do not add pointer-mode vLLM and LMCache rows as if CUDA IPC mappings were independent physical allocations. See [VRAM_ACCOUNTING.md](VRAM_ACCOUNTING.md).

## Gate 9: restart persistence

After all checks pass, perform one normal Compose recreate and repeat:

- `/v1/models`,
- short chat,
- LMCache `/status`,
- deterministic L2 restore,
- restart count,
- strict error scan.

This catches patches that were applied interactively inside a container but not persisted as bind mounts.

When `K3_LMCACHE_RESET_L2_ON_START=auto`, also require the layout guard to report `preserve-compatible` and verify that the retained inventory belongs to the current layout fingerprint. A geometry-changing deployment must increment the layout revision so incompatible token-hash-keyed objects are removed once, before the new canonical store.

## Gate 10: gateway continuation and concurrency

The direct vLLM probe does not validate an agent gateway. Through the production gateway:

1. require one tool call with progressive JSON argument deltas,
2. append the assistant reasoning/tool call and the tool result,
3. require a second tool call and append its result,
4. require a final tool-free answer without changing the tool schema,
5. run two forced-tool requests concurrently and observe two running sequences.

Validate tool names, call IDs, reconstructed JSON, finish reasons, and completed client streams. An aggregate `num_requests_running=1` after the canary is not by itself a leak: correlate gateway request IDs, user agents, request sizes/hashes, and timestamps before attributing unrelated live traffic.

## Gate 11: immutable cutover

Diagnostic bind mounts are not the final product. Bake qualified code into a digest-pinned image, verify source hashes during the build, and remove checksum/dev/reset overrides from the running container. The final mount set should contain only model, cache, temporary-data, and persistent-L2 data paths. Preserve the pre-cutover Compose as a rollback artifact rather than leaving it active.
