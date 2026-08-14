# Troubleshooting

## Error-to-fix index

| Symptom | Meaning | Correct action |
|---|---|---|
| `Failed to promote local KV cache specs to one unified type` | Hybrid manager was disabled for mixed KDA/Mamba + MLA specs | Remove `--disable-hybrid-kv-cache-manager` |
| `expected a Mamba [conv_state, ssm_state] tensor list, got Tensor` | Old LMCache group edit cannot read vLLM 0.26 unified Mamba tensor | Verify all four LMCache overlays are mounted read-only |
| `chunk_size (1536) must be a multiple of ... tokens_per_block (12288)` | Rank-local scheduler step was used as the DCP-global LMCache chunk | Keep scheduler 1536; set LMCache chunk 12288 for DCP8 |
| `EngineKVFormat` attribute/enum error | Python overlay and native LMCache extension are from different ABIs | Stop; restore matched files or port against the new native build |
| LMCache server exits before vLLM | Bad L1/L2 args, no disk access, native import failure, or CUDA registration error | Read the LMCache log from its first line before changing settings |
| No cache objects after a request | Prompt may be shorter than one chunk, request may have failed, or store is still in progress | Check prompt token count (>12,288), status, metrics and logs |
| Fast repeated request but no external metrics | vLLM GPU prefix cache served it | Restart while preserving L2, then resend the exact prompt |
| `check_image.py` reports mismatch | Different image/file build | Do not mount overlays; perform a source-level port |

## Preserve the first exception

Do not repeatedly recreate a failing container before saving the original error:

```bash
docker inspect kimik3 \
  --format 'status={{.State.Status}} exit={{.State.ExitCode}} error={{.State.Error}}'
docker logs --timestamps kimik3 > /tmp/kimik3-startup.log 2>&1
```

Find the first traceback, not only the final `EngineDead` wrapper.

## Compose validates but engine still dies

`docker compose config --quiet` checks YAML and interpolation only. Cache-spec promotion, tensor views, native formats and DCP block calculations occur during vLLM engine startup. A successful Compose validation does not narrow those failures to zero.

## Overlay appears not to load

Inspect mounts:

```bash
docker inspect kimik3 \
  --format '{{range .Mounts}}{{println .Source "->" .Destination .Mode}}{{end}}'
```

Verify hashes inside the running container against `patched_sha256` values:

```bash
docker exec kimik3 sha256sum \
  /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/kv_cache_group_edits.py \
  /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/lmcache_mp_connector.py \
  /opt/venv/lib/python3.12/site-packages/lmcache/v1/gpu_connector/utils.py \
  /opt/venv/lib/python3.12/site-packages/lmcache/v1/platform/cuda/cache_context.py
```

All mounts should be `ro`.

## LMCache chunk-size mismatch for another DCP size

Do not blindly copy 12,288. Capture each engine group's `tokens_per_block` from startup diagnostics. The selected chunk must be a multiple of all of them.

For the documented layout only:

```text
1536 scheduler tokens × DCP size = effective LMCache block
```

Changing model architecture, interleave, block size or connector logic can change this relationship.

## Short requests produce no L2 object

LMCache stores complete chunks. With `chunk_size=12288`, a request below 12,288 reusable prefix tokens can result in zero complete objects. This is expected. Use `cache_probe.py`, read `prompt_tokens`, and increase `--characters` until it exceeds one chunk.

## L2 files exist but restore does not hit

Check, in order:

1. The second prompt SHA is identical.
2. Model ID, tokenizer, chat template and request prefix are identical.
3. The mounted L2 directory is unchanged and visible after restart.
4. L2 writes completed before shutdown.
5. Object-group separation is still enabled.
6. No cache ABI/model configuration changed between runs.
7. vLLM metrics report `external_kv_transfer`, not only an internal prefix hit.
8. LMCache logs show L2 prefetch and per-rank loads.

Changing one system prompt byte changes the cache key.

## O_DIRECT or filesystem errors

The example uses `fs_native` with `use_odirect=true`. Verify:

```bash
stat -f -c '%T' "$LMCACHE_L2_PATH"
df -h "$LMCACHE_L2_PATH"
touch "$LMCACHE_L2_PATH/.write-test" && rm "$LMCACHE_L2_PATH/.write-test"
```

Use a local filesystem with appropriate O_DIRECT support and sufficient space. Network filesystems, unusual mount options, or unaligned custom adapters need separate qualification. Do not silently turn O_DIRECT off and claim equivalent performance.

## Host RAM pressure

The example reserves 48 GB for L1. Check:

```bash
free -h
curl -fsS http://127.0.0.1:8088/status | python3 -m json.tool
```

LMCache L1 is not free memory. Leave headroom for vLLM workers, page cache, Docker, compilation and operating-system allocations. If reducing L1, retest eviction, L2 flush and long-session behavior.

## LMCache server still appears on every GPU

Expected in pointer/CUDA IPC mode. The server needs GPU visibility to open IPC handles and run transfer kernels. The process-accounting number includes shared mappings.

Verify the staging configuration is present in the server process environment/log:

```text
CUDA_MODULE_LOADING=LAZY
LMCACHE_MP_GPU_STAGING_BATCH_SIZE=1
```

Do not set `CUDA_VISIBLE_DEVICES=""`; that does not transform pointer mode into CPU-only transfer.

## LMCache VRAM did not decrease

Check that the mounted `cache_context.py` contains `LMCACHE_MP_GPU_STAGING_BATCH_SIZE` and the server was actually recreated. Compare the LMCache PID, not a stale process. A one-chunk staging batch changes real temporary allocation, while the shared 576 MiB mapping remains visible.

## Why not switch to `engine_driven`?

The installed implementation does not safely support the four hybrid cache groups produced by this Kimi K3 setup. Do not use a closed or experimental multi-group patch as a drop-in production fix. It needs independent protocol, lock, pinned-memory, stale-copy, DCP and output-correctness validation.

## Why not `pip install -U lmcache`?

The image uses `0.5.2+glm52dcp.5`, not stock 0.5.2. A whole-package upgrade can replace:

- custom DCP group math,
- native extension symbols,
- `EngineKVFormat` enums,
- connector protocol behavior.

The repository's narrow overlay exists specifically to avoid that uncontrolled replacement. For a clean new image, prefer a fully matched recent LMCache/vLLM pair rather than layering this backport.

## Status JSON field errors in custom scripts

Build-specific `/status` responses may not contain names such as `object_count`, `registered_kv_cache_ids`, or `current_size_memory`, or may expose integers where a script assumed arrays. First print the real JSON:

```bash
curl -fsS http://127.0.0.1:8088/status | python3 -m json.tool
```

Treat a client-side `KeyError` or `TypeError` separately from server health.

## CUDA OOM during startup

Do not lower memory settings randomly. Capture:

```bash
nvidia-smi
free -h
docker logs --timestamps kimik3 > /tmp/kimik3-oom.log 2>&1
```

Distinguish model weights, vLLM active KV, CUDA graphs, LMCache staging, and shared IPC attribution. The tested explicit active-KV value is 603,979,776 bytes (576 MiB) per rank. Different weights, graphs or GPUs require a new memory budget.

## Reporting an issue

Include sanitized versions of:

- `manifest.json` checker output,
- exact image ID and package versions,
- Compose command and non-secret environment values,
- first complete traceback,
- cache-group edit summary,
- all engine-group `tokens_per_block` values,
- `/status` keys and relevant metrics,
- `nvidia-smi` process accounting,
- prompt SHA/token counts for miss and restore.

Never attach `.env`, access tokens, model files or raw private prompts.
