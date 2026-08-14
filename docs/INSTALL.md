# Installation and rollback

## Prerequisites

- Linux host with the NVIDIA driver and NVIDIA Container Toolkit
- Docker Engine with Compose v2
- Eight visible GPUs for the exact documented TP8/DCP8 profile
- The tested container image available locally
- A complete Kimi K3 QSRT checkpoint
- Host RAM for LMCache L1 (48 GB in the example)
- An XFS/ext4-backed L2 directory with adequate space and O_DIRECT support
- Python 3.11+ on the host for repository scripts

No Python package is installed on the host, and the container image is not rebuilt.

## 1. Clone and inspect

```bash
git clone https://github.com/myshytf/kimi-k3-qsrt-lmcache.git
cd kimi-k3-qsrt-lmcache
python3 scripts/check_image.py
```

Do not continue unless the final result is `COMPATIBLE` and all sixteen artifacts match, including the declared absence of the new `transfer_plan.py` base path.

## 2. Prepare host directories

Example only; choose paths appropriate for your host:

```bash
export DEPLOY_DIR=/opt/kimi-k3-lmcache
sudo mkdir -p "$DEPLOY_DIR"
sudo chown "$USER":"$USER" "$DEPLOY_DIR"
mkdir -p /srv/kimi-k3/cache /srv/kimi-k3/tmp
mkdir -p /srv/lmcache/kimi-k3/l2 /srv/lmcache/kimi-k3/tmp
```

The model directory should already exist. Do not place model shards or L2 cache objects inside the Git checkout.

## 3. Install overlay payloads

Preview:

```bash
python3 scripts/install_overlays.py \
  --destination "$DEPLOY_DIR" \
  --dry-run
```

Install:

```bash
python3 scripts/install_overlays.py \
  --destination "$DEPLOY_DIR"
```

Behavior:

- repository payload hashes are verified first,
- target paths are constrained below the destination,
- files are copied atomically,
- identical files are reported as `UNCHANGED`,
- different existing files are never overwritten by default.

To intentionally replace existing patchwork files:

```bash
python3 scripts/install_overlays.py \
  --destination "$DEPLOY_DIR" \
  --force
```

Every replaced file is copied to `filename.bak.<UTC timestamp>` first.

## 4. Add the credential-free deployment templates

```bash
cp examples/compose.yml "$DEPLOY_DIR/compose.yml"
cp examples/.env.example "$DEPLOY_DIR/.env"
chmod 0600 "$DEPLOY_DIR/.env"
```

Edit `.env`. Required host-specific fields:

```dotenv
MODEL_PATH=/path/to/Kimi-K3-QSRT-K2
KIMI_CACHE_PATH=/path/to/kimi/cache
KIMI_TMP_PATH=/path/to/kimi/tmp
LMCACHE_L2_PATH=/path/to/lmcache/l2
LMCACHE_TMP_PATH=/path/to/lmcache/tmp
```

Set `HF_TOKEN` only if runtime tokenizer/config access requires it. The example contains no token and `.gitignore` excludes `.env`.

## 5. Validate before loading 700+ GB of weights

```bash
cd "$DEPLOY_DIR"
bash -n patchwork/serve-kimi-k3-qsrt-lmcache.sh
python3 -m py_compile \
  patchwork/lmcache/integration/vllm/kv_cache_group_edits.py
python3 -m compileall -q patchwork/lmcache patchwork/vllm
docker compose config --quiet
```

Review rendered mounts without dumping environment secrets into a shared log:

```bash
docker compose config --services
docker compose config --volumes
```

## 6. Start

```bash
docker compose up -d
```

Preserve the first failure if startup exits:

```bash
docker inspect kimik3 \
  --format 'status={{.State.Status}} exit={{.State.ExitCode}} restarts={{.RestartCount}}'
docker compose logs --no-color --timestamps kimik3 > startup.log
```

The launcher starts a CPU-only LMCache server in multi-group `engine_driven` mode, waits up to 120 seconds for `ZMQ cache server is running`, and only then executes vLLM. Model initialization can take several additional minutes.

## 7. Verify

```bash
python3 /path/to/kimi-k3-qsrt-lmcache/scripts/verify_runtime.py \
  --vllm-url http://127.0.0.1:8090 \
  --lmcache-url http://127.0.0.1:8088
```

Continue with persistent L2 and VRAM validation in [VALIDATION.md](VALIDATION.md).

## Updating this repository payload

```bash
cd /path/to/kimi-k3-qsrt-lmcache
git pull --ff-only
python3 scripts/check_image.py
python3 scripts/install_overlays.py --destination "$DEPLOY_DIR" --dry-run
python3 scripts/install_overlays.py --destination "$DEPLOY_DIR" --force
```

Re-run static checks before recreating the container.

## Rollback

### Fast rollback to pre-overlay deployment

1. Stop the service gracefully:

   ```bash
   cd "$DEPLOY_DIR"
   docker compose stop --timeout 60
   ```

2. Restore your previous Compose and launcher, or remove these read-only mount targets:

   ```text
   /usr/local/bin/serve-kimi-k3-qsrt.sh
   /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/kv_cache_group_edits.py
   /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/lmcache_mp_connector.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/gpu_connector/utils.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/platform/cuda/cache_context.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/custom_types.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/modules/engine_driven_transfer.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/modules/server_transfer.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/transfer_context/async_engine_driven.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/transfer_context/base.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/transfer_context/pickle.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/transfer_context/shm.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/transfer_context/worker_transfer.py
   /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/transfer_plan.py
   /opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/b12x_mla.py
   ```

3. If `--force` was used, restore the desired `*.bak.<UTC timestamp>` files.
4. Recreate with the previous configuration:

   ```bash
   docker compose up -d --force-recreate
   ```

### L2 cache handling

The L2 directory is not modified by installing/removing source overlays. Preserve it during a normal rollback. Delete it only when intentionally invalidating cache objects after a model, tokenizer, cache format, or incompatible layout change.

Never bind an old L2 object store to a different cache ABI without a clean-cache validation.
