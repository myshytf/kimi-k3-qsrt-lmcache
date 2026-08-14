# Kimi K3 QSRT + LMCache compatibility overlay

A reproducible, fail-closed compatibility kit for serving **Kimi K3 QSRT** with a vLLM 0.26 hybrid KV cache and an LMCache 0.5.2-derived multiprocess cache server.

This repository documents and packages the exact overlay that recovered an 8-GPU TP8/DCP8 deployment which failed at engine initialization after LMCache was enabled. It also reduces avoidable LMCache staging VRAM while preserving the production-safe CUDA IPC/pointer transfer path.

> [!IMPORTANT]
> This is a narrow backport for the exact image and package builds in [`manifest.json`](manifest.json), not a universal patch for arbitrary vLLM or LMCache releases. Run `scripts/check_image.py` before mounting any overlay. A base-file hash mismatch is a stop condition, not a warning to ignore.

## What was broken

Four separate problems appeared in sequence:

| Stage | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `Failed to promote local KV cache specs to one unified type` | `--disable-hybrid-kv-cache-manager` forced incompatible KDA/Mamba and MLA cache specs into one type | Keep vLLM's hybrid KV manager enabled |
| 2 | `expected a Mamba [conv_state, ssm_state] tensor list, got Tensor` | vLLM 0.26 exposes unified Mamba/KDA state as one tensor; the installed LMCache expected the older two-tensor representation | Backport unified-Mamba and subpaged-MLA group edits from the LMCache 0.5.3 line while preserving the custom DCP/native ABI |
| 3 | `chunk_size (1536) must be a multiple of ... tokens_per_block (12288)` | 1536 is the rank-local scheduler step, but LMCache sees a DCP-aware global block | Keep `max_num_batched_tokens=1536`; set LMCache chunk to `1536 × 8 = 12288` |
| 4 | LMCache server showed about 960 MiB on every GPU | Pointer mode allocated four reusable staging chunks per rank in addition to CUDA IPC mappings | Make staging batch configurable and set it to one chunk |

Read the complete failure chain in [docs/ROOT_CAUSE.md](docs/ROOT_CAUSE.md).

## Validated result

Validated on 2026-08-14:

- Model: [`lukealonso/Kimi-K3-QSRT-K2`](https://huggingface.co/lukealonso/Kimi-K3-QSRT-K2)
- GPUs: 8 × NVIDIA RTX PRO 6000 Blackwell Max-Q 96 GB
- Parallelism: TP8 / DCP8 / A2A / KV interleave 1
- vLLM: `0.26.1rc0+infernal.invocation.cu133.r7.vllm7ed814e.b12x5d648d9`
- LMCache: `0.5.2+glm52dcp.5`
- Python 3.12.3, PyTorch 2.13.0
- FP8 KV, `max_model_len=200000`, `max_num_seqs=2`
- Active vLLM KV reservation: 576 MiB per rank
- LMCache: 48 GB host-RAM L1 + `fs_native`/O_DIRECT L2

Observed validation:

| Check | Result |
|---|---:|
| Container startup | Running, restart count 0 |
| Direct vLLM and proxy requests | HTTP 200 |
| Reused tokens after L2-preserving restart | 12,288 / 18,103 (67.9%) |
| Full miss vs external restore | ~17.9 s vs ~5.95–6.24 s |
| L2 prefetch | 8/8 chunks |
| CPU-to-GPU load | 8/8 ranks, 5.23–8.73 GB/s per rank |
| LMCache server VRAM attribution | ~960 MiB/GPU → ~724–726 MiB/GPU |
| Avoided staging allocation | ~234–236 MiB/GPU |

The remaining ~724–726 MiB shown for the LMCache server is **not all duplicate allocation**. About 576 MiB is the already-existing vLLM KV tensor mapped into the server through CUDA IPC. See [docs/VRAM_ACCOUNTING.md](docs/VRAM_ACCOUNTING.md).

## Repository layout

```text
manifest.json                       Exact tested versions, paths and hashes
launcher/                           LMCache-aware Kimi K3 QSRT launcher
overlays/lmcache/                   Four narrow LMCache compatibility overlays
overlays/vllm/                      Tested B12X MLA DCP8 prerequisite overlay
examples/compose.yml                Credential-free TP8/DCP8 Compose example
examples/.env.example               Host-specific values to configure
scripts/check_image.py              Verifies image ID and all unpatched base hashes
scripts/install_overlays.py         Atomic installer with dry-run and backups
scripts/verify_runtime.py           Models + inference + LMCache status checks
docs/                               Root cause, install, validation and operations
```

All Python overlays are mounted read-only over the image's original files. The container image is not rebuilt or modified.

## Quick start

### 1. Clone

```bash
git clone https://github.com/myshytf/kimi-k3-qsrt-lmcache.git
cd kimi-k3-qsrt-lmcache
```

### 2. Verify the target image before applying anything

```bash
python3 scripts/check_image.py
```

Expected final line:

```text
COMPATIBLE: 6 artifact(s) matched
```

If any file is `MISMATCH` or `MISSING`, stop and use [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md). Do not force an overlay onto a different ABI.

### 3. Create a deployment directory and install the payload

```bash
export DEPLOY_DIR="$PWD/deployment"
mkdir -p "$DEPLOY_DIR"
python3 scripts/install_overlays.py --destination "$DEPLOY_DIR" --dry-run
python3 scripts/install_overlays.py --destination "$DEPLOY_DIR"
cp examples/compose.yml "$DEPLOY_DIR/compose.yml"
cp examples/.env.example "$DEPLOY_DIR/.env"
```

Edit `$DEPLOY_DIR/.env` and set at least:

- `MODEL_PATH`
- `KIMI_CACHE_PATH`
- `KIMI_TMP_PATH`
- `LMCACHE_L2_PATH`
- optionally `HF_TOKEN` if the local model/tokenizer needs authenticated Hub access

Never commit `.env`.

### 4. Validate and start

```bash
cd "$DEPLOY_DIR"
docker compose config --quiet
docker compose up -d
```

Model loading takes several minutes. Follow startup without discarding the first exception:

```bash
docker compose logs -f --timestamps kimik3
```

Then run the functional checks:

```bash
cd /path/to/kimi-k3-qsrt-lmcache
python3 scripts/verify_runtime.py \
  --vllm-url http://127.0.0.1:8090 \
  --lmcache-url http://127.0.0.1:8088
```

For persistent L2 proof, cache a prompt longer than 12,288 tokens, restart while retaining the L2 directory, repeat the exact prompt, and inspect external-cache metrics as described in [docs/VALIDATION.md](docs/VALIDATION.md).

## Settings that must remain distinct

```text
vLLM scheduler/local step = 1536 tokens
DCP KV shard count          = 8
LMCache global chunk        = 1536 × 8 = 12288 tokens
```

Do **not** change `max_num_batched_tokens` to 12,288 merely to match LMCache. Those values represent different units.

Required settings:

- hybrid KV cache manager: enabled (do not pass `--disable-hybrid-kv-cache-manager`)
- `--mamba-cache-mode align`
- LMCache `--separate-object-groups`
- `K3_MAX_NUM_BATCHED_TOKENS=1536`
- `K3_LMCACHE_CHUNK_SIZE=12288` for DCP8
- `CUDA_MODULE_LOADING=LAZY`
- `LMCACHE_MP_GPU_STAGING_BATCH_SIZE=1`

## Why not upgrade the whole LMCache package?

The tested image contains `lmcache 0.5.2+glm52dcp.5`, a custom DCP-aware build with native extensions and format enums tied to that image. Replacing it wholesale with upstream 0.5.3 could remove local DCP behavior or create a Python/native `EngineKVFormat` mismatch.

This repository therefore backports only the reviewed Python integration needed for:

1. vLLM 0.26 unified Mamba/KDA views,
2. subpaged MLA views,
3. mixed group format discovery,
4. DCP group information propagation, and
5. configurable GPU staging.

The upstream basis is [LMCache PR #4206](https://github.com/LMCache/LMCache/pull/4206), merged in the LMCache 0.5.3 development line. See [`NOTICE`](NOTICE) and [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Known limits

- The supplied payload is hash-qualified for one image build. Other versions require a fresh port and tests.
- The included B12X MLA file is a tested DCP8 prerequisite for the listed infernal-invocation image. It is separated as component `vllm-dcp` in the manifest.
- `engine_driven` transfer is not used. The installed implementation does not safely support Kimi K3's multiple hybrid cache groups.
- Pointer/CUDA IPC mode still needs GPU visibility. Clearing `CUDA_VISIBLE_DEVICES` is not a CPU-only mode and breaks registration.
- LMCache reduces reusable-prefix pressure by extending KV storage to RAM and disk; it does not offload model weights or eliminate vLLM's active GPU KV working set.
- A prompt shorter than one 12,288-token chunk may correctly create no reusable LMCache object.

## Rollback

The installer never modifies container site-packages. Remove the six read-only bind mounts and restore your previous launcher/Compose. If `--force` replaced an existing patchwork file, the installer creates `*.bak.<UTC timestamp>` beside it.

Full rollback steps are in [docs/INSTALL.md#rollback](docs/INSTALL.md#rollback).

## Documentation

- [Compatibility matrix and hash policy](docs/COMPATIBILITY.md)
- [Failure sequence and root cause](docs/ROOT_CAUSE.md)
- [Installation and rollback](docs/INSTALL.md)
- [Functional, L2 and VRAM validation](docs/VALIDATION.md)
- [VRAM accounting](docs/VRAM_ACCOUNTING.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## License and attribution

Apache License 2.0. The overlay includes modified portions of [LMCache](https://github.com/LMCache/LMCache) and [vLLM](https://github.com/vllm-project/vllm), both Apache-2.0 projects. Modified files carry notices and upstream provenance is recorded in [`NOTICE`](NOTICE) and [`manifest.json`](manifest.json).
