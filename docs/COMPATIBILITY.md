# Compatibility and version policy

## Exact tested target

The source of truth is [`../manifest.json`](../manifest.json).

| Component | Tested value |
|---|---|
| Container image | `voipmonitor/vllm:infernal-invocation-vllm7ed814e-b12x5d648d9-fi1ac6942-cu133-torch213-20260813-r7` |
| Image ID | `sha256:58568d18ac87bf79095c758f5bc985f3b7a00d133819bf5bd47b935038f3f759` |
| Python | 3.12.3 |
| PyTorch | 2.13.0 |
| vLLM | `0.26.1rc0+infernal.invocation.cu133.r7.vllm7ed814e.b12x5d648d9` |
| LMCache | `0.5.2+glm52dcp.5` |
| Model | `lukealonso/Kimi-K3-QSRT-K2` |
| GPUs | 8 × RTX PRO 6000 Blackwell Max-Q 96 GB |
| Parallelism | TP8, DCP8, A2A, interleave 1 |
| Scheduler step | 1536 tokens |
| LMCache chunk | 12,288 tokens |
| Maximum model length | 420,000 tokens |
| Active KV reservation | 2 GiB per rank |
| Reported KV capacity | 1,034,634 tokens; 2.46× at 420K |
| Transfer mode | Multi-group `engine_driven`; CPU-only standalone server |

## Fail-closed base hashes

Each artifact in the manifest has two hashes:

- `base_sha256`: the file inside the unmodified tested image
- `patched_sha256`: the read-only overlay supplied by this repository

An artifact added by the backport and absent from the tested base image uses
`base_absent: true` instead of a fabricated base hash. Compatibility then
requires the path to be absent; an unexpected file at that path fails closed.

Run:

```bash
python3 scripts/check_image.py
```

The checker:

1. compares the local Docker image ID,
2. creates a stopped probe container with `/bin/true`,
3. extracts each target file with `docker cp`,
4. compares all base SHA-256 values or verifies declared base-file absence, and
5. removes the probe container in `finally` on success or failure.

A mismatch means the code surrounding the patch may have changed. Do not use `--allow-image-id-mismatch` unless you are intentionally auditing a rebuilt image; even then, every file hash still has to match.

## Components

| Manifest component | Purpose | Required for tested deployment |
|---|---|---:|
| `launcher` | Starts LMCache MP, waits for readiness, and launches vLLM with hybrid-safe settings | Yes |
| `lmcache` | Unified Mamba, subpaged MLA, mixed-format/DCP registration, pointer fallback staging, and multi-group engine-driven transfer | Yes |
| `vllm-dcp` | B12X MLA DCP planning/metadata for this image | Yes for the documented TP8/DCP8 profile |

The B12X overlay is separated because it is a serving prerequisite, not the LMCache root-cause fix itself.

## Upstream relationship

The unified-cache compatibility work is based on:

- [LMCache PR #4206 — vLLM 0.26 unified Mamba support](https://github.com/LMCache/LMCache/pull/4206)
- merge commit `f1ab19a148bf666b79fc2ce0babdb67dd637b430`
- LMCache v0.5.3 repository commit `140819c9d57a975dbc5678a6459a218e544cb58b`

The engine-driven work is based on:

- [LMCache PR #4410 — multi-group engine-driven transfer](https://github.com/LMCache/LMCache/pull/4410)
- reviewed PR-head commit `0cc4b50d2ff67d79fd29d36ef2467dbb1af4f7f5`
- a local Kimi K3 adaptation from 12,288 logical DCP tokens to 1,536
  physical rank-local blocks per transfer group

The files are not copied wholesale from current `dev`. They combine the required upstream integration with the tested image's custom DCP behavior and native format ABI.

## What may work but is not claimed

- Other RTX Blackwell models with the same CUDA/image build
- Other TP/DCP sizes after recalculating the effective block
- A rebuilt image whose file hashes remain identical
- Other Kimi K3 QSRT checkpoints with the same cache structure

These require fresh startup, inference, persistent-L2 and output-correctness tests.

## Explicitly unsupported without a new port

- Stock LMCache 0.5.2 without this image's DCP extension
- Arbitrary upstream LMCache 0.5.3+ native extensions mixed with these Python files
- vLLM versions whose unified cache specs or connector API differ
- Engine-driven layouts other than the tested four Kimi K3 object groups
- Running the standalone server CPU-only while negotiating pointer/CUDA IPC mode
- `NH != 1` unified KDA tensors using the compatibility flattening
- DCP sizes other than 8 while retaining `K3_LMCACHE_CHUNK_SIZE=12288`

## Recalculate chunk size for another DCP size

Use the engine group's actual `tokens_per_block` from startup diagnostics. For the documented layout:

```text
LMCache chunk = scheduler/local recurrent step × DCP KV shard count
```

Examples if the local step stays 1536:

| DCP | Candidate effective block |
|---:|---:|
| 1 | 1536 |
| 2 | 3072 |
| 4 | 6144 |
| 8 | 12288 |

These are calculations, not validation claims. Confirm every LMCache engine group's reported `tokens_per_block`; the selected chunk must be a multiple of all groups.

## Porting checklist for a different build

1. Extract clean base files from the new image.
2. Diff its connector and `EngineKVFormat` implementation against the tested base and current LMCache upstream.
3. Confirm vLLM cache spec kinds and raw tensor shapes at registration.
4. Verify unified KDA has `NH == 1` before using the 3-D view adapter.
5. Preserve any local DCP calculation in `create_engine_group_infos_from_vllm()`.
6. Add new base and patched hashes to a separate manifest revision.
7. For engine-driven mode, verify every group's logical and rank-local block geometry.
8. Run synthetic pointer and engine-driven shape/transfer tests.
9. Run full model startup, short inference, long-prefix L2 restart, and output-correctness tests.
10. Record the exact image digest, not only a mutable tag.
