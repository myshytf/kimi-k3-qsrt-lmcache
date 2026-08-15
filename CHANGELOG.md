# Changelog

All notable changes to this project are documented here.

## [0.3.0] - 2026-08-16

### Fixed

- Complete recurrent destination aliases now restore only the newest logical snapshot in both pickle and SHM engine-driven paths, preventing concurrent H2D writes to one physical block.
- Mamba/KDA recurrent cache geometry now remains DCP-replicated instead of being expanded as sequence-sharded attention KV.
- `mamba_cache_mode=align` exposes a one-block tail window; non-align Mamba and attention groups retain their existing semantics.

### Added

- Hash-qualified `kv_cache_groups.py` overlay and focused direct/uniform/nested Mamba geometry tests.
- Regression coverage for complete aliases, partial aliases, token skips, invalid geometry, and unique attention destinations.
- Immutable image, layout fingerprint, upstream PR, and restarted L2 qualification provenance in `manifest.json`.
- Corrected the stale base-image hash for `vllm_multi_process_adapter.py`; the manifest now matches the pinned image ID's actual file bytes.

### Validated

- Cold, same-process local, and 73,728-token restarted external outputs are equivalent.
- A suffix-changed request restored 36,864 of 40,259 prompt tokens after restart.
- Two-round tool continuation and two simultaneous forced-tool streams passed through the 5318 gateway.
- The qualified immutable container remained running with restart count 0 and `OOMKilled=false`.

## [0.2.1] - 2026-08-14

### Fixed

- Runtime verification now checks LMCache's canonical `registered_gpu_ids` field, retains the legacy alias, and fails closed when both disagree.
- Standalone launcher KV reservation now defaults to the validated 2 GiB/rank profile instead of the older 3.6 GiB value.
- LMCache server environment assignments are validated and passed as a Bash array instead of unquoted word splitting.

### Security

- Documented that the unauthenticated pickle/ZeroMQ MP transport must remain loopback-only.

## [0.2.0] - 2026-08-14

### Added

- Ten hash-qualified LMCache multiprocess overlays based on PR #4410.
- Multi-group engine-driven registration for Kimi K3's four hybrid object groups.
- DCP8 logical 12,288-token to physical 1,536-token rank-local transfer conversion.
- Fail-closed `base_absent` manifest/checker support for newly introduced files.
- Regression coverage for expected-absent and unexpected-present base paths.

### Changed

- Standalone LMCache server is now CPU-only; GPU gather/scatter runs in the existing vLLM workers.
- Default validated profile is `max_model_len=420000` with a 2 GiB/rank active KV reservation.
- Public Compose template mounts sixteen immutable artifacts and negotiates `engine_driven` explicitly.

### Validated

- Eight of eight non-CUDA transfer contexts registered.
- Standalone LMCache GPU process attribution reduced from ~724–726 MiB/GPU to 0 MiB/GPU.
- GPU KV capacity reached 1,034,634 tokens (2.46× at 420K).
- Persistent L2 restore remained functional with 12,288 external tokens restored.

## [0.1.0] - 2026-08-14

### Added

- Hash-qualified compatibility manifest for the tested infernal-invocation image.
- Read-only LMCache overlays for vLLM 0.26 unified Mamba/KDA and subpaged MLA cache groups.
- Mixed KDA/MLA format discovery while preserving custom DCP group calculations.
- Configurable LMCache MP GPU staging batch with a tested minimum of one.
- B12X MLA DCP8 prerequisite overlay.
- LMCache-aware launcher using hybrid cache groups, `mamba-cache-mode=align`, separate object groups and a 12,288-token DCP8 chunk.
- Atomic installer with dry-run, checksum verification, overwrite refusal and timestamped backups.
- Fail-closed Docker image/base-file compatibility checker.
- Runtime and deterministic long-prefix verification tools.
- Detailed root-cause, compatibility, installation, validation, VRAM and troubleshooting documentation.
- Regression tests and GitHub Actions CI.

### Validated

- TP8/DCP8 startup and short inference.
- L2-preserving restart with 12,288 external tokens restored from an 18,103-token prompt.
- LMCache staging VRAM attribution reduced from roughly 960 MiB/GPU to 724–726 MiB/GPU.
