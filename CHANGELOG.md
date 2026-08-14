# Changelog

All notable changes to this project are documented here.

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
