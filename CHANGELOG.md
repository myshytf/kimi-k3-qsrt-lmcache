# Changelog

All notable changes to this project are documented here.

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
