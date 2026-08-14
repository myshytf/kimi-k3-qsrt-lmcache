# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's **Report a vulnerability** feature for this repository. Do not open a public issue containing credentials, private model prompts, host inventory or exploitable deployment details.

Include:

- affected commit and image/version,
- impact and reproduction steps,
- whether the issue requires local Docker access,
- a minimal sanitized proof of concept.

## Supported version

Only the latest commit on `main` is maintained. The compatibility payload itself is intentionally qualified for the exact versions in `manifest.json`.

## Deployment security notes

- `examples/compose.yml` mirrors the privileges of the validated high-performance deployment and uses `privileged: true`, host networking, host IPC and all GPUs. Review and reduce these permissions for your threat model before exposing the service.
- Bind API ports to a trusted network or put them behind authenticated access control.
- Keep `.env` mode 0600 and never commit tokens.
- Treat model files and LMCache L2 objects as sensitive if prompts or derived KV state are sensitive.
- The engine-driven MP transport uses Python pickle over unauthenticated ZeroMQ. Keep `K3_LMCACHE_MP_HOST=127.0.0.1`, do not publish port 5555, and allow only trusted local processes to reach it; exposing it to an untrusted peer is equivalent to remote code execution.
- Read-only source overlays reduce mutation risk but do not make an untrusted image safe.
- Run `scripts/check_image.py` against a trusted local image before mounting overlays.
- Review every overlay and its upstream provenance before production use.

## Secret handling

Repository tests reject common live-token patterns and private author paths. This is a guardrail, not a substitute for a dedicated secret scanner or credential rotation after accidental disclosure.
