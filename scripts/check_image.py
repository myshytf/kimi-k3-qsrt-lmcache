#!/usr/bin/env python3
"""Fail-closed compatibility check for the tested Kimi K3 container image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifest.json"


def run_docker(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *arguments],
        text=True,
        capture_output=True,
        check=check,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Compatibility manifest (default: repository manifest.json).",
    )
    parser.add_argument(
        "--image",
        help="Container image to inspect (default: manifest tested image).",
    )
    parser.add_argument(
        "--component",
        action="append",
        choices=("launcher", "lmcache", "vllm-dcp"),
        help="Only check this component; repeat to select more than one.",
    )
    parser.add_argument(
        "--allow-image-id-mismatch",
        action="store_true",
        help="Continue to file checks when the image ID differs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise SystemExit(
            f"Unsupported manifest schema_version={manifest.get('schema_version')!r}; "
            "expected 2"
        )
    tested = manifest["tested_environment"]
    image = args.image or tested["container_image"]
    expected_image_id = tested.get("container_image_id")

    inspected = run_docker("image", "inspect", "--format", "{{.Id}}", image)
    actual_image_id = inspected.stdout.strip()
    if expected_image_id and actual_image_id != expected_image_id:
        print(
            f"IMAGE-ID MISMATCH expected={expected_image_id} "
            f"actual={actual_image_id}"
        )
        if not args.allow_image_id_mismatch:
            return 1
    else:
        print(f"IMAGE-ID MATCH {actual_image_id}")

    selected = [
        artifact
        for artifact in manifest["artifacts"]
        if not args.component or artifact["component"] in args.component
    ]
    if not selected:
        raise SystemExit("No artifacts selected")

    created = run_docker("create", "--entrypoint", "/bin/true", image)
    container_id = created.stdout.strip()
    if not container_id:
        raise SystemExit("docker create returned an empty container ID")

    failures = 0
    try:
        with tempfile.TemporaryDirectory(prefix="kimi-k3-image-check-") as directory:
            probe = Path(directory)
            for index, artifact in enumerate(selected):
                extracted = probe / f"{index:02d}-{Path(artifact['container_path']).name}"
                base_absent = bool(artifact.get("base_absent", False))
                copied = run_docker(
                    "cp",
                    f"{container_id}:{artifact['container_path']}",
                    str(extracted),
                    check=False,
                )
                if copied.returncode != 0:
                    detail = copied.stderr.strip() or copied.stdout.strip()
                    missing_error = any(
                        marker in detail.lower()
                        for marker in (
                            "could not find the file",
                            "no such file or directory",
                        )
                    )
                    if base_absent and missing_error:
                        print(f"MATCH-ABSENT {artifact['name']}")
                        continue
                    failures += 1
                    label = "COPY-ERROR" if base_absent else "MISSING"
                    print(f"{label} {artifact['name']}: {detail}")
                    continue

                if base_absent:
                    failures += 1
                    print(
                        f"UNEXPECTED-PRESENT {artifact['name']}: "
                        f"{artifact['container_path']} must be absent in the base image"
                    )
                    continue

                actual = sha256(extracted)
                expected = artifact["base_sha256"]
                if actual == expected:
                    print(f"MATCH {artifact['name']} {actual}")
                else:
                    failures += 1
                    print(
                        f"MISMATCH {artifact['name']} "
                        f"expected={expected} actual={actual}"
                    )
    finally:
        removed = run_docker("rm", "-f", container_id, check=False)
        if removed.returncode != 0:
            detail = removed.stderr.strip() or removed.stdout.strip()
            print(f"WARNING failed to remove probe container {container_id}: {detail}")

    if failures:
        print(f"INCOMPATIBLE: {failures} artifact(s) did not match")
        return 1
    print(f"COMPATIBLE: {len(selected)} artifact(s) matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
