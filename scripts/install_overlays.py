#!/usr/bin/env python3
"""Install the tested Kimi K3 LMCache overlay payloads."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifacts() -> list[dict[str, str]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    for artifact in artifacts:
        source = (ROOT / artifact["source"]).resolve()
        source.relative_to(ROOT.resolve())
        actual = sha256(source)
        expected = artifact["patched_sha256"]
        if actual != expected:
            raise SystemExit(
                f"Payload checksum mismatch for {artifact['name']}: "
                f"expected {expected}, got {actual}"
            )
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="Deployment directory that will receive patchwork/ files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the copy plan without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Back up and replace existing files whose content differs.",
    )
    return parser.parse_args()


def install(source: Path, destination: Path, backup_suffix: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and backup_suffix is not None:
        backup = destination.with_name(destination.name + backup_suffix)
        shutil.copy2(destination, backup)
        print(f"BACKUP {destination} -> {backup}")
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    artifacts = load_artifacts()

    planned: list[tuple[dict[str, str], Path, Path, bool]] = []
    for artifact in artifacts:
        source = (ROOT / artifact["source"]).resolve()
        destination_root = args.destination.resolve()
        destination = (destination_root / artifact["install_path"]).resolve()
        try:
            destination.relative_to(destination_root)
        except ValueError as error:
            raise SystemExit(
                f"Refusing destination outside deployment root: {destination}"
            ) from error
        differs = destination.exists() and sha256(destination) != artifact["patched_sha256"]
        planned.append((artifact, source, destination, differs))

    conflicts = [entry for entry in planned if entry[3]]
    if conflicts and not args.force:
        paths = ", ".join(str(entry[2]) for entry in conflicts)
        raise SystemExit(
            "Refusing to overwrite existing modified file(s): "
            f"{paths}. Re-run with --force to create backups first."
        )

    backup_suffix = ".bak." + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")

    for artifact, source, destination, differs in planned:
        if args.dry_run:
            action = "BACKUP+INSTALL" if differs else "INSTALL"
            print(
                f"PLAN {action} {artifact['name']}: "
                f"{artifact['source']} -> {destination}"
            )
        elif destination.exists() and not differs:
            print(f"UNCHANGED {artifact['name']}: {destination}")
        else:
            install(source, destination, backup_suffix if differs else None)
            print(f"INSTALLED {artifact['name']}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
