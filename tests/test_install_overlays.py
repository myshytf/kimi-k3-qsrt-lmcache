from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install_overlays.py"
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))


class InstallOverlayTests(unittest.TestCase):
    def test_dry_run_lists_every_artifact_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--destination",
                    str(destination),
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for artifact in MANIFEST["artifacts"]:
                self.assertIn(artifact["name"], result.stdout)
            self.assertEqual(list(destination.iterdir()), [])

    def test_dry_run_refuses_existing_different_file_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            artifact = MANIFEST["artifacts"][0]
            installed = destination / artifact["install_path"]
            installed.parent.mkdir(parents=True)
            installed.write_bytes(b"local modification")

            def run(*extra: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--destination",
                        str(destination),
                        *extra,
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )

            result = run("--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(installed.read_bytes(), b"local modification")

            forced = run("--dry-run", "--force")
            self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertIn("PLAN BACKUP+INSTALL", forced.stdout)
            self.assertEqual(installed.read_bytes(), b"local modification")

    def test_install_copies_every_manifest_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--destination", str(destination)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for artifact in MANIFEST["artifacts"]:
                source = ROOT / artifact["source"]
                installed = destination / artifact["install_path"]
                self.assertEqual(installed.read_bytes(), source.read_bytes())
                self.assertIn(artifact["name"], result.stdout)

    def test_existing_different_file_is_not_overwritten_without_force(self) -> None:
        artifact = MANIFEST["artifacts"][0]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            installed = destination / artifact["install_path"]
            installed.parent.mkdir(parents=True)
            installed.write_bytes(b"local customization\n")

            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--destination", str(destination)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(installed.read_bytes(), b"local customization\n")

    def test_force_backs_up_existing_file_before_replacing_it(self) -> None:
        artifact = MANIFEST["artifacts"][0]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            installed = destination / artifact["install_path"]
            installed.parent.mkdir(parents=True)
            installed.write_bytes(b"local customization\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--destination",
                    str(destination),
                    "--force",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                installed.read_bytes(), (ROOT / artifact["source"]).read_bytes()
            )
            backups = list(installed.parent.glob(installed.name + ".bak.*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b"local customization\n")


if __name__ == "__main__":
    unittest.main()
