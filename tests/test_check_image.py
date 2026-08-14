from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_image.py"


class ImageCompatibilityTests(unittest.TestCase):
    def _run(self, payload: bytes) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            rootfs = work / "rootfs"
            target = rootfs / "opt" / "fixture.py"
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)

            expected = hashlib.sha256(b"expected base\n").hexdigest()
            manifest = {
                "schema_version": 1,
                "tested_environment": {
                    "container_image": "fixture:test",
                    "container_image_id": "sha256:fixture",
                },
                "artifacts": [
                    {
                        "name": "fixture",
                        "component": "lmcache",
                        "source": "unused.py",
                        "install_path": "patchwork/unused.py",
                        "container_path": "/opt/fixture.py",
                        "base_sha256": expected,
                        "patched_sha256": "0" * 64,
                    }
                ],
            }
            manifest_path = work / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            docker = work / "docker"
            docker.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import os
                    from pathlib import Path
                    import shutil
                    import sys

                    args = sys.argv[1:]
                    log = Path(os.environ["FAKE_DOCKER_LOG"])
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write(" ".join(args) + "\\n")
                    if args[:2] == ["image", "inspect"]:
                        print("sha256:fixture")
                    elif args and args[0] == "create":
                        print("fixture-container")
                    elif args and args[0] == "cp":
                        source = args[1].split(":", 1)[1].lstrip("/")
                        shutil.copy2(Path(os.environ["FAKE_ROOTFS"]) / source, args[2])
                    elif args[:2] == ["rm", "-f"]:
                        pass
                    else:
                        raise SystemExit(2)
                    """
                ),
                encoding="utf-8",
            )
            docker.chmod(0o755)
            log = work / "docker.log"
            env = os.environ.copy()
            env["PATH"] = str(work) + os.pathsep + env["PATH"]
            env["FAKE_ROOTFS"] = str(rootfs)
            env["FAKE_DOCKER_LOG"] = str(log)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--manifest",
                    str(manifest_path),
                    "--image",
                    "fixture:test",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )
            result.docker_log = (  # type: ignore[attr-defined]
                log.read_text(encoding="utf-8") if log.exists() else ""
            )
            return result

    def test_matching_image_passes_and_removes_probe_container(self) -> None:
        result = self._run(b"expected base\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MATCH fixture", result.stdout)
        self.assertIn("rm -f fixture-container", result.docker_log)  # type: ignore[attr-defined]

    def test_base_hash_mismatch_fails_closed(self) -> None:
        result = self._run(b"different base\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MISMATCH fixture", result.stdout)
        self.assertIn("rm -f fixture-container", result.docker_log)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
