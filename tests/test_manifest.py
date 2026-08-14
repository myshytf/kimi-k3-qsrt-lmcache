from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.json"


class ManifestContractTests(unittest.TestCase):
    def test_manifest_is_self_consistent(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["tested_environment"]["gpu_count"], 8)
        self.assertEqual(data["tested_environment"]["dcp_size"], 8)

        artifacts = data["artifacts"]
        self.assertGreaterEqual(len(artifacts), 6)
        self.assertEqual(
            {artifact["component"] for artifact in artifacts},
            {"launcher", "lmcache", "vllm-dcp"},
        )

        seen_container_paths: set[str] = set()
        for artifact in artifacts:
            source = (ROOT / artifact["source"]).resolve()
            source.relative_to(ROOT.resolve())
            self.assertTrue(source.is_file(), artifact["source"])

            install_path = Path(artifact["install_path"])
            self.assertFalse(install_path.is_absolute())
            self.assertNotIn("..", install_path.parts)

            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(digest, artifact["patched_sha256"])
            self.assertRegex(artifact["base_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(artifact["patched_sha256"], r"^[0-9a-f]{64}$")

            container_path = artifact["container_path"]
            self.assertTrue(container_path.startswith("/"))
            self.assertNotIn(container_path, seen_container_paths)
            seen_container_paths.add(container_path)


if __name__ == "__main__":
    unittest.main()
