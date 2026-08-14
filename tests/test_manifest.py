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
        self.assertEqual(data["schema_version"], 2)
        tested = data["tested_environment"]
        self.assertEqual(tested["gpu_count"], 8)
        self.assertEqual(tested["dcp_size"], 8)
        self.assertEqual(tested["max_model_len"], 420000)
        self.assertEqual(tested["max_num_seqs"], 2)
        self.assertEqual(tested["kv_cache_memory_bytes_per_rank"], 2147483648)
        self.assertEqual(tested["reported_gpu_kv_cache_tokens"], 1034634)
        self.assertEqual(tested["lmcache_transfer_mode"], "engine_driven")
        self.assertEqual(tested["registered_non_cuda_contexts"], 8)
        self.assertEqual(tested["registered_gpu_contexts"], 0)

        artifacts = data["artifacts"]
        self.assertEqual(len(artifacts), 16)
        self.assertEqual(
            {artifact["component"] for artifact in artifacts},
            {"launcher", "lmcache", "vllm-dcp"},
        )

        seen_container_paths: set[str] = set()
        absent_bases = 0
        for artifact in artifacts:
            source = (ROOT / artifact["source"]).resolve()
            source.relative_to(ROOT.resolve())
            self.assertTrue(source.is_file(), artifact["source"])

            install_path = Path(artifact["install_path"])
            self.assertFalse(install_path.is_absolute())
            self.assertNotIn("..", install_path.parts)

            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(digest, artifact["patched_sha256"])
            if artifact.get("base_absent"):
                absent_bases += 1
                self.assertNotIn("base_sha256", artifact)
            else:
                self.assertRegex(artifact["base_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(artifact["patched_sha256"], r"^[0-9a-f]{64}$")

            container_path = artifact["container_path"]
            self.assertTrue(container_path.startswith("/"))
            self.assertNotIn(container_path, seen_container_paths)
            seen_container_paths.add(container_path)

        self.assertEqual(absent_bases, 1)


if __name__ == "__main__":
    unittest.main()
