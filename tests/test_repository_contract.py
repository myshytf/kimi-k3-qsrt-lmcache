from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_required_public_files_exist(self) -> None:
        required = [
            "README.md",
            "LICENSE",
            "NOTICE",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "docs/COMPATIBILITY.md",
            "docs/ROOT_CAUSE.md",
            "docs/INSTALL.md",
            "docs/VALIDATION.md",
            "docs/TROUBLESHOOTING.md",
            "docs/VRAM_ACCOUNTING.md",
            "examples/compose.yml",
            "examples/.env.example",
            ".github/workflows/ci.yml",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_root_cause_document_covers_startup_and_transfer_failures(self) -> None:
        text = (ROOT / "docs" / "ROOT_CAUSE.md").read_text(encoding="utf-8")
        required = [
            "Failed to promote local KV cache specs to one unified type",
            "expected a Mamba [conv_state, ssm_state] tensor list, got Tensor",
            "chunk_size (1536) must be a multiple of",
            "1536 × 8 = 12288",
            "stock engine-driven transfer could not serve Kimi K3",
        ]
        for phrase in required:
            self.assertIn(phrase, text)

    def test_compose_uses_read_only_overlays_and_keeps_hybrid_manager(self) -> None:
        text = (ROOT / "examples" / "compose.yml").read_text(encoding="utf-8")
        self.assertEqual(text.count("- ./patchwork/"), 16)
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        for artifact in manifest["artifacts"]:
            mount = (
                f"- ./{artifact['install_path']}:{artifact['container_path']}:ro"
            )
            self.assertIn(mount, text, artifact["name"])
        self.assertIn("K3_LMCACHE_CHUNK_SIZE: ${K3_LMCACHE_CHUNK_SIZE:-12288}", text)
        self.assertIn("K3_MAX_MODEL_LEN: ${K3_MAX_MODEL_LEN:-420000}", text)
        self.assertIn("K3_KV_CACHE_MEMORY_BYTES: ${K3_KV_CACHE_MEMORY_BYTES:-2147483648}", text)
        self.assertIn("K3_LMCACHE_TRANSFER_MODE: ${K3_LMCACHE_TRANSFER_MODE:-engine_driven}", text)
        self.assertIn('K3_LMCACHE_SERVER_ENV: "CUDA_VISIBLE_DEVICES= CUDA_MODULE_LOADING=LAZY"', text)
        self.assertNotIn("--disable-hybrid-kv-cache-manager", text)
        self.assertNotIn("/home/" + "g0san", text)

    def test_launcher_keeps_required_hybrid_settings(self) -> None:
        text = (
            ROOT / "launcher" / "serve-kimi-k3-qsrt-lmcache.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--disable-hybrid-kv-cache-manager", text)
        self.assertIn("--separate-object-groups", text)
        self.assertIn('--supported-transfer-mode "${K3_LMCACHE_TRANSFER_MODE}"', text)
        self.assertIn('"lmcache.mp.mp_transfer_mode":"%s"', text)
        self.assertIn('--mamba-cache-mode "${K3_MAMBA_CACHE_MODE:-align}"', text)

    def test_repository_contains_no_live_credentials_or_private_host_paths(self) -> None:
        patterns = {
            "hugging-face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
            "OpenAI-style secret": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
            "GitHub token": re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
            "Cloudflare token": re.compile(r"cfat_[A-Za-z0-9_-]{20,}"),
            "private home path": re.compile(r"/home/" + r"g0san(?:/|\b)"),
        }
        ignored_parts = {".git", "__pycache__"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or ignored_parts.intersection(path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for label, pattern in patterns.items():
                self.assertIsNone(
                    pattern.search(text),
                    f"{label} found in {path.relative_to(ROOT)}",
                )

    def test_every_overlay_has_a_modification_notice(self) -> None:
        for path in (ROOT / "overlays").rglob("*.py"):
            first_lines = "\n".join(path.read_text(encoding="utf-8").splitlines()[:8])
            self.assertIn("Modified by myshytf/kimi-k3-qsrt-lmcache", first_lines)


if __name__ == "__main__":
    unittest.main()
