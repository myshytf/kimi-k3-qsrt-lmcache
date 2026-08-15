from __future__ import annotations

import importlib.util
import logging
import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "overlays/lmcache/integration/vllm/kv_cache_groups.py"


def _load_overlay():
    saved = {
        name: sys.modules.get(name)
        for name in (
            "lmcache",
            "lmcache.logging",
            "lmcache.v1",
            "lmcache.v1.multiprocess",
            "lmcache.v1.multiprocess.group_view",
        )
    }
    package = types.ModuleType("lmcache")
    package.__path__ = []
    logging_module = types.ModuleType("lmcache.logging")
    logging_module.init_logger = logging.getLogger
    v1 = types.ModuleType("lmcache.v1")
    v1.__path__ = []
    multiprocess = types.ModuleType("lmcache.v1.multiprocess")
    multiprocess.__path__ = []
    group_view = types.ModuleType("lmcache.v1.multiprocess.group_view")
    group_view.EngineGroupInfo = object
    sys.modules.update(
        {
            "lmcache": package,
            "lmcache.logging": logging_module,
            "lmcache.v1": v1,
            "lmcache.v1.multiprocess": multiprocess,
            "lmcache.v1.multiprocess.group_view": group_view,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "qualified_kv_cache_groups", MODULE_PATH
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@dataclass
class FullAttentionSpec:
    block_size: int
    dcp_kv_shard_count: int | None = None


@dataclass
class MambaSpec:
    block_size: int
    mamba_cache_mode: str = "align"


@dataclass
class UniformTypeKVCacheSpecs:
    block_size: int
    kv_cache_specs: dict[str, object] = field(default_factory=dict)


@dataclass
class Group:
    layer_names: list[str]
    kv_cache_spec: object


class MambaGeometryPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_overlay()

    def test_mamba_span_is_replicated_while_attention_is_sharded(self) -> None:
        self.assertEqual(
            self.module.effective_tokens_per_block(MambaSpec(16), dcp_size=8),
            16,
        )
        self.assertEqual(
            self.module.effective_tokens_per_block(
                UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={"mamba.0": MambaSpec(16)},
                ),
                dcp_size=8,
            ),
            16,
        )
        self.assertEqual(
            self.module.effective_tokens_per_block(FullAttentionSpec(16), dcp_size=8),
            128,
        )
        self.assertEqual(
            self.module.effective_tokens_per_block(
                FullAttentionSpec(16, dcp_kv_shard_count=2), dcp_size=8
            ),
            32,
        )

    def test_align_window_resolves_through_uniform_wrapper(self) -> None:
        groups = [
            Group(
                ["mamba.0", "mamba.1"],
                UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={
                        "mamba.0": MambaSpec(16, "align"),
                        "mamba.1": MambaSpec(16, "none"),
                    },
                ),
            )
        ]
        self.assertEqual(
            self.module._resolve_per_layer_sw_sizes(
                groups,
                {"mamba.0": 0, "mamba.1": 1},
                2,
            ),
            [16, -1],
        )


if __name__ == "__main__":
    unittest.main()
