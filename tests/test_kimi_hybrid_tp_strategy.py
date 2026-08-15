from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import types
from typing import Any, Callable, cast
import unittest


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "overlays/lmcache/integration/vllm/utils.py"
ADAPTER = ROOT / "overlays/lmcache/integration/vllm/vllm_multi_process_adapter.py"
CONNECTOR = ROOT / "overlays/lmcache/integration/vllm/lmcache_mp_connector.py"


def _exec_definitions(
    path: Path, names: set[str], namespace: dict[str, Any]
) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    missing = names - {node.name for node in selected}
    if missing:
        raise AssertionError(f"{path}: missing definitions {sorted(missing)}")
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *selected,
            ],
            type_ignores=[],
        )
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class KimiHybridTPStrategyTests(unittest.TestCase):
    def test_mla_only_rejects_hybrid_mla_models(self) -> None:
        namespace: dict[str, Any] = {
            "mla_enabled": lambda config: bool(config.mla_enabled),
        }
        _exec_definitions(
            UTILS,
            {"_use_multiple_attentions", "mla_only"},
            namespace,
        )
        mla_only = cast(Callable[[object], bool], namespace["mla_only"])

        pure_mla = types.SimpleNamespace(mla_enabled=True, is_hybrid=False)
        hybrid_mla_mamba = types.SimpleNamespace(mla_enabled=True, is_hybrid=True)
        non_mla = types.SimpleNamespace(mla_enabled=False, is_hybrid=True)

        self.assertTrue(mla_only(pure_mla))
        self.assertFalse(mla_only(hybrid_mla_mamba))
        self.assertFalse(mla_only(non_mla))

    def test_hybrid_tp8_stores_distinct_rank_shards(self) -> None:
        namespace: dict[str, Any] = {"dataclass": dataclass}
        _exec_definitions(ADAPTER, {"ParallelStrategy"}, namespace)
        strategy_type = cast(Callable[..., Any], namespace["ParallelStrategy"])

        strategies = [
            strategy_type(
                mla_only=False,
                vllm_world_size=8,
                vllm_worker_id=rank,
                tp_size=8,
                pp_size=1,
                n_servers=1,
                dcp_size=8,
            )
            for rank in range(8)
        ]

        self.assertEqual({strategy.kv_world_size for strategy in strategies}, {8})
        self.assertEqual([strategy.kv_worker_id for strategy in strategies], list(range(8)))
        self.assertTrue(all(strategy.is_kv_writer for strategy in strategies))

    def test_pure_mla_without_dcp_keeps_rank_zero_only_optimization(self) -> None:
        namespace: dict[str, Any] = {"dataclass": dataclass}
        _exec_definitions(ADAPTER, {"ParallelStrategy"}, namespace)
        strategy_type = cast(Callable[..., Any], namespace["ParallelStrategy"])

        strategies = [
            strategy_type(
                mla_only=True,
                vllm_world_size=8,
                vllm_worker_id=rank,
                tp_size=8,
                pp_size=1,
                n_servers=1,
                dcp_size=1,
            )
            for rank in range(8)
        ]

        self.assertEqual({strategy.kv_world_size for strategy in strategies}, {1})
        self.assertEqual({strategy.kv_worker_id for strategy in strategies}, {0})
        self.assertEqual(
            [strategy.is_kv_writer for strategy in strategies],
            [True, False, False, False, False, False, False, False],
        )

    def test_connector_builds_parallel_strategy_from_mla_only(self) -> None:
        tree = ast.parse(CONNECTOR.read_text(encoding="utf-8"), filename=str(CONNECTOR))
        imports = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "lmcache.integration.vllm.utils"
            for alias in node.names
        ]
        self.assertIn("mla_only", imports)
        self.assertNotIn("mla_enabled", imports)

        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_parallel_strategy_from_vllm_config"
        )
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        strategy_call = next(
            call
            for call in calls
            if isinstance(call.func, ast.Name) and call.func.id == "ParallelStrategy"
        )
        keywords = {keyword.arg: keyword.value for keyword in strategy_call.keywords}
        self.assertIn("mla_only", keywords)
        value = keywords["mla_only"]
        assert isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Name)
        self.assertEqual(value.func.id, "mla_only")
        self.assertNotIn("use_mla", keywords)

    def test_cross_rank_fix_is_packaged_as_installable_overlays(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        container_paths = {artifact["container_path"] for artifact in manifest["artifacts"]}
        expected = {
            "/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/utils.py",
            "/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_multi_process_adapter.py",
            "/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/lmcache_mp_connector.py",
        }
        self.assertTrue(expected.issubset(container_paths))


if __name__ == "__main__":
    unittest.main()
