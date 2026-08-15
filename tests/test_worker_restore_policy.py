from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import cast
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "overlays/lmcache/v1/multiprocess/transfer_context/worker_transfer.py"
)
HELPER = "_collapse_chunks_for_single_destination"


def load_helper() -> Callable[
    [list[object], list[int], int, int], tuple[list[object], list[int]]
]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == HELPER
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return cast(
        Callable[
            [list[object], list[int], int, int],
            tuple[list[object], list[int]],
        ],
        namespace[HELPER],
    )


class WorkerRestorePolicyTests(unittest.TestCase):
    def test_full_alias_keeps_only_newest_snapshot(self) -> None:
        collapse = load_helper()
        chunks = [object(), object(), object()]

        selected_chunks, selected_ids = collapse(
            chunks,
            [4, 5, 4, 5, 4, 5],
            2,
            2,
        )

        self.assertEqual(selected_chunks, chunks[-1:])
        self.assertEqual(selected_ids, [4, 5])

    def test_unique_mixed_partial_and_skewed_inputs_are_unchanged(self) -> None:
        collapse = load_helper()
        chunks = [object(), object(), object()]
        cases = [
            ([0, 1, 2], 1, 1),
            ([0, 0, 1], 1, 1),
            ([0, 0, 0], 2, 1),
            ([0, 0], 1, 1),
        ]

        for block_ids, blocks_in_chunk, blocks_per_window in cases:
            with self.subTest(block_ids=block_ids):
                selected_chunks, selected_ids = collapse(
                    chunks,
                    block_ids,
                    blocks_in_chunk,
                    blocks_per_window,
                )
                self.assertIs(selected_chunks, chunks)
                self.assertIs(selected_ids, block_ids)

    def test_pickle_and_shm_retrieve_paths_apply_policy(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        for name in (
            "_submit_retrieve_multigroup_pickle",
            "_submit_retrieve_multigroup",
        ):
            calls = [
                node
                for node in ast.walk(methods[name])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == HELPER
            ]
            self.assertEqual(len(calls), 1, name)


if __name__ == "__main__":
    unittest.main()
