"""Targeted tests for the external sort-merge oversized-partition path."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from join_exec import (
    Budget,
    ControlPoint,
    QueryEngine,
    ReferenceInterpreter,
    SchemaCatalog,
    SpillManager,
)
from join_exec.types import schema_from_json
from tests.util import lineage_set


class SortMergePathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        specs = [
            ("A", [{"name": "k", "type": "int64"},
                   {"name": "v", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"},
                   {"name": "w", "type": "int64"}]),
        ]
        self.cat = SchemaCatalog(
            [schema_from_json(r, c) for r, c in specs])

    def _write(self, name, rows):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return path

    def _tree(self, a, b, jtype="inner"):
        return {"node": "join", "id": "j1", "type": jtype,
                "left": {"node": "scan", "relation": "A", "alias": "a",
                         "file": a},
                "right": {"node": "scan", "relation": "B", "alias": "b",
                          "file": b},
                "on": [["a.k", "b.k"]]}

    def _engine(self, mem, mode="force_disk", skew=None):
        budget = Budget(mem)
        control = ControlPoint()
        spill = SpillManager(budget, base_dir=self.tmp)
        engine = QueryEngine(self.cat, budget, spill, control,
                             mode=mode, skew_threshold=skew)
        return engine, spill

    def test_merge_path_inner_bag_equality(self):
        # One dominant hot key so its single partition exceeds the tiny
        # budget, forcing the external sort-merge branch.
        a = self._write("A.jsonl", [
            {"row_id": i, "k": 1, "v": i} for i in range(400)
        ] + [{"row_id": 1000 + i, "k": 50 + i, "v": i} for i in range(50)])
        b = self._write("B.jsonl", [
            {"row_id": 5000 + i, "k": 1, "w": i} for i in range(300)
        ] + [{"row_id": 9000 + i, "k": 50 + i, "w": i} for i in range(50)])
        tree = self._tree(a, b)
        ref = ReferenceInterpreter(tree).run()
        engine, spill = self._engine(8 * 1024)
        try:
            rows = list(engine.run(engine.build(tree)))
            stats = engine.stats()["joins"][0]
            self.assertEqual(len(rows), len(ref))
            self.assertEqual(lineage_set(rows),
                             {tuple(sorted(l)) for l, _ in ref})
            self.assertTrue(stats["merge_partitions"],
                            "oversized partition must use the merge branch")
            self.assertEqual(stats["mode_final"], "disk")
        finally:
            spill.cleanup()

    def test_merge_path_left_padding_once(self):
        a = self._write("A.jsonl", [
            {"row_id": i, "k": 1, "v": i} for i in range(200)
        ] + [{"row_id": 900, "k": 77, "v": 1}])  # unmatched
        b = self._write("B.jsonl", [
            {"row_id": 5000 + i, "k": 1, "w": i} for i in range(200)])
        tree = self._tree(a, b, jtype="left")
        ref = ReferenceInterpreter(tree).run()
        engine, spill = self._engine(8 * 1024)
        try:
            rows = list(engine.run(engine.build(tree)))
            stats = engine.stats()["joins"][0]
            self.assertTrue(stats["merge_partitions"])
            self.assertEqual(len(rows), len(ref))
            # exactly one padded row, the unmatched a=900
            pads = [r for r in rows if r.values[-1] is None]
            self.assertEqual(len(pads), 1)
            self.assertEqual(pads[0].rid_of("a"), 900)
            self.assertEqual(stats["left_padded"], 1)
        finally:
            spill.cleanup()

    def test_merge_path_semi_once_per_left(self):
        a = self._write("A.jsonl", [
            {"row_id": i, "k": 1, "v": i} for i in range(150)])
        b = self._write("B.jsonl", [
            {"row_id": 5000 + i, "k": 1, "w": i} for i in range(150)])
        tree = self._tree(a, b, jtype="semi")
        engine, spill = self._engine(8 * 1024)
        try:
            rows = list(engine.run(engine.build(tree)))
            self.assertEqual(len(rows), 150)  # one per left row, no dupes
            self.assertEqual(sorted(r.rid_of("a") for r in rows),
                             list(range(150)))
        finally:
            spill.cleanup()

    def test_merge_residual_suppression_no_duplicates(self):
        # Adaptive switch after a few outputs, then merge must suppress the
        # already delivered right rids and still produce the full bag.
        a = self._write("A.jsonl", [
            {"row_id": i, "k": 1, "v": i} for i in range(120)])
        b = self._write("B.jsonl", [
            {"row_id": 5000 + i, "k": 1, "w": i} for i in range(120)])
        tree = self._tree(a, b)
        ref = ReferenceInterpreter(tree).run()
        budget = Budget(8 * 1024)
        control = ControlPoint()
        spill = SpillManager(budget, base_dir=self.tmp)
        engine = QueryEngine(self.cat, budget, spill, control,
                             mode="adaptive")
        control.request_switch_after_outputs(5, "j1")
        try:
            rows = list(engine.run(engine.build(tree)))
            stats = engine.stats()["joins"][0]
            self.assertEqual(len(rows), len(ref))
            self.assertEqual(lineage_set(rows),
                             {tuple(sorted(l)) for l, _ in ref})
            self.assertTrue(stats["residual_count"] >= 1
                            or stats["merge_partitions"])
        finally:
            spill.cleanup()


if __name__ == "__main__":
    unittest.main()
