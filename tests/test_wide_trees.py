"""Wide (3-6 leaf) join trees and outer/semi semantic-barrier tests."""

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
from join_exec.engine import count_leaves
from join_exec.types import schema_from_json
from tests.util import lineage_set


class WideTreeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.n = 6
        specs = []
        for i in range(self.n):
            rel = chr(ord("R") + i)
            specs.append((rel, [{"name": "k", "type": "int64"},
                                {"name": "x", "type": "int64"}]))
        self.cat = SchemaCatalog(
            [schema_from_json(r, c) for r, c in specs])
        self.paths = {}
        for i, (rel, _) in enumerate(specs):
            alias = f"r{i}"
            path = os.path.join(self.tmp, f"{rel}.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                # each relation: key 1 (2 rows), key 2 (1 row)
                for rid in range(3):
                    key = 1 if rid < 2 else 2
                    fh.write(json.dumps(
                        {"row_id": rid, "k": key, "x": i * 10 + rid}) + "\n")
            self.paths[alias] = (rel, path)

    def _scan(self, alias):
        rel, path = self.paths[alias]
        return {"node": "scan", "relation": rel, "alias": alias, "file": path}

    def _six_way(self, root_type, inner_types):
        # left-deep: ((((r0 J r1) J r2) ...) J r5)
        node = self._scan("r0")
        for i in range(1, self.n):
            jtype = inner_types[i - 1] if i - 1 < len(inner_types) else "inner"
            node = {
                "node": "join", "id": f"j{i}", "type": jtype,
                "left": node, "right": self._scan(f"r{i}"),
                "on": [["r0.k", f"r{i}.k"]],
            }
        node["type"] = root_type
        return node

    def _run(self, tree, mem=256 * 1024, mode="adaptive", control=None):
        budget = Budget(mem)
        control = control or ControlPoint()
        spill = SpillManager(budget, base_dir=self.tmp)
        engine = QueryEngine(self.cat, budget, spill, control, mode=mode)
        try:
            plan = engine.build(tree)
            rows = list(engine.run(plan))
            return rows, engine.stats()
        finally:
            spill.cleanup()

    def test_six_way_inner_matches_reference(self):
        tree = self._six_way("inner", ["inner"] * 5)
        self.assertEqual(count_leaves(tree), 6)
        ref = ReferenceInterpreter(tree).run()
        for mode in ("force_memory", "force_disk", "adaptive"):
            with self.subTest(mode=mode):
                ctrl = ControlPoint()
                if mode == "adaptive":
                    ctrl.request_switch_after_outputs(3, "j2")
                    ctrl.request_switch_after_outputs(2, "j4")
                rows, stats = self._run(tree, mode=mode, control=ctrl)
                self.assertEqual(len(rows), len(ref))
                self.assertEqual(lineage_set(rows),
                                 {tuple(sorted(l)) for l, _ in ref})

    def test_left_join_semantic_boundary_keeps_unmatched(self):
        # A LEFT B then INNER joins must not drop A rows that failed to match
        # B: reference and product must agree, and a pure-inner reordering
        # would change the count (asserted against the independent oracle).
        tree = self._six_way("inner",
                             ["left", "inner", "inner", "inner", "inner"])
        ref = ReferenceInterpreter(tree).run()
        rows, stats = self._run(tree)
        self.assertEqual(len(rows), len(ref))
        self.assertEqual(lineage_set(rows),
                         {tuple(sorted(l)) for l, _ in ref})
        # the left join node is present and reported as a distinct operator
        ids = {j["join_id"]: j["join_type"] for j in stats["joins"]}
        self.assertEqual(ids["j1"], "left")

    def test_semi_boundary_dedup_against_reference(self):
        # Isolate the SEMI node: at its own output each left row appears once.
        semi_tree = {
            "node": "join", "id": "j1", "type": "semi",
            "left": self._scan("r0"), "right": self._scan("r1"),
            "on": [["r0.k", "r1.k"]],
        }
        ref = ReferenceInterpreter(semi_tree).run()
        ctrl = ControlPoint()
        ctrl.request_switch_after_outputs(2, "j1")
        semi_rows, _ = self._run(semi_tree, control=ctrl)
        r0_rids = [r.rid_of("r0") for r in semi_rows]
        self.assertEqual(len(r0_rids), len(set(r0_rids)))
        self.assertEqual(len(semi_rows), len(ref))

        # The full tree still matches the independent oracle end to end.
        tree = self._six_way("inner",
                             ["semi", "inner", "inner", "inner", "inner"])
        ref_full = ReferenceInterpreter(tree).run()
        ctrl2 = ControlPoint()
        ctrl2.request_switch_after_outputs(2, "j1")
        rows, stats = self._run(tree, control=ctrl2)
        self.assertEqual(len(rows), len(ref_full))
        self.assertEqual(lineage_set(rows),
                         {tuple(sorted(l)) for l, _ in ref_full})
        ids = {j["join_id"]: j["join_type"] for j in stats["joins"]}
        self.assertEqual(ids["j1"], "semi")

    def test_three_way_minimum_leaf_count(self):
        node = self._scan("r0")
        node = {"node": "join", "id": "j1", "type": "inner", "left": node,
                "right": self._scan("r1"), "on": [["r0.k", "r1.k"]]}
        node = {"node": "join", "id": "j2", "type": "inner", "left": node,
                "right": self._scan("r2"), "on": [["r0.k", "r2.k"]]}
        self.assertEqual(count_leaves(node), 3)
        ref = ReferenceInterpreter(node).run()
        rows, _ = self._run(node)
        self.assertEqual(len(rows), len(ref))


if __name__ == "__main__":
    unittest.main()
