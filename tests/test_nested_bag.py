"""Regression tests for nested-join bag multiplicity under disk execution.

These cover two bugs found by randomized testing:

1. a Scan must be safely re-openable (disk joins re-run whole subtrees);
2. a SEMI join over an already-joined left input must preserve every bag
   element: after an upstream join, one base row_id appears in many distinct
   input tuples, and SEMI dedup must key on full lineage, not the base rid.
"""

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


class NestedBagTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.specs = [
            (c, [{"name": "k", "type": "int64"},
                 {"name": "v", "type": "int64"},
                 {"name": "t", "type": "text"}])
            for c in ("R", "S", "T", "U")
        ]
        self.cat = SchemaCatalog(
            [schema_from_json(r, c) for r, c in self.specs])
        self.paths = {}
        for c, al in (("R", "r0"), ("S", "r1"), ("T", "r2"), ("U", "r3")):
            p = os.path.join(self.tmp, f"{c}.jsonl")
            self.paths[al] = p
            with open(p, "w", encoding="utf-8") as fh:
                rid = 0
                for k in list(range(1, 7)) + [None]:
                    for _ in range(12):
                        fh.write(json.dumps(
                            {"row_id": rid, "k": k, "v": rid,
                             "t": "z" * 40 if c != "R" else ""}) + "\n")
                        rid += 1

    def _scan(self, al, rel):
        return {"node": "scan", "relation": rel, "alias": al,
                "file": self.paths[al]}

    def _tree(self):
        return {
            "node": "join", "id": "j3", "type": "inner",
            "left": {
                "node": "join", "id": "j2", "type": "semi",
                "left": {
                    "node": "join", "id": "j1", "type": "left",
                    "left": self._scan("r0", "R"),
                    "right": self._scan("r1", "S"),
                    "on": [["r0.k", "r1.k"]],
                },
                "right": self._scan("r2", "T"),
                "on": [["r0.k", "r2.k"]],
            },
            "right": self._scan("r3", "U"),
            "on": [["r0.k", "r3.k"]],
        }

    def test_nested_semi_preserves_bag_multiplicity(self):
        tree = self._tree()
        hint = {c: ["k", "v", "t"] for c, _ in self.specs}
        ref = ReferenceInterpreter(tree, schemas=hint).run()
        for mode in ("force_disk", "adaptive"):
            budget = Budget(16 * 1024)
            spill = SpillManager(budget, base_dir=self.tmp)
            control = ControlPoint()
            engine = QueryEngine(self.cat, budget, spill, control, mode=mode)
            try:
                rows = list(engine.run(engine.build(tree)))
                self.assertEqual(len(rows), len(ref))
                self.assertEqual(
                    lineage_set(rows),
                    {tuple(sorted(l)) for l, _ in ref})
            finally:
                spill.cleanup()

    def test_scan_reopen_safe(self):
        # Re-opening a scan source must not trip duplicate-row_id detection.
        from join_exec.scan import Scan
        schema = self.cat.get("R")
        scan = Scan(schema, self.paths["r0"], "r0")
        first = list(scan)
        second = list(scan)
        self.assertEqual(len(first), len(second))
        self.assertEqual([r.rids for r in first], [r.rids for r in second])


if __name__ == "__main__":
    unittest.main()
