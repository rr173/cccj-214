"""Edge-case and fault tests for the adaptive executor."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from join_exec import (
    BadRowError,
    ControlPoint,
    MemoryLimitExceeded,
    RunCancelled,
    SpillError,
)
from join_exec.budget import Budget
from join_exec.engine import QueryEngine
from join_exec.reference import ReferenceInterpreter
from join_exec.schema_catalog import SchemaCatalog
from join_exec.spill import SpillManager
from join_exec.types import INT64_MAX, schema_from_json
from tests.util import lineage_set, reference_bag, run_tree


def write_jsonl(rows, tmpdir, name):
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def make_catalog(tmpdir, specs, files):
    schemas = [schema_from_json(rel, cols) for rel, cols in specs]
    return SchemaCatalog(schemas)


def join_tree(path_a, path_b, jtype="inner",
              keys=(("a.k", "b.k"),), extra_left=None, extra_right=None):
    left = {"node": "scan", "relation": "A", "alias": "a", "file": path_a}
    right = {"node": "scan", "relation": "B", "alias": "b", "file": path_b}
    if extra_left:
        left = {"node": "filter", "expr": extra_left, "input": left}
    return {"node": "join", "id": "j1", "type": jtype,
            "left": left, "right": right,
            "on": [[l, r] for l, r in keys]}


class EngineHarness:
    def __init__(self, tmpdir, mem=256 * 1024, mode="adaptive",
                 skew=None, control=None, catalog=None):
        self.tmpdir = tmpdir
        self.budget = Budget(mem)
        self.control = control or ControlPoint()
        self.spill = SpillManager(self.budget, base_dir=tmpdir)
        self.engine = QueryEngine(catalog, self.budget, self.spill,
                                  self.control, mode=mode,
                                  skew_threshold=skew)
    def run(self, tree):
        plan = self.engine.build(tree)
        rows = list(self.engine.run(plan))
        return rows, self.engine.stats()

    def close(self):
        self.spill.cleanup()


class CompositeAndDuplicateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        specs = [
            ("A", [{"name": "k1", "type": "int64"},
                   {"name": "k2", "type": "text"},
                   {"name": "v", "type": "int64"}]),
            ("B", [{"name": "k1", "type": "int64"},
                   {"name": "k2", "type": "text"},
                   {"name": "w", "type": "int64"}]),
        ]
        self.cat = make_catalog(self.tmp, specs, None)
        # duplicate business values, distinct rids -> bag multiplicity kept
        self.a = write_jsonl([
            {"row_id": 1, "k1": 1, "k2": "x", "v": 10},
            {"row_id": 2, "k1": 1, "k2": "x", "v": 10},   # dup business row
            {"row_id": 3, "k1": 1, "k2": None, "v": 20},  # partial NULL
            {"row_id": 4, "k1": None, "k2": "x", "v": 30},
            {"row_id": 5, "k1": 2, "k2": "y", "v": 40},
        ], self.tmp, "A.jsonl")
        self.b = write_jsonl([
            {"row_id": 10, "k1": 1, "k2": "x", "w": 1},
            {"row_id": 11, "k1": 1, "k2": "x", "w": 2},   # dup business row
            {"row_id": 12, "k1": 1, "k2": None, "w": 3},
            {"row_id": 13, "k1": None, "k2": "x", "w": 4},
            {"row_id": 14, "k1": 2, "k2": "y", "w": 5},
        ], self.tmp, "B.jsonl")

    def test_composite_null_and_duplicates(self):
        tree = join_tree(
            self.a, self.b,
            keys=(("a.k1", "b.k1"), ("a.k2", "b.k2")))
        ref = ReferenceInterpreter(tree).run()
        for mode in ("force_memory", "force_disk", "adaptive"):
            with self.subTest(mode=mode):
                h = EngineHarness(self.tmp, catalog=self.cat, mode=mode)
                try:
                    ctrl = h.control
                    if mode == "adaptive":
                        ctrl.request_switch_after_outputs(2, "j1")
                    rows, stats = h.run(tree)
                    self.assertEqual(len(rows), len(ref))
                    self.assertEqual(lineage_set(rows),
                                     {tuple(sorted(l)) for l, _ in ref})
                    # 2 left dup rows x 2 right dup rows = 4 pairs on (1,'x')
                    # plus one pair (2,'y') = 5; null-component rows excluded
                    self.assertEqual(len(rows), 5)
                finally:
                    h.close()

    def test_semi_composite(self):
        tree = join_tree(
            self.a, self.b, jtype="semi",
            keys=(("a.k1", "b.k1"), ("a.k2", "b.k2")))
        h = EngineHarness(self.tmp, catalog=self.cat)
        try:
            rows, _ = h.run(tree)
            self.assertEqual(sorted(r.rid_of("a") for r in rows), [1, 2, 5])
        finally:
            h.close()


class ThreeValuedAndAggregateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        specs = [
            ("A", [{"name": "k", "type": "int64"},
                   {"name": "v", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"},
                   {"name": "w", "type": "int64"}]),
        ]
        self.cat = make_catalog(self.tmp, specs, None)
        self.a = write_jsonl([
            {"row_id": 1, "k": 1, "v": None},
            {"row_id": 2, "k": 1, "v": None},
            {"row_id": 3, "k": 2, "v": 5},
        ], self.tmp, "A.jsonl")
        self.b = write_jsonl([
            {"row_id": 10, "k": 1, "w": None},
            {"row_id": 11, "k": 2, "w": 7},
        ], self.tmp, "B.jsonl")

    def test_all_null_sum_is_null(self):
        tree = {
            "node": "aggregate", "keys": ["a.k"], "sum": "a.v",
            "input": join_tree(self.a, self.b),
        }
        h = EngineHarness(self.tmp, catalog=self.cat)
        try:
            rows, _ = h.run(tree)
            by_key = {r.values[0]: r for r in rows}
            self.assertIsNone(by_key[1].values[-1])   # all-null group
            self.assertEqual(by_key[2].values[-1], 5)
            self.assertEqual(by_key[1].values[-2], 2)  # count still 2
        finally:
            h.close()

    def test_three_valued_filter(self):
        tree = {
            "node": "filter",
            "expr": {"op": "or", "args": [
                {"op": "gt", "left": "a.v", "right": {"literal": 0}},
                {"op": "eq", "left": "a.v", "right": {"literal": None}},
            ]},
            "input": join_tree(self.a, self.b),
        }
        ref = ReferenceInterpreter(tree).run()
        h = EngineHarness(self.tmp, catalog=self.cat)
        try:
            rows, _ = h.run(tree)
            # NULL > 0 is UNKNOWN, v = NULL literal is UNKNOWN -> UNKNOWN,
            # so only the v=5 row passes.
            self.assertEqual(len(rows), len(ref))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].rid_of("a"), 3)
        finally:
            h.close()

    def test_int64_overflow_on_sum(self):
        specs = [("A", [{"name": "k", "type": "int64"},
                        {"name": "v", "type": "int64"}]),
                 ("B", [{"name": "k", "type": "int64"}])]
        cat = make_catalog(self.tmp, specs, None)
        a = write_jsonl([{"row_id": 1, "k": 1, "v": INT64_MAX},
                         {"row_id": 2, "k": 1, "v": 1}], self.tmp, "Ao.jsonl")
        b = write_jsonl([{"row_id": 10, "k": 1}], self.tmp, "Bo.jsonl")
        tree = {"node": "aggregate", "keys": [], "sum": "a.v",
                "input": join_tree(a, b,
                                   keys=(("a.k", "b.k"),))}
        h = EngineHarness(self.tmp, catalog=cat)
        try:
            with self.assertRaises(OverflowError):
                h.run(tree)
        finally:
            h.close()


class EmptyAndFaultTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        specs = [
            ("A", [{"name": "k", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"}]),
        ]
        self.cat = make_catalog(self.tmp, specs, None)
        self.a = write_jsonl([], self.tmp, "empty.jsonl")
        self.b = write_jsonl([{"row_id": 1, "k": 1}], self.tmp, "B.jsonl")

    def test_empty_inputs(self):
        for jtype in ("inner", "left", "semi"):
            tree = join_tree(self.a, self.b, jtype)
            h = EngineHarness(self.tmp, catalog=self.cat)
            try:
                rows, stats = h.run(tree)
                self.assertEqual(rows, [])
                ref = ReferenceInterpreter(tree).run()
                self.assertEqual(len(ref), 0)
            finally:
                h.close()

    def test_empty_right_left_join_pads_every_left(self):
        # Non-empty left, empty right: LEFT pads each left row once; the
        # oracle must infer right columns from the schema hint.
        left = write_jsonl(
            [{"row_id": 1, "k": 1}, {"row_id": 2, "k": 2}],
            self.tmp, "nonempty.jsonl")
        empty_right = write_jsonl([], self.tmp, "empty_right.jsonl")
        tree = join_tree(left, empty_right, "left")
        hint = {"A": ["k"], "B": ["k"]}
        ref = ReferenceInterpreter(tree, schemas=hint).run()
        for mode in ("force_memory", "force_disk", "adaptive"):
            h = EngineHarness(self.tmp, catalog=self.cat, mode=mode)
            try:
                rows, stats = h.run(tree)
                self.assertEqual(len(rows), 2)
                self.assertEqual(stats["joins"][0]["left_padded"], 2)
                self.assertEqual(len(rows), len(ref))
                # right-side slot is the padded NULL
                self.assertTrue(all(r.values[-1] is None for r in rows))
            finally:
                h.close()

    def test_bad_row_terminates_run(self):
        bad = os.path.join(self.tmp, "bad.jsonl")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write('{"row_id": 1, "k": 1}\n{not json}\n')
        tree = join_tree(bad, self.b)
        h = EngineHarness(self.tmp, catalog=self.cat)
        try:
            with self.assertRaises(BadRowError):
                h.run(tree)
        finally:
            h.close()
            self.assertEqual(os.listdir(self.tmp).count("bad.jsonl"), 1)

    def test_out_of_range_int(self):
        bad = os.path.join(self.tmp, "range.jsonl")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write('{"row_id": 1, "k": 9999999999999999999999}\n')
        tree = join_tree(bad, self.b)
        h = EngineHarness(self.tmp, catalog=self.cat)
        try:
            with self.assertRaises(BadRowError):
                h.run(tree)
        finally:
            h.close()

    def test_spill_failure(self):
        specs = [("A", [{"name": "k", "type": "int64"},
                        {"name": "v", "type": "text"}]),
                 ("B", [{"name": "k", "type": "int64"}])]
        cat = make_catalog(self.tmp, specs, None)
        a = write_jsonl([{"row_id": i, "k": 1, "v": "x" * 8}
                         for i in range(20)], self.tmp, "As.jsonl")
        b = write_jsonl([{"row_id": 100, "k": 1}], self.tmp, "Bs.jsonl")
        budget = Budget(4096)
        spill = SpillManager(budget, base_dir=self.tmp)
        control = ControlPoint()
        engine = QueryEngine(cat, budget, spill, control, mode="adaptive",
                             skew_threshold=1)
        # Make every spill write fail to simulate a full/failing disk.
        from join_exec.spill import SpillFile

        original_writer = SpillFile.write_record

        def boom(self, record):
            raise SpillError("simulated disk full")
        SpillFile.write_record = boom  # type: ignore[assignment]
        tree = join_tree(a, b)
        try:
            with self.assertRaises(SpillError):
                list(engine.run(engine.build(tree)))
        finally:
            SpillFile.write_record = original_writer  # type: ignore[assignment]
            spill.cleanup()

    def test_cancellation_closes_cleanly(self):
        specs = [("A", [{"name": "k", "type": "int64"}]),
                 ("B", [{"name": "k", "type": "int64"}])]
        cat = make_catalog(self.tmp, specs, None)
        a = write_jsonl([{"row_id": i, "k": i} for i in range(50)],
                        self.tmp, "Ac.jsonl")
        b = write_jsonl([{"row_id": 100, "k": 1}], self.tmp, "Bc.jsonl")
        control = ControlPoint()
        h = EngineHarness(self.tmp, catalog=cat, control=control,
                          skew=5)
        try:
            gen = iter(h.engine.run(h.engine.build(join_tree(a, b))))
            next(gen)
            control.cancel("test cancel")
            with self.assertRaises(RunCancelled):
                list(gen)
        finally:
            h.close()

    def test_consumer_early_close(self):
        specs = [("A", [{"name": "k", "type": "int64"}]),
                 ("B", [{"name": "k", "type": "int64"}])]
        cat = make_catalog(self.tmp, specs, None)
        a = write_jsonl([{"row_id": i, "k": 1} for i in range(100)],
                        self.tmp, "Ae.jsonl")
        b = write_jsonl([{"row_id": 100, "k": 1}], self.tmp, "Be.jsonl")
        h = EngineHarness(self.tmp, catalog=cat, mode="force_disk")
        try:
            gen = iter(h.engine.run(h.engine.build(join_tree(a, b))))
            first = next(gen)
            gen.close()  # consumer goes away
            self.assertEqual(first.rid_of("a"), 0)
        finally:
            h.close()


class SkewAndInterleavingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.specs = [
            ("A", [{"name": "k", "type": "int64"},
                   {"name": "v", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"},
                   {"name": "w", "type": "int64"}]),
            ("C", [{"name": "k", "type": "int64"}]),
        ]
        self.cat = make_catalog(self.tmp, self.specs, None)

    def _skew_files(self, hot=400, cold=50):
        a_rows = [{"row_id": i, "k": 1, "v": i} for i in range(hot)]
        a_rows += [{"row_id": 1000 + i, "k": 2, "v": i} for i in range(cold)]
        b_rows = [{"row_id": 5000 + i, "k": 1, "w": i} for i in range(hot)]
        b_rows += [{"row_id": 9000 + i, "k": 2, "w": i} for i in range(cold)]
        c_rows = [{"row_id": 7000, "k": 1}, {"row_id": 7001, "k": 2}]
        return (write_jsonl(a_rows, self.tmp, "Ah.jsonl"),
                write_jsonl(b_rows, self.tmp, "Bh.jsonl"),
                write_jsonl(c_rows, self.tmp, "Ch.jsonl"))

    def test_hot_group_larger_than_budget_merge(self):
        a, b, _ = self._skew_files(hot=300, cold=20)
        tree = join_tree(a, b)
        # tiny budget forces build-phase switch; the hot partition then
        # exceeds the available share and must go through block nested loop.
        h = EngineHarness(self.tmp, mem=16 * 1024, catalog=self.cat,
                          mode="adaptive", skew=10)
        try:
            rows, stats = h.run(tree)
            ref = ReferenceInterpreter(tree).run()
            self.assertEqual(len(rows), len(ref))
            self.assertEqual(lineage_set(rows),
                             {tuple(sorted(l)) for l, _ in ref})
            js = stats["joins"][0]
            self.assertEqual(js["mode_final"], "disk")
            self.assertTrue(js["merge_partitions"])
            self.assertGreater(stats["spill"]["bytes_written"], 0)
        finally:
            h.close()

    def test_interleaved_multi_join_switches(self):
        a, b, c = self._skew_files(hot=120, cold=30)
        tree = {
            "node": "join", "id": "j2", "type": "inner",
            "left": {
                "node": "join", "id": "j1", "type": "inner",
                "left": {"node": "scan", "relation": "A", "alias": "a",
                         "file": a},
                "right": {"node": "scan", "relation": "B", "alias": "b",
                          "file": b},
                "on": [["a.k", "b.k"]],
            },
            "right": {"node": "scan", "relation": "C", "alias": "c",
                      "file": c},
            "on": [["a.k", "c.k"]],
        }
        ctrl = ControlPoint()
        ctrl.request_switch_after_outputs(50, "j1")
        ctrl.request_switch_after_outputs(25, "j2")
        h = EngineHarness(self.tmp, mem=64 * 1024, catalog=self.cat,
                          control=ctrl)
        try:
            rows, stats = h.run(tree)
            ref = ReferenceInterpreter(tree).run()
            self.assertEqual(len(rows), len(ref))
            self.assertEqual(lineage_set(rows),
                             {tuple(sorted(l)) for l, _ in ref})
            ids = {j["join_id"]: j for j in stats["joins"]}
            self.assertEqual(len(ids["j1"]["switch_events"]), 1)
            self.assertEqual(len(ids["j2"]["switch_events"]), 1)
        finally:
            h.close()

    def test_budget_overflow_is_switch_not_failure(self):
        a, b, _ = self._skew_files(hot=200, cold=10)
        tree = join_tree(a, b)
        h = EngineHarness(self.tmp, mem=8 * 1024, catalog=self.cat)
        try:
            rows, stats = h.run(tree)
            ref = ReferenceInterpreter(tree).run()
            self.assertEqual(len(rows), len(ref))
            self.assertEqual(stats["joins"][0]["mode_final"], "disk")
        finally:
            h.close()

    def test_force_memory_breach_raises(self):
        a, b, _ = self._skew_files(hot=200, cold=10)
        tree = join_tree(a, b)
        h = EngineHarness(self.tmp, mem=4 * 1024, catalog=self.cat,
                          mode="force_memory")
        try:
            with self.assertRaises(MemoryLimitExceeded):
                h.run(tree)
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
