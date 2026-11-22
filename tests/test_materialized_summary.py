"""Tests for the durable, incrementally maintained summary view.

The maintained shape is fixed::

    AGGREGATE keys=b.g sum=f.amt
      SEMI(f.ek = c.ck)
        LEFT(f.fk = b.bk)
          scan F, scan B
        scan C

Every maintained result is checked against two independent references:

* :mod:`join_exec.incremental_oracle` - row-by-row enumeration over plain
  maps, sharing no incremental machinery;
* the original one-shot :class:`~join_exec.reference.ReferenceInterpreter`
  running the full query tree over the same JSONL files - proving the new
  feature reproduces the existing semantics without modifying them.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import unittest

from join_exec.incremental_oracle import (
    apply_corrections_to_maps,
    load_relations,
    oracle_summary,
)
from join_exec.materialized import (
    CorrectionError,
    MaterializedError,
    MaterializedSummary,
    OverflowRejected,
    spec_from_tree,
)
from join_exec.reference import ReferenceInterpreter
from join_exec.schema_catalog import SchemaCatalog
from join_exec.types import INT64_MAX, INT64_MIN, schema_from_json

FACT, CAT, ELIG = "F", "B", "C"


def make_catalog() -> SchemaCatalog:
    return SchemaCatalog([
        schema_from_json(FACT, [
            {"name": "fk", "type": "int64"},
            {"name": "ek", "type": "int64"},
            {"name": "amt", "type": "int64"},
        ]),
        schema_from_json(CAT, [
            {"name": "bk", "type": "int64"},
            {"name": "g", "type": "text"},
        ]),
        schema_from_json(ELIG, [
            {"name": "ck", "type": "int64"},
        ]),
    ])


def make_tree(paths: dict) -> dict:
    return {
        "node": "aggregate", "keys": ["b.g"], "sum": "f.amt",
        "input": {
            "node": "join", "type": "semi", "id": "s1",
            "on": [["f.ek", "c.ck"]],
            "left": {
                "node": "join", "type": "left", "id": "l1",
                "on": [["f.fk", "b.bk"]],
                "left": {"node": "scan", "relation": FACT, "alias": "f",
                         "file": paths[FACT]},
                "right": {"node": "scan", "relation": CAT, "alias": "b",
                          "file": paths[CAT]},
            },
            "right": {"node": "scan", "relation": ELIG, "alias": "c",
                      "file": paths[ELIG]},
        },
    }


def write_jsonl(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def engine_summary(tree: dict) -> list:
    """The original one-shot engine's aggregate output as comparable dicts."""
    out = []
    for lineage, values in ReferenceInterpreter(tree).run():
        out.append({
            "group": values["agg.key_0"],
            "count": values["agg.count"],
            "sum": values["agg.sum_f_amt"],
        })
    out.sort(key=lambda d: (
        (0, 0) if d["group"] is None
        else (1, d["group"]) if isinstance(d["group"], int)
        else (2, d["group"])))
    return out


class Harness:
    def __init__(self, facts: list, cats: list, eligs: list):
        self.tmp = tempfile.mkdtemp(prefix="matsum-")
        self.paths = {}
        for name, rows in ((FACT, facts), (CAT, cats), (ELIG, eligs)):
            self.paths[name] = os.path.join(self.tmp, name + ".jsonl")
            write_jsonl(self.paths[name], rows)
        self.catalog = make_catalog()
        self.tree = make_tree(self.paths)
        self.db = os.path.join(self.tmp, "state.db")
        self.view = MaterializedSummary.initialize(
            self.db, self.tree, self.catalog)
        self.maps = load_relations(self.paths)

    def close(self) -> None:
        self.view.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def expect(self) -> list:
        return oracle_summary(self.maps, self.view.spec)


INITIAL_FACTS = [
    {"row_id": 1, "fk": 10, "ek": 100, "amt": 10},
    {"row_id": 2, "fk": 10, "ek": 100, "amt": 20},
    {"row_id": 3, "fk": 20, "ek": 200, "amt": 30},
    {"row_id": 4, "fk": 30, "ek": None, "amt": 40},
    {"row_id": 5, "fk": 99, "ek": 300, "amt": None},
    {"row_id": 6, "fk": 10, "ek": 200, "amt": 50},
]
INITIAL_CATS = [
    {"row_id": 10, "bk": 10, "g": "A"},
    {"row_id": 11, "bk": 10, "g": "A"},
    {"row_id": 12, "bk": 20, "g": "B"},
    {"row_id": 13, "bk": 30, "g": "A"},
]
INITIAL_ELIG = [
    {"row_id": 100, "ck": 100},
    {"row_id": 101, "ck": 100},
    {"row_id": 102, "ck": 200},
    {"row_id": 103, "ck": 300},
]


class InitialLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)

    def tearDown(self) -> None:
        self.h.close()

    def test_matches_oracle_and_original_engine(self) -> None:
        got = self.h.view.read_summary()
        self.assertEqual(got, self.h.expect())
        self.assertEqual(got, engine_summary(self.h.tree))
        self.assertEqual(self.h.view.version(), 1)

    def test_hand_computed_initial_values(self) -> None:
        by_group = {r["group"]: r for r in self.h.view.read_summary()}
        # f1, f2, f6 semi-survive on key 10; B has two key-10 rows =>
        # 3 facts x 2 category rows = 6 joined rows in A (f3/f4 do not join).
        self.assertEqual(by_group["A"], {"group": "A", "count": 6, "sum": 160})
        # f3 matches B#12 once.
        self.assertEqual(by_group["B"], {"group": "B", "count": 1, "sum": 30})
        # f4 dropped (NULL semi key); f5 pads with NULL amount -> sum NULL.
        self.assertEqual(by_group[None],
                         {"group": None, "count": 1, "sum": None})

    def test_reopen_preserves_state(self) -> None:
        before = self.h.view.read_summary()
        self.h.view.close()
        view = MaterializedSummary.open(self.h.db)
        self.assertEqual(view.version(), 1)
        self.assertEqual(view.read_summary(), before)
        view.close()
        self.h.view = view  # avoid double close error on teardown

    def test_initialize_rejects_existing_db(self) -> None:
        with self.assertRaises(MaterializedError):
            MaterializedSummary.initialize(
                self.h.db, self.h.tree, self.h.catalog)


class ShapeValidationTests(unittest.TestCase):
    def test_wrong_shapes_rejected(self) -> None:
        h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)
        try:
            catalog = h.catalog
            base = h.tree
            # inner instead of left
            t = json.loads(json.dumps(base))
            t["input"]["left"]["type"] = "inner"
            with self.assertRaises(Exception):
                spec_from_tree(t, catalog)
            # root not an aggregate
            t = json.loads(json.dumps(base))
            t["node"] = "join"
            with self.assertRaises(Exception):
                spec_from_tree(t, catalog)
            # two group keys
            t = json.loads(json.dumps(base))
            t["keys"] = ["b.g", "f.fk"]
            with self.assertRaises(Exception):
                spec_from_tree(t, catalog)
            # sum on a text column
            t = json.loads(json.dumps(base))
            t["sum"] = "b.g"
            with self.assertRaises(Exception):
                spec_from_tree(t, catalog)
            # group column from the fact relation
            t = json.loads(json.dumps(base))
            t["keys"] = ["f.fk"]
            with self.assertRaises(Exception):
                spec_from_tree(t, catalog)
            # composite join key unsupported for this view
            t = json.loads(json.dumps(base))
            t["input"]["on"] = [["f.ek", "c.ck"], ["f.fk", "b.bk"]]
            with self.assertRaises(Exception):
                spec_from_tree(t, catalog)
        finally:
            h.close()


class BatchOneTests(unittest.TestCase):
    """Simultaneous category-key move, match-count change, fact-key move."""

    BATCH = [
        {"relation": CAT, "row_id": 12, "op": "upsert",
         "row": {"row_id": 12, "bk": 10, "g": "B"}},
        {"relation": ELIG, "row_id": 101, "op": "upsert",
         "row": {"row_id": 101, "ck": 999}},
        {"relation": FACT, "row_id": 3, "op": "upsert",
         "row": {"row_id": 3, "fk": 10, "ek": 100, "amt": 30}},
    ]

    def setUp(self) -> None:
        self.h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)

    def tearDown(self) -> None:
        self.h.close()

    def test_batch_result_and_locality(self) -> None:
        evidence = self.h.view.apply_corrections(self.BATCH)
        apply_corrections_to_maps(self.h.maps, self.BATCH)
        self.assertEqual(self.h.view.read_summary(), self.h.expect())
        self.assertEqual(self.h.view.version(), 2)
        # Facts 1,2,6 already on key 10 plus moved fact 3; the cat move
        # exposes old key 20 and the elig move exposes key 100.
        self.assertEqual(evidence["affected_fact_row_ids"], [1, 2, 3, 6])
        self.assertEqual(set(evidence["groups_updated"]), {"A", "B"})
        # Reopened file-based engine result must agree on current rows.
        write_jsonl(self.h.paths[FACT], list(self.h.maps[FACT].values()))
        write_jsonl(self.h.paths[CAT], list(self.h.maps[CAT].values()))
        write_jsonl(self.h.paths[ELIG], list(self.h.maps[ELIG].values()))
        self.assertEqual(self.h.view.read_summary(),
                         engine_summary(self.h.tree))

    def test_order_independence_of_batch(self) -> None:
        results = set()
        for seed in range(6):
            shuffled = list(self.BATCH)
            random.Random(seed).shuffle(shuffled)
            h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)
            try:
                h.view.apply_corrections(shuffled)
                key = tuple((r["group"], r["count"],
                             r["sum"]) for r in h.view.read_summary())
                results.add(key)
            finally:
                h.close()
        self.assertEqual(len(results), 1)


class BatchTwoTests(unittest.TestCase):
    """Last eligibility match removed, NULL group born, expired group dies."""

    BATCH = [
        {"relation": ELIG, "row_id": 103, "op": "delete"},
        {"relation": CAT, "row_id": 13, "op": "upsert",
         "row": {"row_id": 13, "bk": 40, "g": None}},
        {"relation": FACT, "row_id": 4, "op": "upsert",
         "row": {"row_id": 4, "fk": 40, "ek": 400, "amt": 40}},
        {"relation": ELIG, "row_id": 104, "op": "upsert",
         "row": {"row_id": 104, "ck": 400}},
    ]

    def setUp(self) -> None:
        self.h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)
        self.h.view.apply_corrections(BatchOneTests.BATCH)
        apply_corrections_to_maps(self.h.maps, BatchOneTests.BATCH)

    def tearDown(self) -> None:
        self.h.close()

    def test_null_group_and_expiry(self) -> None:
        evidence = self.h.view.apply_corrections(self.BATCH)
        apply_corrections_to_maps(self.h.maps, self.BATCH)
        got = self.h.view.read_summary()
        self.assertEqual(got, self.h.expect())
        by_group = {r["group"]: r for r in got}
        # f5 lost its only eligibility match: its NULL pad disappears, while
        # f4 now semi-survives on key 400 and matches B#13 (NULL group value)
        # -> exactly one NULL-group row, sum 40.
        self.assertEqual(by_group[None],
                         {"group": None, "count": 1, "sum": 40})
        # Only facts 4 and 5 were in reach of the changed keys.
        self.assertEqual(evidence["affected_fact_row_ids"], [4, 5])


class LocalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)
        self.distractors = [
            {"row_id": 1000 + i, "fk": 777, "ek": 888, "amt": i}
            for i in range(200)
        ]
        self.h.view.apply_corrections([
            {"relation": FACT, "row_id": d["row_id"], "op": "upsert",
             "row": d} for d in self.distractors
        ] + [
            {"relation": CAT, "row_id": 900, "op": "upsert",
             "row": {"row_id": 900, "bk": 777, "g": "D"}},
            {"relation": ELIG, "row_id": 901, "op": "upsert",
             "row": {"row_id": 901, "ck": 888}},
        ])
        self.h.view.reset_access_trace()

    def tearDown(self) -> None:
        self.h.close()

    def test_tiny_batch_touches_only_affected_fact(self) -> None:
        evidence = self.h.view.apply_corrections([
            {"relation": FACT, "row_id": 1, "op": "upsert",
             "row": {"row_id": 1, "fk": 10, "ek": 100, "amt": 11}},
        ])
        self.assertEqual(evidence["affected_fact_row_ids"], [1])
        trace = self.h.view.last_access_trace()
        fact_reads = {t["row_id"] for t in trace
                      if t["access"] == "fact_by_row_id"}
        self.assertEqual(fact_reads, {1})
        # No distractor row id appears anywhere in the trace.
        touched = {t["row_id"] for t in trace if t["row_id"] is not None}
        self.assertTrue({1} <= touched)
        self.assertFalse(any(r >= 1000 for r in touched))

    def test_explain_query_plan_uses_indexes_only(self) -> None:
        for entry in self.h.view.explain_local_access():
            plan = " ".join(entry["plan"]).upper()
            self.assertIn("SEARCH", plan, entry)
            self.assertNotIn("SCAN", plan, entry)

    def test_cat_key_change_expands_via_index_only(self) -> None:
        evidence = self.h.view.apply_corrections([
            {"relation": CAT, "row_id": 10, "op": "upsert",
             "row": {"row_id": 10, "bk": 11, "g": "A"}},
        ])
        # Old key 10 pulls in its facts; new key 11 has none; distractors on
        # fk 777 are not reached.
        self.assertNotIn(1000, evidence["affected_fact_row_ids"])
        plan = {e["access"]: " ".join(e["plan"]).upper()
                for e in self.h.view.explain_local_access()}
        self.assertIn("IDX_FACT_FK", plan["facts_by_category_key"])
        # The corrected category row itself is read locally by row id too.
        trace = self.h.view.last_access_trace()
        self.assertIn("corrected_row_by_id",
                      {t["access"] for t in trace})
        corrected = [t for t in trace
                     if t["access"] == "corrected_row_by_id"]
        self.assertEqual({t["row_id"] for t in corrected}, {10})
        self.assertTrue(all(t["table_name"] == "cat_rows" for t in corrected))


class RejectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)

    def tearDown(self) -> None:
        self.h.close()

    def test_bad_batches_rollback(self) -> None:
        good_summary = self.h.view.read_summary()
        cases = [
            [{"relation": FACT, "row_id": 999999, "op": "delete"}],
            [{"relation": FACT, "row_id": 1, "op": "delete"},
             {"relation": FACT, "row_id": 1, "op": "upsert",
              "row": {"row_id": 1, "fk": 10, "ek": 100, "amt": 1}}],
            [{"relation": FACT, "row_id": 7, "op": "upsert",
              "row": {"row_id": 7, "fk": 10, "ek": 100, "amt": 2 ** 63}}],
            [{"relation": FACT, "row_id": 7, "op": "upsert",
              "row": {"row_id": 7, "fk": 10, "ek": 100}}],
            [{"relation": "X", "row_id": 1, "op": "delete"}],
            [{"relation": FACT, "row_id": 7, "op": "replace",
              "row": {"row_id": 7, "fk": 10, "ek": 100, "amt": 1}}],
            [{"relation": FACT, "row_id": 7, "op": "upsert",
              "row": {"row_id": 8, "fk": 10, "ek": 100, "amt": 1}}],
            [],
        ]
        for batch in cases:
            with self.subTest(batch=batch):
                with self.assertRaises((CorrectionError, OverflowRejected)):
                    self.h.view.apply_corrections(batch)
                self.assertEqual(self.h.view.read_summary(), good_summary)
                self.assertEqual(self.h.view.version(), 1)

    def test_bool_is_not_int64(self) -> None:
        with self.assertRaises(CorrectionError):
            self.h.view.apply_corrections([
                {"relation": FACT, "row_id": 7, "op": "upsert",
                 "row": {"row_id": 7, "fk": 10, "ek": 100, "amt": True}},
            ])

    def test_overflow_rejects_and_rolls_back(self) -> None:
        version = self.h.view.version()
        summary = self.h.view.read_summary()
        with self.assertRaises(OverflowRejected):
            self.h.view.apply_corrections([
                {"relation": FACT, "row_id": 70, "op": "upsert",
                 "row": {"row_id": 70, "fk": 10, "ek": 100,
                          "amt": INT64_MAX}},
            ])
        self.assertEqual(self.h.view.version(), version)
        self.assertEqual(self.h.view.read_summary(), summary)

    def test_negative_overflow_order_independent(self) -> None:
        # The group sits exactly at INT64_MIN; any further negative
        # contribution crosses the boundary and must be rejected regardless of
        # the order corrections are listed.
        h2 = Harness(
            [{"row_id": 1, "fk": 10, "ek": 100, "amt": INT64_MIN},
             {"row_id": 2, "fk": 10, "ek": 100, "amt": 1}],
            [{"row_id": 10, "bk": 10, "g": "A"}],
            [{"row_id": 100, "ck": 100}])
        try:
            self.assertEqual(
                h2.view.read_summary(),
                [{"group": "A", "count": 2, "sum": INT64_MIN + 1}])
            with self.assertRaises(OverflowRejected):
                h2.view.apply_corrections([
                    {"relation": FACT, "row_id": 3, "op": "upsert",
                     "row": {"row_id": 3, "fk": 10, "ek": 100, "amt": -2}},
                ])
            # Rollback leaves the prior in-range state intact.
            self.assertEqual(
                h2.view.read_summary(),
                [{"group": "A", "count": 2, "sum": INT64_MIN + 1}])
        finally:
            h2.close()


class FaultAndReopenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)

    def tearDown(self) -> None:
        self.h.close()

    def test_fault_after_first_write_rolls_back(self) -> None:
        version = self.h.view.version()
        summary = self.h.view.read_summary()
        batch = [{"relation": FACT, "row_id": 6, "op": "upsert",
                  "row": {"row_id": 6, "fk": 10, "ek": 100, "amt": 51}}]
        with self.assertRaises(RuntimeError):
            self.h.view.apply_corrections(
                batch, inject_fault="after_first_write")
        self.assertEqual(self.h.view.version(), version)
        self.assertEqual(self.h.view.read_summary(), summary)

    def test_fault_before_commit_rolls_back_and_reopen_resumes(self) -> None:
        version = self.h.view.version()
        summary = self.h.view.read_summary()
        batch = [
            {"relation": CAT, "row_id": 12, "op": "upsert",
             "row": {"row_id": 12, "bk": 10, "g": "B"}},
            {"relation": FACT, "row_id": 3, "op": "upsert",
             "row": {"row_id": 3, "fk": 10, "ek": 100, "amt": 30}},
        ]
        with self.assertRaises(RuntimeError):
            self.h.view.apply_corrections(batch, inject_fault="before_commit")
        self.assertEqual(self.h.view.read_summary(), summary)
        self.h.view.close()
        view = MaterializedSummary.open(self.h.db)
        self.assertEqual(view.version(), version)
        self.assertEqual(view.read_summary(), summary)
        # Retry after reopen publishes exactly one new version.
        evidence = view.apply_corrections(batch)
        self.assertEqual(view.version(), version + 1)
        apply_corrections_to_maps(self.h.maps, batch)
        self.assertEqual(view.read_summary(),
                         oracle_summary(self.h.maps, view.spec))
        view.close()
        self.h.view = view

    def test_reader_sees_old_version_during_open_writer(self) -> None:
        # A second connection must read only committed data.
        reader = MaterializedSummary.open(self.h.db)
        self.assertEqual(reader.version(), 1)
        self.h.view.apply_corrections([
            {"relation": FACT, "row_id": 1, "op": "upsert",
             "row": {"row_id": 1, "fk": 10, "ek": 100, "amt": 11}},
        ])
        self.assertEqual(reader.version(), 2)
        reader.close()


class VersioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)

    def tearDown(self) -> None:
        self.h.close()

    def test_every_nonempty_batch_advances_version_even_if_identical(self) -> None:
        row = dict(self.h.maps[FACT][1])
        v0 = self.h.view.version()
        summary = self.h.view.read_summary()
        self.h.view.apply_corrections(
            [{"relation": FACT, "row_id": 1, "op": "upsert", "row": row}])
        self.assertEqual(self.h.view.version(), v0 + 1)
        self.assertEqual(self.h.view.read_summary(), summary)

    def test_summary_only_updated_for_affected_groups(self) -> None:
        evidence = self.h.view.apply_corrections([
            {"relation": FACT, "row_id": 1, "op": "upsert",
             "row": {"row_id": 1, "fk": 10, "ek": 100, "amt": 11}},
        ])
        self.assertEqual(evidence["groups_updated"], ["A"])


class GroupSemanticsTests(unittest.TestCase):
    """NULL amounts, integer group columns and group disappearance."""

    def test_all_null_amounts_publish_null_sum_then_group_dies(self) -> None:
        facts = [
            {"row_id": 1, "fk": 10, "ek": 100, "amt": None},
            {"row_id": 2, "fk": 10, "ek": 100, "amt": None},
        ]
        cats = [{"row_id": 10, "bk": 10, "g": "A"}]
        eligs = [{"row_id": 100, "ck": 100}]
        h = Harness(facts, cats, eligs)
        try:
            self.assertEqual(
                h.view.read_summary(),
                [{"group": "A", "count": 2, "sum": None}])
            # One real amount makes the sum concrete.
            h.view.apply_corrections([
                {"relation": FACT, "row_id": 1, "op": "upsert",
                 "row": {"row_id": 1, "fk": 10, "ek": 100, "amt": 7}}])
            self.assertEqual(
                h.view.read_summary(),
                [{"group": "A", "count": 2, "sum": 7}])
            # Removing the only eligibility row empties the whole group.
            h.view.apply_corrections([
                {"relation": ELIG, "row_id": 100, "op": "delete"}])
            self.assertEqual(h.view.read_summary(), [])
        finally:
            h.close()

    def test_integer_group_column_and_text_join_keys(self) -> None:
        catalog = SchemaCatalog([
            schema_from_json(FACT, [
                {"name": "fk", "type": "text"},
                {"name": "ek", "type": "text"},
                {"name": "amt", "type": "int64"},
            ]),
            schema_from_json(CAT, [
                {"name": "bk", "type": "text"},
                {"name": "g", "type": "int64"},
            ]),
            schema_from_json(ELIG, [
                {"name": "ck", "type": "text"},
            ]),
        ])
        tmp = tempfile.mkdtemp(prefix="matsum-intg-")
        try:
            paths = {}
            for name, rows in (
                    (FACT, [{"row_id": 1, "fk": "x", "ek": "q", "amt": 5},
                            {"row_id": 2, "fk": "z", "ek": "q", "amt": -5}]),
                    (CAT, [{"row_id": 10, "bk": "x", "g": 777}]),
                    (ELIG, [{"row_id": 100, "ck": "q"}])):
                paths[name] = os.path.join(tmp, name + ".jsonl")
                write_jsonl(paths[name], rows)
            tree = make_tree(paths)
            db = os.path.join(tmp, "state.db")
            view = MaterializedSummary.initialize(db, tree, catalog)
            # f1 matches B#10 -> group 777; f2 has no cat match -> NULL pad.
            self.assertEqual(view.read_summary(), [
                {"group": None, "count": 1, "sum": -5},
                {"group": 777, "count": 1, "sum": 5},
            ])
            maps = load_relations(paths)
            batch = [{"relation": CAT, "row_id": 10, "op": "upsert",
                      "row": {"row_id": 10, "bk": "z", "g": 777}}]
            view.apply_corrections(batch)
            apply_corrections_to_maps(maps, batch)
            self.assertEqual(view.read_summary(), oracle_summary(maps, view.spec))
            view.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_null_keys_never_match_on_either_join(self) -> None:
        facts = [
            {"row_id": 1, "fk": None, "ek": 100, "amt": 3},
            {"row_id": 2, "fk": 10, "ek": None, "amt": 4},
            {"row_id": 3, "fk": None, "ek": None, "amt": 5},
        ]
        cats = [{"row_id": 10, "bk": None, "g": "A"},
                {"row_id": 11, "bk": 10, "g": "A"}]
        eligs = [{"row_id": 100, "ck": None}, {"row_id": 101, "ck": 100}]
        h = Harness(facts, cats, eligs)
        try:
            # f1: SEMI survives, LEFT NULL key -> one NULL pad.
            # f2/f3: SEMI NULL key -> dropped entirely (no pad).
            self.assertEqual(
                h.view.read_summary(),
                [{"group": None, "count": 1, "sum": 3}])
            # A cat correction touching NULL keys must not expand to facts:
            # only a real-key change can affect anything.
            evidence = h.view.apply_corrections([
                {"relation": CAT, "row_id": 10, "op": "upsert",
                 "row": {"row_id": 10, "bk": None, "g": "Z"}}])
            self.assertEqual(evidence["affected_fact_row_ids"], [])
            self.assertEqual(evidence["groups_updated"], [])
            self.assertEqual(
                h.view.read_summary(),
                [{"group": None, "count": 1, "sum": 3}])
        finally:
            h.close()


class TraceHygieneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)

    def tearDown(self) -> None:
        self.h.close()

    def test_failed_batch_leaves_no_trace_and_no_untagged_commits(self) -> None:
        import sqlite3
        self.h.view.apply_corrections([
            {"relation": FACT, "row_id": 1, "op": "upsert",
             "row": {"row_id": 1, "fk": 10, "ek": 100, "amt": 11}}])
        with sqlite3.connect(self.h.db) as conn:
            seqs_before = {r[0] for r in conn.execute(
                "SELECT DISTINCT batch_seq FROM access_trace")}
        with self.assertRaises(RuntimeError):
            self.h.view.apply_corrections([
                {"relation": FACT, "row_id": 2, "op": "upsert",
                 "row": {"row_id": 2, "fk": 10, "ek": 100, "amt": 999}}],
                inject_fault="before_commit")
        with sqlite3.connect(self.h.db) as conn:
            seqs_after = {r[0] for r in conn.execute(
                "SELECT DISTINCT batch_seq FROM access_trace")}
            untagged = conn.execute(
                "SELECT COUNT(*) FROM access_trace "
                "WHERE version_after IS NULL").fetchone()[0]
        self.assertEqual(seqs_before, seqs_after)
        self.assertEqual(untagged, 0)


class RandomizedPropertyTests(unittest.TestCase):
    """Random correction sequences vs the independent oracle."""

    def test_random_batches_against_oracle(self) -> None:
        rng = random.Random(20261122)
        h = Harness(INITIAL_FACTS, INITIAL_CATS, INITIAL_ELIG)
        try:
            next_rid = 5000
            for _ in range(40):
                batch = []
                for _ in range(rng.randint(1, 4)):
                    rel = rng.choice([FACT, CAT, ELIG])
                    existing = list(h.maps[rel].keys())
                    if existing and rng.random() < 0.6:
                        rid = rng.choice(existing)
                        op = rng.choice(["upsert", "delete"])
                    else:
                        rid = next_rid
                        next_rid += 1
                        op = "upsert"
                    if op == "delete":
                        batch.append(
                            {"relation": rel, "row_id": rid, "op": "delete"})
                        continue
                    if rel == FACT:
                        row = {"row_id": rid,
                               "fk": rng.choice([10, 20, 30, 40, None, 777]),
                               "ek": rng.choice([100, 200, 300, 400, None]),
                               "amt": rng.choice(
                                   [0, 1, -1, 100, None, INT64_MIN])}
                    elif rel == CAT:
                        row = {"row_id": rid,
                               "bk": rng.choice([10, 20, 30, 40, None, 777]),
                               "g": rng.choice(["A", "B", None, "D"])}
                    else:
                        row = {"row_id": rid,
                               "ck": rng.choice([100, 200, 300, 400, None])}
                    batch.append({"relation": rel, "row_id": rid,
                                  "op": "upsert", "row": row})
                # De-duplicate within the batch the way the contract demands;
                # if a random batch itself duplicates, assert rejection.
                keys = [(c["relation"], c["row_id"]) for c in batch]
                if len(set(keys)) != len(keys):
                    with self.assertRaises(CorrectionError):
                        h.view.apply_corrections(batch)
                    continue
                # A batch deleting a missing id must be rejected, not applied.
                missing = [c for c in batch if c["op"] == "delete"
                           and c["row_id"] not in h.maps[c["relation"]]]
                if missing:
                    with self.assertRaises(CorrectionError):
                        h.view.apply_corrections(batch)
                    continue
                try:
                    h.view.apply_corrections(batch)
                except OverflowRejected:
                    # Oracle arithmetic on Python ints does not overflow; the
                    # view must keep the previous, in-range state.
                    expected = oracle_summary(h.maps, h.view.spec)
                    self.assertEqual(h.view.read_summary(), expected)
                    continue
                apply_corrections_to_maps(h.maps, batch)
                self.assertEqual(
                    h.view.read_summary(),
                    oracle_summary(h.maps, h.view.spec))
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
