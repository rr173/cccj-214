"""Provenance semantics, durability and failure tests.

Every scenario is checked against an independent oracle
(:mod:`tests.proof_oracle`) that enumerates expected witnesses row by row
from the raw JSONL inputs without using any product provenance, spill or
switch machinery.  Both business results and proof contents are asserted;
memory and disk executions must produce identical proof sets.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from join_exec import (
    BadRowError,
    ControlPoint,
    ProofError,
    ProofReader,
    RunCancelled,
)
from join_exec.proof_store import (
    aggregate_proof_ref,
    row_proof_ref,
)
from tests.proof_oracle import ProofOracle, expected_run_id
from tests.util import write_jsonl


def schema_spec(rels):
    return {"relations": {rel: cols for rel, cols in rels}}


def write_catalog(tmp, name, rels, rows_by_rel):
    from join_exec import SchemaCatalog
    from join_exec.types import schema_from_json
    paths = {}
    for rel, _cols in rels:
        paths[rel] = write_jsonl(
            os.path.join(tmp, f"{name}_{rel}.jsonl"), rows_by_rel[rel])
    catalog = SchemaCatalog(
        [schema_from_json(rel, cols) for rel, cols in rels])
    return catalog, paths


def doc_sources(doc):
    if doc["kind"] in ("count", "sum"):
        out = set()
        for c in doc["contributions"]:
            out.update(tuple(s) for s in c["sources"])
        return out
    out = set()
    for w in doc["witnesses"]:
        out.update(tuple(s) for s in w["sources"])
    return out


def doc_match_tuples(doc):
    """All (alias,row_id) combinations carried by a join/semi proof."""
    return {
        tuple(sorted(tuple(s) for s in w["sources"]))
        for w in doc["witnesses"]
        if w["role"] in ("join_match", "semi_match")
    }


class ProofHarness:
    """Runs one tree through the provenance engine and exposes both the
    business rows and the durable proof reader."""

    def __init__(self, tree, catalog, tmp, mode="adaptive",
                 mem=256 * 1024, control=None, shards=8,
                 write_fault=None, skew=None, overrides=None):
        self.tree = tree
        self.tmp = tmp
        self.spill_dir = os.path.join(tmp, "spill")
        self.proof_dir = os.path.join(tmp, "proofs")
        os.makedirs(self.spill_dir, exist_ok=True)
        # Engine builds its own catalog-based engine via util helper which
        # uses the fixed fixture catalog; so we replicate with a local
        # engine construction instead.
        from join_exec import Budget, ProofStore, ProvenanceSession, \
            QueryEngine, SpillManager
        from tests.proof_oracle import stable_node_ids
        self.budget = Budget(mem)
        self.control = control or ControlPoint()
        self.spill = SpillManager(self.budget, base_dir=self.spill_dir)
        self.engine = QueryEngine(catalog, self.budget, self.spill,
                                  self.control, mode=mode,
                                  skew_threshold=skew,
                                  join_overrides=overrides or {})
        self.store = ProofStore(self.proof_dir, self.budget, shards=shards)
        if write_fault is not None:
            self.store.write_fault = write_fault
        self.session = ProvenanceSession(
            self.store, tree, stable_node_ids(tree))
        self.engine.provenance = self.session

    def run(self):
        plan = self.engine.build(self.tree)
        self.rows = list(self.engine.run(plan))
        self.reader = ProofReader.open(self.proof_dir,
                                       self.session.run_id)
        return self.rows, self.reader

    def run_iter(self):
        plan = self.engine.build(self.tree)
        return iter(self.engine.run(plan))

    def close(self):
        self.spill.cleanup()


# ---------------------------------------------------------------- fixtures
class ThreeWayProvenanceTest(unittest.TestCase):
    """The headline scenario: filter -> project(collision) -> two joins ->
    count/sum, with independent witness enumeration."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rels = [
            ("T1", [{"name": "k", "type": "int64"},
                    {"name": "v", "type": "int64"}]),
            ("T2", [{"name": "k", "type": "int64"},
                    {"name": "w", "type": "int64"}]),
            ("T3", [{"name": "k", "type": "int64"},
                    {"name": "z", "type": "int64"}]),
        ]
        # Two T1 rows have IDENTICAL business values (collision) but keep
        # distinct rids; one filter-UNKNOWN row and one filter-FALSE row.
        self.t1 = [
            {"row_id": 1, "k": 7, "v": 100},
            {"row_id": 2, "k": 7, "v": 100},   # same business row as rid 1
            {"row_id": 3, "k": 7, "v": None},  # UNKNOWN under filter, dropped
            {"row_id": 4, "k": 0, "v": 9},     # FALSE under filter, dropped
        ]
        self.t2 = [
            {"row_id": 10, "k": 7, "w": 1},
            {"row_id": 11, "k": 7, "w": 2},
        ]
        self.t3 = [
            {"row_id": 100, "k": 7, "z": 5},
            {"row_id": 101, "k": 7, "z": 6},
        ]
        self.catalog, self.paths = write_catalog(
            self.tmp, "tw", rels,
            {"T1": self.t1, "T2": self.t2, "T3": self.t3})

    def tree(self):
        p = self.paths
        scan1 = {"node": "scan", "relation": "T1", "alias": "t1",
                 "file": p["T1"]}
        flt = {"node": "filter",
               # One predicate exercises all three truth values:
               # rid1/2 v=100 -> TRUE, rid3 v=NULL -> UNKNOWN,
               # rid4 v=9 -> FALSE.
               "expr": {"op": "gt", "left": "t1.v",
                        "right": {"literal": 50}},
               "input": scan1}
        # Collision: rid 1 and rid 2 project to identical business values
        # (t1.k=7, t1.v=100) but keep distinct rids and distinct proofs.
        proj = {"node": "project",
                "outputs": [["t1", "t1.k"], ["t1", "t1.v"]],
                "input": flt}
        j1 = {"node": "join", "id": "j1", "type": "inner",
              "left": proj,
              "right": {"node": "scan", "relation": "T2", "alias": "t2",
                        "file": p["T2"]},
              "on": [["t1.k", "t2.k"]]}
        j2 = {"node": "join", "id": "j2", "type": "inner",
              "left": j1,
              "right": {"node": "scan", "relation": "T3", "alias": "t3",
                        "file": p["T3"]},
              "on": [["t1.k", "t3.k"]]}
        return {"node": "aggregate", "keys": [], "sum": "t1.v",
                "input": j2}
    def _assert_oracle_equality(self, reader, tree):
        oracle = ProofOracle(tree)
        all_docs = {d["proof_ref"]: d for d in reader.iter_proofs()}
        # 1. every expected proof exists, no extras
        self.assertEqual(set(all_docs), set(oracle.by_ref),
                         msg="proof reference sets differ")
        for ref, exp in oracle.by_ref.items():
            doc = all_docs[ref]
            self.assertEqual(doc["node"], exp.node_id)
            self.assertEqual(doc["kind"], exp.kind)
            if exp.kind in ("count", "sum"):
                got_contrib = {
                    tuple(sorted(tuple(s) for s in c["sources"]))
                    for c in doc["contributions"]}
                self.assertEqual(got_contrib, set(exp.contributors),
                                 msg=f"{exp.kind} contributors for {ref}")
                self.assertEqual(doc["metric_value"], exp.value)
                if exp.kind == "sum":
                    self.assertEqual(
                        doc["excluded_null_contributors"],
                        exp.null_contributors)
                    got_vals = sorted(c["value"]
                                      for c in doc["contributions"])
                    self.assertEqual(got_vals, sorted(exp.sum_values))
            elif exp.kind == "semi":
                self.assertEqual(doc_match_tuples(doc),
                                 set(exp.match_sources))
            else:
                self.assertEqual(doc_sources(doc), set(exp.sources))
        return oracle

    def test_three_way_business_and_proofs_memory(self):
        h = ProofHarness(self.tree(), self.catalog, self.tmp,
                         mode="force_memory")
        try:
            rows, reader = h.run()
        finally:
            h.close()
        # Business: 2 T1 survivors x 2 T2 x 2 T3 = 8 joined rows.
        self.assertEqual(len(rows), 1)
        self.assertEqual(tuple(rows[0].values), (8, 800))  # count, sum(p.v)
        oracle = self._assert_oracle_equality(reader, self.tree())
        # Explicit hand check of sum: both colliding rows carry v=100, and
        # each reaches the aggregate through 2x2 = 4 join combinations.
        sum_ref = aggregate_proof_ref(oracle.root_id, "sum", ())
        sum_doc = reader.get_proof(sum_ref)
        self.assertEqual(sorted(c["value"] for c in sum_doc["contributions"]),
                         [100] * 8)
        # Projection collision: two distinct proofs for identical values.
        proj_node = [nid for nid, exps in oracle.expected.items()
                     if any(e.kind == "project" for e in exps)]
        self.assertEqual(len(proj_node), 1)
        proj_refs = {e.proof_ref for e in oracle.expected[proj_node[0]]}
        self.assertEqual(len(proj_refs), 2)  # rids 1 and 2, not merged
        # Filtered/UNKNOWN rows never reach any node above the filter,
        # though their scan proofs still exist as input records.
        for doc in reader.iter_proofs():
            if doc["node"] == proj_node[0] or doc["node"] in (
                    "j1", "j2", oracle.root_id):
                self.assertNotIn(("t1", 3), doc_sources(doc))
                self.assertNotIn(("t1", 4), doc_sources(doc))
        # ...but the scan proof of rid 3 exists as an input record only.
        scan3 = reader.get_proof(row_proof_ref(
            [n for n, exps in oracle.expected.items()
             if any(e.kind == "scan" for e in exps)][0],
            "scan", (("t1", 3),)))
        self.assertEqual(doc_sources(scan3), {("t1", 3)})

    def test_memory_and_disk_proof_sets_identical(self):
        sets = {}
        business = {}
        for mode in ("force_memory", "force_disk", "adaptive"):
            tmp = tempfile.mkdtemp(dir=self.tmp)
            ctrl = ControlPoint()
            if mode == "adaptive":
                ctrl.request_switch_after_outputs(2, "j1")
                ctrl.request_switch_after_outputs(3, "j2")
            h = ProofHarness(self.tree(), self.catalog, tmp, mode=mode,
                             control=ctrl)
            try:
                rows, reader = h.run()
                refs = {d["proof_ref"]: d for d in reader.iter_proofs()}
            finally:
                h.close()
            sets[mode] = {
                ref: (doc["kind"],
                      doc_sources(doc),
                      doc_match_tuples(doc)
                      if doc["kind"] == "semi" else None,
                      [c["sources"] for c in doc["contributions"]]
                      if doc["kind"] in ("count", "sum") else None)
                for ref, doc in refs.items()}
            business[mode] = [list(r.values) for r in rows]
        self.assertEqual(business["force_memory"], business["force_disk"])
        self.assertEqual(business["force_memory"], business["adaptive"])
        self.assertEqual(set(sets["force_memory"]),
                         set(sets["force_disk"]))
        self.assertEqual(set(sets["force_memory"]),
                         set(sets["adaptive"]))
        for ref in sets["force_memory"]:
            self.assertEqual(sets["force_memory"][ref],
                             sets["force_disk"][ref],
                             msg=f"proof {ref} differs memory vs disk")
            self.assertEqual(sets["force_memory"][ref],
                             sets["adaptive"][ref])

    def test_stable_refs_survive_new_process_layout(self):
        # Re-running in a fresh temp directory must yield identical refs:
        # no temp-dir/order/address dependence.
        h1 = ProofHarness(self.tree(), self.catalog,
                          tempfile.mkdtemp(dir=self.tmp),
                          mode="force_memory", shards=4)
        h2 = ProofHarness(self.tree(), self.catalog,
                          tempfile.mkdtemp(dir=self.tmp),
                          mode="force_disk", shards=16)
        try:
            r1, reader1 = h1.run()
            r2, reader2 = h2.run()
        finally:
            h1.close()
            h2.close()
        self.assertEqual(
            sorted(d["proof_ref"] for d in reader1.iter_proofs()),
            sorted(d["proof_ref"] for d in reader2.iter_proofs()))
        self.assertEqual(reader1.stats()["run_id"],
                         reader2.stats()["run_id"])
        self.assertEqual(reader1.stats()["run_id"],
                         expected_run_id(self.tree()))


class LeftAndSemiProofTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rels = [
            ("L", [{"name": "k", "type": "int64"},
                   {"name": "v", "type": "int64"}]),
            ("R", [{"name": "k", "type": "int64"},
                   {"name": "w", "type": "int64"}]),
        ]
        self.l = [
            {"row_id": 1, "k": 5, "v": 1},
            {"row_id": 2, "k": 6, "v": 2},   # no right match -> pad
            {"row_id": 3, "k": None, "v": 3},  # NULL key -> pad
        ]
        self.r = [
            {"row_id": 10, "k": 5, "w": 10},
            {"row_id": 11, "k": 5, "w": 11},
            {"row_id": 12, "k": 5, "w": 12},  # 3 matches for L1
            {"row_id": 13, "k": None, "w": 13},
        ]
        self.catalog, self.paths = write_catalog(
            self.tmp, "ls", rels, {"L": self.l, "R": self.r})

    def _tree(self, jtype):
        p = self.paths
        return {"node": "join", "id": "j1", "type": jtype,
                "left": {"node": "scan", "relation": "L", "alias": "l",
                         "file": p["L"]},
                "right": {"node": "scan", "relation": "R", "alias": "r",
                          "file": p["R"]},
                "on": [["l.k", "r.k"]]}

    def test_left_pad_proof_has_no_fabricated_right_rid(self):
        for mode in ("force_memory", "force_disk", "adaptive"):
            with self.subTest(mode=mode):
                ctrl = ControlPoint()
                if mode == "adaptive":
                    ctrl.request_switch_after_outputs(1, "j1")
                h = ProofHarness(self._tree("left"), self.catalog,
                                 tempfile.mkdtemp(dir=self.tmp),
                                 mode=mode, control=ctrl)
                try:
                    rows, reader = h.run()
                finally:
                    h.close()
                oracle = ProofOracle(self._tree("left"))
                self.assertEqual(len(rows), 5)  # 3 matches + 2 pads
                pad_docs = [d for d in reader.iter_proofs()
                            if d["kind"] == "pad"]
                self.assertEqual(len(pad_docs), 2)
                for doc in pad_docs:
                    self.assertTrue(doc["no_right_match"])
                    sources = doc_sources(doc)
                    self.assertTrue(all(a == "l" for a, _ in sources))
                    self.assertNotIn(("r", 13), sources)
                    # exactly one pad per unmatched left row
                    self.assertEqual(len(sources), 1)
                pad_lefts = sorted(next(iter(doc_sources(d)))
                                   for d in pad_docs)
                self.assertEqual(pad_lefts, [("l", 2), ("l", 3)])
                self._assert_set_equality(reader, oracle)

    def test_semi_proof_lists_all_real_matches_but_one_output(self):
        for mode in ("force_memory", "force_disk", "adaptive"):
            with self.subTest(mode=mode):
                ctrl = ControlPoint()
                if mode == "adaptive":
                    ctrl.request_switch_after_outputs(1, "j1")
                h = ProofHarness(self._tree("semi"), self.catalog,
                                 tempfile.mkdtemp(dir=self.tmp),
                                 mode=mode, control=ctrl)
                try:
                    rows, reader = h.run()
                finally:
                    h.close()
                # Only L1 matches; SEMI emits it exactly once.
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].rid_of("l"), 1)
                oracle = ProofOracle(self._tree("semi"))
                semi_docs = [d for d in reader.iter_proofs()
                             if d["kind"] == "semi"]
                self.assertEqual(len(semi_docs), 1)
                doc = semi_docs[0]
                # left witness + three real right-match combinations
                self.assertEqual(len(doc["witnesses"]), 4)
                roles = [w["role"] for w in doc["witnesses"]]
                self.assertEqual(roles.count("semi_left"), 1)
                self.assertEqual(roles.count("semi_match"), 3)
                combos = doc_match_tuples(doc)
                self.assertEqual(combos, {
                    (("l", 1), ("r", 10)),
                    (("l", 1), ("r", 11)),
                    (("l", 1), ("r", 12)),
                })
                self._assert_set_equality(reader, oracle)

    def _assert_set_equality(self, reader, oracle):
        docs = {d["proof_ref"]: d for d in reader.iter_proofs()}
        self.assertEqual(set(docs), set(oracle.by_ref))
        for ref, exp in oracle.by_ref.items():
            doc = docs[ref]
            if exp.kind == "semi":
                self.assertEqual(doc_match_tuples(doc),
                                 set(exp.match_sources))
            else:
                self.assertEqual(doc_sources(doc), set(exp.sources))


class CompositeNullKeyProofTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rels = [
            ("A", [{"name": "k1", "type": "int64"},
                   {"name": "k2", "type": "text"},
                   {"name": "v", "type": "int64"}]),
            ("B", [{"name": "k1", "type": "int64"},
                   {"name": "k2", "type": "text"}]),
        ]
        self.a = [
            {"row_id": 1, "k1": 1, "k2": "x", "v": 7},
            {"row_id": 2, "k1": 1, "k2": None, "v": 8},   # partial NULL
            {"row_id": 3, "k1": None, "k2": "x", "v": 9},
        ]
        self.b = [
            {"row_id": 10, "k1": 1, "k2": "x"},
            {"row_id": 11, "k1": 1, "k2": None},
        ]
        self.catalog, self.paths = write_catalog(
            self.tmp, "cn", rels, {"A": self.a, "B": self.b})

    def test_partial_null_composite_keys_never_witnessed(self):
        p = self.paths
        tree = {"node": "join", "id": "j1", "type": "inner",
                "left": {"node": "scan", "relation": "A", "alias": "a",
                         "file": p["A"]},
                "right": {"node": "scan", "relation": "B", "alias": "b",
                          "file": p["B"]},
                "on": [["a.k1", "b.k1"], ["a.k2", "b.k2"]]}
        for mode in ("force_memory", "force_disk"):
            h = ProofHarness(tree, self.catalog,
                             tempfile.mkdtemp(dir=self.tmp), mode=mode)
            try:
                rows, reader = h.run()
            finally:
                h.close()
            self.assertEqual(len(rows), 1)  # only (a1,b10)
            join_docs = [d for d in reader.iter_proofs()
                         if d["kind"] == "match"]
            self.assertEqual(len(join_docs), 1)
            self.assertEqual(doc_sources(join_docs[0]),
                             {("a", 1), ("b", 10)})
            for doc in reader.iter_proofs():
                if doc["kind"] == "match":
                    self.assertNotIn(("a", 2), doc_sources(doc))
                    self.assertNotIn(("a", 3), doc_sources(doc))
                    self.assertNotIn(("b", 11), doc_sources(doc))


class EmptyAggregateProofTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rels = [
            ("A", [{"name": "k", "type": "int64"},
                   {"name": "v", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"}]),
        ]
        self.empty = write_jsonl(
            os.path.join(self.tmp, "empty.jsonl"), [])
        self.b = write_jsonl(
            os.path.join(self.tmp, "b.jsonl"),
            [{"row_id": 1, "k": 1}])
        from join_exec import SchemaCatalog
        from join_exec.types import schema_from_json
        self.catalog = SchemaCatalog(
            [schema_from_json(r, c) for r, c in rels])

    def _tree(self, sum_ref):
        join = {"node": "join", "id": "j1", "type": "inner",
                "left": {"node": "scan", "relation": "A", "alias": "a",
                         "file": self.empty},
                "right": {"node": "scan", "relation": "B", "alias": "b",
                          "file": self.b},
                "on": [["a.k", "b.k"]]}
        return {"node": "aggregate", "keys": [], "sum": sum_ref,
                "input": join}

    def test_empty_count_zero_no_sources(self):
        tree = self._tree(None)
        h = ProofHarness(tree, self.catalog,
                         tempfile.mkdtemp(dir=self.tmp))
        try:
            rows, reader = h.run()
        finally:
            h.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(tuple(rows[0].values), (0,))
        oracle = ProofOracle(tree)
        count_ref = aggregate_proof_ref(oracle.root_id, "count", ())
        doc = reader.get_proof(count_ref)
        self.assertEqual(doc["contributions"], [])
        self.assertEqual(doc["metric_value"], 0)
        self.assertEqual(doc_sources(doc), set())

    def test_empty_sum_null_no_contributors(self):
        tree = self._tree("a.v")
        h = ProofHarness(tree, self.catalog,
                         tempfile.mkdtemp(dir=self.tmp))
        try:
            rows, reader = h.run()
        finally:
            h.close()
        self.assertEqual(tuple(rows[0].values), (0, None))
        oracle = ProofOracle(tree)
        sum_ref = aggregate_proof_ref(oracle.root_id, "sum", ())
        doc = reader.get_proof(sum_ref)
        self.assertIsNone(doc["metric_value"])
        self.assertEqual(doc["contributions"], [])
        self.assertEqual(doc["excluded_null_contributors"], 0)

    def test_all_null_sum_proof_excludes_null_inputs(self):
        rels = [
            ("A", [{"name": "k", "type": "int64"},
                   {"name": "v", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"}]),
        ]
        a = write_jsonl(os.path.join(self.tmp, "an.jsonl"), [
            {"row_id": 1, "k": 1, "v": None},
            {"row_id": 2, "k": 1, "v": None},
        ])
        b = write_jsonl(os.path.join(self.tmp, "bn.jsonl"), [
            {"row_id": 10, "k": 1}])
        tree = {"node": "aggregate", "keys": [], "sum": "a.v",
                "input": {"node": "join", "id": "j1", "type": "inner",
                          "left": {"node": "scan", "relation": "A",
                                   "alias": "a", "file": a},
                          "right": {"node": "scan", "relation": "B",
                                    "alias": "b", "file": b},
                          "on": [["a.k", "b.k"]]}}
        h = ProofHarness(tree, self.catalog,
                         tempfile.mkdtemp(dir=self.tmp))
        try:
            rows, reader = h.run()
        finally:
            h.close()
        self.assertEqual(tuple(rows[0].values), (2, None))
        oracle = ProofOracle(tree)
        sum_doc = reader.get_proof(
            aggregate_proof_ref(oracle.root_id, "sum", ()))
        self.assertEqual(sum_doc["contributions"], [])
        self.assertEqual(sum_doc["excluded_null_contributors"], 2)
        count_doc = reader.get_proof(
            aggregate_proof_ref(oracle.root_id, "count", ()))
        self.assertEqual(len(count_doc["contributions"]), 2)


class MidSwitchProofMergeTest(unittest.TestCase):
    """Proofs across a mid-output memory->disk switch form one complete set:
    no replay, no omission, suppressed duplicates not counted as sources."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rels = [
            ("A", [{"name": "k", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"}]),
        ]
        self.a = write_jsonl(os.path.join(self.tmp, "a.jsonl"),
                             [{"row_id": i, "k": 1} for i in range(6)])
        self.b = write_jsonl(os.path.join(self.tmp, "b.jsonl"),
                             [{"row_id": 100 + i, "k": 1}
                              for i in range(6)])
        from join_exec import SchemaCatalog
        from join_exec.types import schema_from_json
        self.catalog = SchemaCatalog(
            [schema_from_json(r, c) for r, c in rels])
        self.tree = {"node": "join", "id": "j1", "type": "inner",
                     "left": {"node": "scan", "relation": "A", "alias": "a",
                              "file": self.a},
                     "right": {"node": "scan", "relation": "B",
                               "alias": "b", "file": self.b},
                     "on": [["a.k", "b.k"]]}

    def test_switch_after_every_boundary(self):
        oracle = ProofOracle(self.tree)
        for after in range(1, 8):
            with self.subTest(after=after):
                ctrl = ControlPoint()
                ctrl.request_switch_after_outputs(after, "j1")
                h = ProofHarness(self.tree, self.catalog,
                                 tempfile.mkdtemp(dir=self.tmp),
                                 control=ctrl)
                try:
                    rows, reader = h.run()
                    stats = h.engine.stats()
                finally:
                    h.close()
                # 6x6 = 36 pairs, never more, never fewer.
                self.assertEqual(len(rows), 36)
                docs = {d["proof_ref"]: d for d in reader.iter_proofs()}
                self.assertEqual(len([d for d in docs.values()
                                      if d["kind"] == "match"]), 36)
                self.assertEqual(set(docs), set(oracle.by_ref))
                ev = stats["joins"][0]["switch_events"]
                self.assertEqual(len(ev), 1)
                self.assertEqual(
                    stats["joins"][0]["mode_final"], "disk")
                # rollup witnesses exactly 36, no suppressed duplication
                roll = reader.node_rollup()
                jroll = next(v for n, v in roll.items() if n == "j1")
                self.assertEqual(jroll["proofs"], 36)
                self.assertEqual(jroll["witnesses"], 36)
                # Each witness contributes two base sources.
                self.assertEqual(jroll["source_occurrences"], 36)
                self.assertEqual(jroll["distinct_sources"], 72)

    def test_left_switch_includes_pads_once(self):
        rels = [
            ("A", [{"name": "k", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"}]),
        ]
        a = write_jsonl(os.path.join(self.tmp, "al.jsonl"),
                        [{"row_id": 1, "k": 1}, {"row_id": 2, "k": 9}])
        b = write_jsonl(os.path.join(self.tmp, "bl.jsonl"),
                        [{"row_id": 10, "k": 1}])
        from join_exec import SchemaCatalog
        from join_exec.types import schema_from_json
        catalog = SchemaCatalog(
            [schema_from_json(r, c) for r, c in rels])
        tree = {"node": "join", "id": "j1", "type": "left",
                "left": {"node": "scan", "relation": "A", "alias": "a",
                         "file": a},
                "right": {"node": "scan", "relation": "B", "alias": "b",
                          "file": b},
                "on": [["a.k", "b.k"]]}
        ctrl = ControlPoint()
        ctrl.request_switch_after_outputs(1, "j1")
        h = ProofHarness(tree, catalog,
                         tempfile.mkdtemp(dir=self.tmp), control=ctrl)
        try:
            rows, reader = h.run()
        finally:
            h.close()
        self.assertEqual(len(rows), 2)
        kinds = sorted(d["kind"] for d in reader.iter_proofs()
                       if d["node"] == "j1")
        self.assertEqual(kinds, ["match", "pad"])


class HighFanoutProofTest(unittest.TestCase):
    """A single proof with more sources than the memory budget: it must be
    streamed, stay under budget, and remain exact."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rels = [
            ("A", [{"name": "k", "type": "int64"},
                   {"name": "v", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"}]),
        ]
        n_left, n_right = 400, 50
        a = write_jsonl(os.path.join(self.tmp, "hf_a.jsonl"), [
            {"row_id": i, "k": 1, "v": i % 7} for i in range(n_left)])
        b = write_jsonl(os.path.join(self.tmp, "hf_b.jsonl"), [
            {"row_id": 10_000 + i, "k": 1} for i in range(n_right)])
        from join_exec import SchemaCatalog
        from join_exec.types import schema_from_json
        self.catalog = SchemaCatalog(
            [schema_from_json(r, c) for r, c in rels])
        self.tree = {"node": "aggregate", "keys": [], "sum": "a.v",
                     "input": {"node": "join", "id": "j1",
                               "type": "inner",
                               "left": {"node": "scan", "relation": "A",
                                        "alias": "a", "file": a},
                               "right": {"node": "scan", "relation": "B",
                                         "alias": "b", "file": b},
                               "on": [["a.k", "b.k"]]}}
        self.expected_pairs = n_left * n_right

    def test_single_count_proof_streamed_under_tiny_budget(self):
        # Budget far smaller than the 20k source combinations; no
        # whole-result materialisation is permitted.
        h = ProofHarness(self.tree, self.catalog,
                         tempfile.mkdtemp(dir=self.tmp),
                         mode="adaptive", mem=12 * 1024, shards=4)
        try:
            rows, reader = h.run()
            peak = h.budget.peak_bytes
        finally:
            h.close()
        self.assertEqual(rows[0].values[0], self.expected_pairs)
        oracle = ProofOracle(self.tree)
        count_ref = aggregate_proof_ref(oracle.root_id, "count", ())
        doc = reader.get_proof(count_ref)
        self.assertEqual(len(doc["contributions"]), self.expected_pairs)
        # Every one of the 20k distinct combinations appears exactly once.
        combos = {
            tuple(sorted(tuple(s) for s in c["sources"]))
            for c in doc["contributions"]}
        self.assertEqual(len(combos), self.expected_pairs)
        self.assertEqual(combos, set(oracle.by_ref[count_ref].contributors))
        stats = reader.stats()
        self.assertGreaterEqual(stats["bytes_written"], 0)
        self.assertEqual(stats["witness_count"],
                         sum(v["witnesses"]
                             for v in reader.node_rollup().values()))
        # Peak logical reservation must stay within the hard budget even
        # though one proof holds 20k witnesses.
        self.assertLessEqual(peak, 12 * 1024)

    def test_high_fanout_semi_proof_spills_distinct(self):
        rels = [
            ("A", [{"name": "k", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"}]),
        ]
        n_right = 300
        a = write_jsonl(os.path.join(self.tmp, "semi_a.jsonl"),
                        [{"row_id": 1, "k": 1}])
        b = write_jsonl(os.path.join(self.tmp, "semi_b.jsonl"), [
            {"row_id": 500 + i, "k": 1} for i in range(n_right)])
        from join_exec import SchemaCatalog
        from join_exec.types import schema_from_json
        catalog = SchemaCatalog(
            [schema_from_json(r, c) for r, c in rels])
        tree = {"node": "join", "id": "j1", "type": "semi",
                "left": {"node": "scan", "relation": "A", "alias": "a",
                         "file": a},
                "right": {"node": "scan", "relation": "B", "alias": "b",
                          "file": b},
                "on": [["a.k", "b.k"]]}
        h = ProofHarness(tree, catalog,
                         tempfile.mkdtemp(dir=self.tmp),
                         mem=8 * 1024, shards=4)
        try:
            rows, reader = h.run()
        finally:
            h.close()
        self.assertEqual(len(rows), 1)
        semi = [d for d in reader.iter_proofs() if d["kind"] == "semi"]
        self.assertEqual(len(semi), 1)
        self.assertEqual(len(semi[0]["witnesses"]), n_right + 1)
        self.assertEqual(len(doc_match_tuples(semi[0])), n_right)
        oracle = ProofOracle(tree)
        self.assertEqual(doc_match_tuples(semi[0]),
                         set(oracle.by_ref[semi[0]["proof_ref"]]
                             .match_sources))


class FailureAndLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rels = [
            ("A", [{"name": "k", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"}]),
        ]
        self.a = write_jsonl(os.path.join(self.tmp, "fa.jsonl"), [
            {"row_id": i, "k": 1} for i in range(200)])
        self.b = write_jsonl(os.path.join(self.tmp, "fb.jsonl"), [
            {"row_id": 900 + i, "k": 1} for i in range(50)])
        from join_exec import SchemaCatalog
        from join_exec.types import schema_from_json
        self.catalog = SchemaCatalog(
            [schema_from_json(r, c) for r, c in rels])
        self.tree = {"node": "join", "id": "j1", "type": "inner",
                     "left": {"node": "scan", "relation": "A", "alias": "a",
                              "file": self.a},
                     "right": {"node": "scan", "relation": "B",
                               "alias": "b", "file": self.b},
                     "on": [["a.k", "b.k"]]}

    def test_proof_write_failure_aborts_and_removes_index(self):
        run_dir_root = tempfile.mkdtemp(dir=self.tmp)

        def fault(path):
            raise OSError("simulated proof disk full")

        h = ProofHarness(self.tree, self.catalog, run_dir_root,
                         mode="force_disk", mem=16 * 1024,
                         write_fault=fault)
        try:
            with self.assertRaises(ProofError):
                h.run()
        finally:
            h.close()
        # No manifest, no run directory, no index mistaken for complete.
        run_dir = os.path.join(h.proof_dir, h.session.run_id)
        self.assertFalse(os.path.exists(run_dir))
        # spill temp resources are fully cleaned (directory removed).
        self.assertFalse(os.path.exists(h.spill.temp_dir))

    def test_consumer_early_close_aborts_proofs(self):
        h = ProofHarness(self.tree, self.catalog,
                         tempfile.mkdtemp(dir=self.tmp),
                         mode="force_disk", mem=16 * 1024)
        gen = h.run_iter()
        next(gen)
        gen.close()
        h.close()
        run_dir = os.path.join(h.proof_dir, h.session.run_id)
        self.assertFalse(os.path.exists(run_dir))

    def test_cancellation_aborts_proofs(self):
        ctrl = ControlPoint()
        h = ProofHarness(self.tree, self.catalog,
                         tempfile.mkdtemp(dir=self.tmp),
                         mode="force_disk", mem=16 * 1024, control=ctrl)
        gen = h.run_iter()
        next(gen)
        ctrl.cancel("test cancel")
        with self.assertRaises(RunCancelled):
            list(gen)
        h.close()
        self.assertFalse(os.path.exists(
            os.path.join(h.proof_dir, h.session.run_id)))

    def test_corrupt_input_is_failure_not_partial_delivery(self):
        bad = os.path.join(self.tmp, "bad.jsonl")
        with open(bad, "w") as fh:
            fh.write('{"row_id": 1, "k": 1}\n{broken json\n')
        tree = {"node": "join", "id": "j1", "type": "inner",
                "left": {"node": "scan", "relation": "A", "alias": "a",
                         "file": bad},
                "right": {"node": "scan", "relation": "B", "alias": "b",
                          "file": self.b},
                "on": [["a.k", "b.k"]]}
        h = ProofHarness(tree, self.catalog,
                         tempfile.mkdtemp(dir=self.tmp))
        try:
            with self.assertRaises(BadRowError):
                h.run()
        finally:
            h.close()
        self.assertFalse(os.path.exists(
            os.path.join(h.proof_dir, h.session.run_id)))

    def test_committed_proofs_survive_spill_cleanup(self):
        run_root = tempfile.mkdtemp(dir=self.tmp)
        h = ProofHarness(self.tree, self.catalog, run_root,
                         mode="force_disk", mem=16 * 1024)
        try:
            rows, reader = h.run()
            run_id = h.session.run_id
            spill_dir = h.spill.temp_dir
        finally:
            h.close()
        # Spill temp directory is gone; proof directory still fully readable.
        self.assertFalse(os.path.exists(spill_dir))
        reader2 = ProofReader.open(h.proof_dir, run_id)
        self.assertEqual(
            sorted(d["proof_ref"] for d in reader2.iter_proofs()),
            sorted(d["proof_ref"] for d in reader.iter_proofs()))
        # Stats report bytes, counts and output correspondence.
        stats = reader2.stats()
        self.assertGreater(stats["bytes_written"], 0)
        self.assertEqual(stats["proof_count"],
                         sum(v["proofs"]
                             for v in reader2.node_rollup().values()))
        mappings, token = reader2.iter_output_mappings(page_size=64)
        all_maps = list(mappings)
        while token:
            mappings, token = reader2.iter_output_mappings(
                token=token, page_size=64)
            all_maps.extend(mappings)
        self.assertEqual(len(all_maps), len(rows))
        # Result order is not promised, but every output position 1..N
        # must have exactly one mapping row.
        self.assertEqual(sorted(m["seq"] for m in all_maps),
                         list(range(1, len(rows) + 1)))
        # Every mapping points at an existing proof.
        all_refs = {d["proof_ref"] for d in reader2.iter_proofs()}
        for m in all_maps:
            self.assertIn(m["row_proof_ref"], all_refs)


class PagingAndRefApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rels = [
            ("A", [{"name": "k", "type": "int64"}]),
            ("B", [{"name": "k", "type": "int64"}]),
        ]
        self.a = write_jsonl(os.path.join(self.tmp, "pa.jsonl"), [
            {"row_id": i, "k": i % 3} for i in range(40)])
        self.b = write_jsonl(os.path.join(self.tmp, "pb.jsonl"), [
            {"row_id": 700 + i, "k": i % 3} for i in range(30)])
        from join_exec import SchemaCatalog
        from join_exec.types import schema_from_json
        self.catalog = SchemaCatalog(
            [schema_from_json(r, c) for r, c in rels])
        self.tree = {"node": "join", "id": "j1", "type": "inner",
                     "left": {"node": "scan", "relation": "A", "alias": "a",
                              "file": self.a},
                     "right": {"node": "scan", "relation": "B",
                               "alias": "b", "file": self.b},
                     "on": [["a.k", "b.k"]]}

    def test_index_paging_covers_every_proof(self):
        h = ProofHarness(self.tree, self.catalog,
                         tempfile.mkdtemp(dir=self.tmp), shards=3)
        try:
            _, reader = h.run()
        finally:
            h.close()
        token = None
        entries = []
        while True:
            page, token = reader.iter_index(token=token, page_size=7)
            entries.extend(page)
            if token is None:
                break
        all_refs = {d["proof_ref"] for d in reader.iter_proofs()}
        self.assertEqual({e["p"] for e in entries}, all_refs)
        # Point lookup through the sparse directory.
        for ref in list(all_refs)[:25]:
            doc = reader.get_proof(ref)
            self.assertEqual(doc["proof_ref"], ref)
        with self.assertRaises(KeyError):
            reader.get_proof("rp_deadbeef")


if __name__ == "__main__":
    unittest.main()
