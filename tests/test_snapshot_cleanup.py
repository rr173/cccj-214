"""Tests for the pre-delivery quarantine snapshot cleanup.

Coverage:

* the two small repeatable deliveries (one of two same-key targets removed vs
  both removed, cascading into downstream tables);
* complete cycles and cycles that lose external support;
* composite keys with NULL components and one row constrained by several rules;
* exact per-relation retained-row cross-check against an independent naive
  fixpoint oracle (:mod:`tests.snapshot_oracle`), including randomized
  schemas/rules/quarantine sets and order-independence checks;
* hot-key many-to-many fan-out with staging-cardinality evidence that no
  referrer x candidate pair table is ever built;
* long thin elimination chains with access-trace evidence that propagation is
  key-local and never rescans unrelated rows;
* failure semantics (fault after the first staged table / first output file:
  no delivery directory, untouched sources) and all input validation errors.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Sequence, Tuple

from join_exec import (
    BadRowError,
    CleanupError,
    DeliveryError,
    QuarantineError,
    RuleError,
    SchemaCatalog,
    SnapshotCleanup,
    clean_snapshot,
)
from join_exec.snapshot_cleanup import rules_from_json
from join_exec.types import schema_from_json
from tests.snapshot_oracle import oracle_clean


# ------------------------------------------------------------- helpers


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SnapshotHarness:
    """Builds catalog/inputs/rules in a fresh temp directory."""

    def __init__(self, schemas: Dict[str, List[dict]],
                 rows: Dict[str, List[dict]],
                 rule_specs: List[dict]):
        self.tmp = tempfile.mkdtemp(prefix="snapclean-")
        self.src_dir = os.path.join(self.tmp, "src")
        os.makedirs(self.src_dir)
        self.catalog = SchemaCatalog(
            [schema_from_json(rel, cols) for rel, cols in schemas.items()])
        self.paths: Dict[str, str] = {}
        for rel, rel_rows in rows.items():
            path = os.path.join(self.src_dir, rel + ".jsonl")
            write_jsonl(path, rel_rows)
            self.paths[rel] = path
        self.rows = rows
        self.rules = rules_from_json(rule_specs, self.catalog)

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def delivery_dir(self, name: str = "delivery") -> str:
        return os.path.join(self.tmp, name)

    def run(self, quarantine: Sequence[Tuple[str, int]],
            name: str = "delivery",
            inject_fault: str = None) -> Dict[str, Any]:
        return clean_snapshot(
            self.catalog, self.paths, self.rules, list(quarantine),
            self.delivery_dir(name)) if inject_fault is None else \
            SnapshotCleanup(
                self.catalog, self.paths, self.rules, list(quarantine),
                self.delivery_dir(name)).run(inject_fault=inject_fault)

    def kept_ids(self, name: str = "delivery") -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for rel in self.paths:
            path = os.path.join(self.delivery_dir(name), rel + ".jsonl")
            out[rel] = [row["row_id"] for row in read_jsonl(path)]
        return out

    def source_hashes(self) -> Dict[str, str]:
        return {rel: file_sha256(path) for rel, path in self.paths.items()}


def assert_matches_oracle(testcase: unittest.TestCase,
                          harness: SnapshotHarness,
                          quarantine: Sequence[Tuple[str, int]],
                          name: str = "delivery") -> None:
    summary = harness.run(quarantine, name=name)
    expected = oracle_clean(harness.paths, harness.rules, list(quarantine))
    got = harness.kept_ids(name=name)
    for rel in harness.paths:
        testcase.assertEqual(
            sorted(got[rel]), sorted(expected[rel]),
            msg=f"relation {rel} retained rows differ")
        testcase.assertEqual(
            summary["kept_rows"][rel], len(expected[rel]))
    return summary


# ------------------------------------------------------------- scenarios


class TwoTargetsCascadeTest(unittest.TestCase):
    """Same-key two targets: isolate one vs both, with a downstream table."""

    def setUp(self) -> None:
        # subjects S: two rows sharing key 10; facts F reference S via fk;
        # ledgers L reference facts via lf (a second required reference hop).
        schemas = {
            "S": [{"name": "k", "type": "int64"}],
            "F": [{"name": "fk", "type": "int64"},
                  {"name": "note", "type": "text"}],
            "L": [{"name": "lf", "type": "int64"}],
        }
        rows = {
            "S": [{"row_id": 1, "k": 10}, {"row_id": 2, "k": 10},
                  {"row_id": 3, "k": 20}],
            "F": [{"row_id": 10, "fk": 10, "note": "needs an S@10"},
                  {"row_id": 11, "fk": 20, "note": "needs S@20"},
                  {"row_id": 12, "fk": None, "note": "null fk, never matches"}],
            "L": [{"row_id": 100, "lf": 10}, {"row_id": 101, "lf": 11}],
        }
        rules = [
            {"rule_id": "f_to_s", "source_relation": "F",
             "target_relation": "S", "columns": [["fk", "k"]]},
            {"rule_id": "l_to_f", "source_relation": "L",
             "target_relation": "F", "columns": [["lf", "row_id"]]},
        ]
        self.h = SnapshotHarness(schemas, rows, rules)

    def tearDown(self) -> None:
        self.h.cleanup()

    def test_isolate_one_target_keeps_referrers(self) -> None:
        # Only S#1 removed: S#2 still carries key 10, so F#10 and L#100 stay.
        summary = assert_matches_oracle(self, self.h, [("S", 1)],
                                        name="one")
        self.assertEqual(
            self.h.kept_ids("one"),
            {"S": [2, 3], "F": [10, 11], "L": [100, 101]})
        # The output path API hands every relation file to downstream.
        for rel, path in summary["output_paths"].items():
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(os.path.dirname(path),
                             self.h.delivery_dir("one"))

    def test_isolate_both_targets_cascades_downstream(self) -> None:
        # Both S@10 rows gone: F#10 loses all support and is removed; L#100
        # then loses its only fact and is removed as well.  F#11/L#101 survive.
        assert_matches_oracle(self, self.h, [("S", 1), ("S", 2)],
                              name="both")
        self.assertEqual(
            self.h.kept_ids("both"),
            {"S": [3], "F": [11], "L": [101]})

    def test_null_key_row_removed_even_with_empty_quarantine(self) -> None:
        assert_matches_oracle(self, self.h, [], name="nullcase")
        kept = self.h.kept_ids("nullcase")
        self.assertNotIn(12, kept["F"])
        # Rows with no outgoing references are never removed by incoming refs.
        self.assertEqual(kept["S"], [1, 2, 3])


class CycleTest(unittest.TestCase):
    def setUp(self) -> None:
        schemas = {
            "acct": [{"name": "owner", "type": "int64"},
                     {"name": "peer", "type": "int64"}],
            "subject": [{"name": "k", "type": "int64"}],
        }
        rows = {
            "subject": [{"row_id": 900, "k": 1}],
            "acct": [
                {"row_id": 1, "owner": 1, "peer": 2},
                {"row_id": 2, "owner": 1, "peer": 1},
                # account 3 mutually references 4 but points at a missing
                # subject (owner = 99, no such subject row).
                {"row_id": 3, "owner": 99, "peer": 4},
                {"row_id": 4, "owner": 1, "peer": 3},
                # self-loop with complete external support.
                {"row_id": 5, "owner": 1, "peer": 5},
            ],
        }
        rules = [
            {"rule_id": "acct_owner_subject", "source_relation": "acct",
             "target_relation": "subject",
             "columns": [["owner", "k"]]},
            {"rule_id": "acct_peer_acct", "source_relation": "acct",
             "target_relation": "acct",
             "columns": [["peer", "row_id"]]},
        ]
        self.h = SnapshotHarness(schemas, rows, rules)

    def tearDown(self) -> None:
        self.h.cleanup()

    def test_complete_cycles_survive(self) -> None:
        assert_matches_oracle(self, self.h, [], name="cycles")
        kept = self.h.kept_ids("cycles")
        # Mutual pair 1<->2 and self-loop 5 keep each other; subject stays.
        self.assertEqual(kept["subject"], [900])
        self.assertEqual(kept["acct"], [1, 2, 5])

    def test_cycle_losing_external_support_collapses(self) -> None:
        # Quarantine the only subject row: every account loses its owner
        # support, including the mutually supporting 3<->4 pair and self-loop.
        assert_matches_oracle(self, self.h, [("subject", 900)],
                              name="broken")
        kept = self.h.kept_ids("broken")
        self.assertEqual(kept["subject"], [])
        self.assertEqual(kept["acct"], [])
        # Empty relation still ships a file.
        self.assertTrue(os.path.isfile(
            os.path.join(self.h.delivery_dir("broken"), "acct.jsonl")))
        self.assertEqual(
            read_jsonl(os.path.join(
                self.h.delivery_dir("broken"), "acct.jsonl")), [])

    def test_isolating_one_cycle_node_collapses_whole_cycle(self) -> None:
        # Remove acct#2: acct#1 loses its peer; acct#3/4 already unsupported;
        # acct#5's self-loop with intact owner support still survives.
        assert_matches_oracle(self, self.h, [("acct", 2)], name="node")
        self.assertEqual(self.h.kept_ids("node"),
                         {"subject": [900], "acct": [5]})


class CompositeAndMultiRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        schemas = {
            "parent": [{"name": "ka", "type": "int64"},
                       {"name": "kb", "type": "text"}],
            "child": [{"name": "fa", "type": "int64"},
                      {"name": "fb", "type": "text"},
                      {"name": "aux", "type": "int64"}],
            "aux": [{"name": "k", "type": "int64"}],
        }
        rows = {
            "parent": [
                {"row_id": 1, "ka": 1, "kb": "x"},
                {"row_id": 2, "ka": 1, "kb": "y"},
                {"row_id": 3, "ka": 2, "kb": None},
            ],
            "child": [
                # full composite match
                {"row_id": 10, "fa": 1, "fb": "x", "aux": 7},
                # partial NULL component: can never match
                {"row_id": 11, "fa": None, "fb": "x", "aux": 7},
                # no parent with (1, z)
                {"row_id": 12, "fa": 1, "fb": "z", "aux": 7},
                # composite match but missing aux support -> second rule fails
                {"row_id": 13, "fa": 1, "fb": "y", "aux": 8},
            ],
            "aux": [{"row_id": 7, "k": 7}],
        }
        rules = [
            {"rule_id": "composite", "source_relation": "child",
             "target_relation": "parent",
             "columns": [["fa", "ka"], ["fb", "kb"]]},
            {"rule_id": "aux_rule", "source_relation": "child",
             "target_relation": "aux", "columns": [["aux", "k"]]},
        ]
        self.h = SnapshotHarness(schemas, rows, rules)

    def tearDown(self) -> None:
        self.h.cleanup()

    def test_composite_null_and_multiple_rules(self) -> None:
        assert_matches_oracle(self, self.h, [], name="comp")
        self.assertEqual(
            self.h.kept_ids("comp"),
            {"parent": [1, 2, 3], "child": [10], "aux": [7]})

    def test_quarantine_one_composite_target_only(self) -> None:
        # Remove parent#1: child#10's composite (1,x) loses its only target.
        assert_matches_oracle(self, self.h, [("parent", 1)], name="q")
        self.assertEqual(self.h.kept_ids("q"),
                         {"parent": [2, 3], "child": [], "aux": [7]})


class DuplicateRowsAndOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        schemas = {
            "T": [{"name": "k", "type": "int64"}],
            "U": [{"name": "k", "type": "int64"}],
        }
        rows = {
            # Equal business values, distinct row_ids: duplicates must be
            # retained independently and never merged.
            "T": [{"row_id": 1, "k": 5}, {"row_id": 2, "k": 5},
                  {"row_id": 3, "k": 6}],
            "U": [{"row_id": 10, "k": 5}, {"row_id": 11, "k": 5}],
        }
        rules = [
            {"rule_id": "u_t", "source_relation": "U",
             "target_relation": "T", "columns": [["k", "k"]]}]
        self.h = SnapshotHarness(schemas, rows, rules)

    def tearDown(self) -> None:
        self.h.cleanup()

    def test_duplicates_not_merged_one_removed(self) -> None:
        assert_matches_oracle(self, self.h, [("T", 1)], name="d1")
        self.assertEqual(self.h.kept_ids("d1"),
                         {"T": [2, 3], "U": [10, 11]})

    def test_all_candidates_removed(self) -> None:
        assert_matches_oracle(self, self.h, [("T", 1), ("T", 2)],
                              name="d2")
        self.assertEqual(self.h.kept_ids("d2"),
                         {"T": [3], "U": []})

    def test_input_row_order_does_not_change_result(self) -> None:
        # Rewrite T and U in reversed order; the retained row_id sets must be
        # identical, and retained values/row_ids must be unchanged.
        for rel, rows in self.h.rows.items():
            write_jsonl(self.h.paths[rel], list(reversed(rows)))
        assert_matches_oracle(self, self.h, [("T", 1)], name="rev")
        ids_rev = self.h.kept_ids("rev")
        # Re-run with original ordering for comparison.
        for rel, rows in self.h.rows.items():
            write_jsonl(self.h.paths[rel], rows)
        assert_matches_oracle(self, self.h, [("T", 1)], name="orig")
        self.assertEqual(ids_rev, self.h.kept_ids("orig"))

    def test_rule_order_does_not_change_result(self) -> None:
        reversed_rules = list(reversed(
            [r.to_dict() for r in self.h.rules]))
        h2 = SnapshotHarness(
            {s.relation: [{"name": c.name, "type": c.dtype}
                          for c in s.columns]
             for s in self.h.catalog.all()},
            self.h.rows, reversed_rules)
        try:
            assert_matches_oracle(self, h2, [("T", 1)], name="ruleorder")
            self.assertEqual(h2.kept_ids("ruleorder"),
                             {"T": [2, 3], "U": [10, 11]})
        finally:
            h2.cleanup()


class RandomizedOracleTest(unittest.TestCase):
    def test_random_graphs_match_naive_oracle(self) -> None:
        rng = random.Random(20261122)
        for trial in range(40):
            n_rels = rng.randint(2, 4)
            rel_names = [f"R{i}" for i in range(n_rels)]
            schemas: Dict[str, List[dict]] = {}
            rows: Dict[str, List[dict]] = {}
            for rel in rel_names:
                n_cols = rng.randint(1, 2)
                col_types = [rng.choice(["int64", "text"])
                             for _ in range(n_cols)]
                schemas[rel] = [
                    {"name": f"c{j}", "type": t}
                    for j, t in enumerate(col_types)]
                rel_rows = []
                for rid in range(1, rng.randint(0, 12) + 1):
                    record = {"row_id": rid}
                    for j, t in enumerate(col_types):
                        if rng.random() < 0.2:
                            record[f"c{j}"] = None
                        elif t == "int64":
                            record[f"c{j}"] = rng.randint(0, 4)
                        else:
                            record[f"c{j}"] = rng.choice(["a", "b", "c"])
                    rel_rows.append(record)
                rows[rel] = rel_rows
            rule_specs: List[dict] = []
            n_rules = rng.randint(0, 4)
            for ri in range(n_rules):
                src = rng.choice(rel_names)
                tgt = rng.choice(rel_names)
                width = rng.randint(1, min(
                    len(schemas[src]), len(schemas[tgt])))
                src_cols = rng.sample(
                    [c["name"] for c in schemas[src]], width)
                # Pair only same-typed columns; skip rule if impossible.
                tgt_pool = [c for c in schemas[tgt]]
                pairs = []
                ok = True
                used = set()
                for sc in src_cols:
                    st = next(c["type"] for c in schemas[src]
                              if c["name"] == sc)
                    candidates = [c["name"] for c in tgt_pool
                                  if c["type"] == st and c["name"] not in used]
                    if not candidates:
                        ok = False
                        break
                    tc = rng.choice(candidates)
                    used.add(tc)
                    pairs.append([sc, tc])
                if not ok:
                    continue
                rule_specs.append({
                    "rule_id": f"g{ri}", "source_relation": src,
                    "target_relation": tgt, "columns": pairs})
            # Deduplicate rule ids while keeping order.
            seen = set()
            uniq_specs = []
            for spec in rule_specs:
                if spec["rule_id"] in seen:
                    continue
                seen.add(spec["rule_id"])
                uniq_specs.append(spec)
            h = SnapshotHarness(schemas, rows, uniq_specs)
            try:
                all_ids = [(rel, row["row_id"])
                           for rel, rel_rows in rows.items()
                           for row in rel_rows]
                quarantine = rng.sample(
                    all_ids, rng.randint(0, min(len(all_ids), 3)))
                assert_matches_oracle(
                    self, h, quarantine, name=f"trial{trial}")
            finally:
                h.cleanup()


class LocalityEvidenceTest(unittest.TestCase):
    def _open(self, h: SnapshotHarness, quarantine, name) -> SnapshotCleanup:
        cleanup = SnapshotCleanup(
            h.catalog, h.paths, h.rules, list(quarantine),
            h.delivery_dir(name))
        return cleanup.run_keeping_staging()

    @staticmethod
    def _chain_harness(chain_len: int, island: int) -> SnapshotHarness:
        # A thin chain L_N -> L_{N-1} -> ... -> L_1 -> root (row 0 points at
        # itself), plus an unrelated island whose rows all reference the
        # single existing anchor key.  Ids above the chain never collide.
        anchor = chain_len + 1
        schemas = {"node": [{"name": "ref", "type": "int64"}]}
        rows: List[dict] = [{"row_id": 0, "ref": 0}]
        for i in range(1, chain_len + 1):
            rows.append({"row_id": i, "ref": i - 1})
        rows.append({"row_id": anchor, "ref": anchor})
        for j in range(island):
            rows.append({"row_id": anchor + 1 + j, "ref": anchor})
        return SnapshotHarness(
            schemas, {"node": rows},
            [{"rule_id": "chain", "source_relation": "node",
              "target_relation": "node", "columns": [["ref", "row_id"]]}]), anchor

    def test_long_chain_does_not_rescan_unrelated_rows(self) -> None:
        # Moderate chain cross-checked against the naive oracle, plus full
        # locality evidence: quarantining root eliminates the whole chain one
        # lost key at a time and never reads the unrelated island.
        chain_len, island = 120, 150
        h, anchor = self._chain_harness(chain_len, island)
        try:
            cleanup = self._open(h, [("node", 0)], "chain")
            try:
                kept = h.kept_ids("chain")
                expected = oracle_clean(
                    h.paths, h.rules, [("node", 0)])
                self.assertEqual(sorted(kept["node"]),
                                 sorted(expected["node"]))
                # The whole chain collapses; the island (including the
                # anchor) is untouched.
                self.assertNotIn(chain_len, kept["node"])
                self.assertIn(anchor, kept["node"])
                self.assertEqual(len(kept["node"]), island + 1)

                trace = cleanup.access_trace()
                # Trace rows are count-aggregated; total localized accesses
                # must be linear in the chain length, never proportional to
                # the whole snapshot (a rescan-per-round design would multiply
                # this by the number of rounds and touch every row each round).
                kill_total = sum(e["count"] for e in trace
                                 if e["event"] == "referrers_killed")
                dec_total = sum(e["count"] for e in trace
                                if e["event"] == "support_decrement")
                self.assertLessEqual(kill_total, chain_len + 1)
                self.assertLessEqual(
                    dec_total, chain_len + 1 + len(h.catalog.all()))
                # And no event may reference the island anchor or island row
                # ids.  Only row_id-bearing payloads name specific rows;
                # count fields are volumes rather than references.
                for event in trace:
                    payload = json.loads(event["key_json"]) if event[
                        "key_json"] else {}
                    referenced = payload.get("row_id")
                    if referenced is not None:
                        self.assertNotEqual(referenced, anchor)
                        self.assertFalse(
                            anchor < referenced <= anchor + island)

                # Locality evidence: row tables (the large per-row stores)
                # are always reached through an index (a partial-index
                # traversal "SCAN ... USING INDEX idx_*" only walks live rows
                # and never touches the table); only the compact support
                # tables (one row per distinct target key, never a
                # referrer x candidate expansion) may be scanned directly.
                for entry in cleanup.explain_local_lookups():
                    plan_lines = entry["plan"]
                    for line in plan_lines:
                        if "SCAN" not in line:
                            continue
                        if "USING INDEX" in line:
                            continue  # index-only traversal, not a table scan
                        scanned = line.split("SCAN", 1)[1].strip().split()[0]
                        self.assertTrue(
                            scanned.startswith("support_"),
                            msg=f"row table scanned in {entry}")
                    self.assertTrue(
                        any("SEARCH" in line or "USING INDEX" in line
                            for line in plan_lines),
                        msg=f"no index access in {entry}")
            finally:
                cleanup.discard_staging()
        finally:
            h.cleanup()

    def test_large_chain_scales_without_roundwise_rescans(self) -> None:
        # A long, thin elimination chain (no O(rounds x rows) oracle here):
        # the key-local propagation must reach fixpoint without any per-round
        # full-relation scan, and the large unrelated island must survive
        # untouched.  The moderate chain test above supplies the oracle
        # cross-check; this one supplies the scale/locality evidence at a
        # practical size.
        chain_len, island = 2000, 2000
        h, anchor = self._chain_harness(chain_len, island)
        try:
            cleanup = self._open(h, [("node", 0)], "bigchain")
            try:
                kept = h.kept_ids("bigchain")
                # Exactly the island remains: the anchor plus its referrers.
                self.assertEqual(len(kept["node"]), island + 1)
                self.assertIn(anchor, kept["node"])
                self.assertNotIn(0, kept["node"])
                self.assertNotIn(chain_len, kept["node"])
                trace = cleanup.access_trace()
                kill_total = sum(e["count"] for e in trace
                                 if e["event"] == "referrers_killed")
                dec_total = sum(e["count"] for e in trace
                                if e["event"] == "support_decrement")
                # A thin chain enqueues one new dead row per hop.  Root is
                # removed directly by quarantine (no referrer kill), so the N
                # chain rows 1..N each produce exactly one kill; every hop
                # produces one decrement.  Accesses are linear in chain
                # length, never multiplied by island size or by a per-round
                # full rescan.
                self.assertEqual(kill_total, chain_len)
                self.assertLessEqual(dec_total, chain_len + 1)
                # No localized event may name the island anchor or island
                # row ids.  Only quarantine/row-reference payloads carry row
                # ids; count fields (dead_rows/rows) are volumes, not refs.
                for event in trace:
                    payload = json.loads(event["key_json"]) if event[
                        "key_json"] else {}
                    referenced = payload.get("row_id")
                    if referenced is not None:
                        self.assertNotEqual(referenced, anchor)
                        self.assertFalse(
                            anchor < referenced <= anchor + island,
                            msg=f"island row reached by propagation: {event}")
                # Staging holds one row per input row plus one row per
                # distinct target key (including keys already reduced to
                # zero), never per referrer/candidate pair.
                card = cleanup.staging_cardinality()
                self.assertEqual(
                    card["rows_r0"], chain_len + island + 2)
                self.assertLessEqual(
                    card["support_u0"], chain_len + island + 2)
            finally:
                cleanup.discard_staging()
        finally:
            h.cleanup()

    def test_large_hot_key_fanout_is_linear_and_pair_free(self) -> None:
        # Scale evidence for the "single popular key" case: M referrers and N
        # same-key target rows, where removing every target cascades to every
        # referrer in one set-based step.  Staging stays O(M+N); a pairwise
        # design would need O(M*N) edges.  No full relation is re-scanned.
        m, n = 12000, 2000
        schemas = {
            "ref": [{"name": "k", "type": "int64"}],
            "src": [{"name": "k", "type": "int64"}],
        }
        ref_rows = [{"row_id": i + 1, "k": 7} for i in range(n)]
        src_rows = [{"row_id": i + 1, "k": 7} for i in range(m)]
        h = SnapshotHarness(schemas, {"ref": ref_rows, "src": src_rows}, [
            {"rule_id": "hot", "source_relation": "src",
             "target_relation": "ref", "columns": [["k", "k"]]}])
        try:
            quarantine = [("ref", i + 1) for i in range(n)]
            cleanup = SnapshotCleanup(
                h.catalog, h.paths, h.rules, quarantine,
                h.delivery_dir("bigfanout")).run_keeping_staging()
            try:
                kept = h.kept_ids("bigfanout")
                self.assertEqual(kept["src"], [])
                self.assertEqual(kept["ref"], [])
                card = cleanup.staging_cardinality()
                # One support row for the single hot key; row tables hold one
                # row per input row.  Nothing approaches M*N = 2.5e8.
                self.assertEqual(card["support_u0"], 1)
                self.assertEqual(card["rows_r0"], n)
                self.assertEqual(card["rows_r1"], m)
                self.assertLess(
                    sum(card.values()), m + n + 4,
                    msg="staging expanded beyond O(M+N)")
            finally:
                cleanup.discard_staging()
        finally:
            h.cleanup()

    def test_hot_key_many_to_many_has_no_pairwise_staging(self) -> None:
        # M referrers and N candidate target rows on one hot key.  Removing
        # one target must leave every referrer alive; staging must hold O(M+N)
        # rows, not O(M*N).
        m, n = 200, 50
        schemas = {
            "ref": [{"name": "k", "type": "int64"}],
            "src": [{"name": "k", "type": "int64"}],
        }
        ref_rows = [{"row_id": i + 1, "k": 7} for i in range(n)]
        src_rows = [{"row_id": i + 1, "k": 7} for i in range(m)]
        h = SnapshotHarness(schemas, {"ref": ref_rows, "src": src_rows}, [
            {"rule_id": "hot", "source_relation": "src",
             "target_relation": "ref", "columns": [["k", "k"]]}])
        try:
            cleanup = self._open(h, [("ref", 1)], "hot")
            try:
                kept = h.kept_ids("hot")
                self.assertEqual(len(kept["src"]), m)
                self.assertEqual(len(kept["ref"]), n - 1)
                card = cleanup.staging_cardinality()
                # Only the two row tables, one single-row support table and
                # an empty queue exist: no M*N edge table anywhere.
                self.assertEqual(card["support_u0"], 1)
                self.assertEqual(card["rows_r0"], n)
                self.assertEqual(card["rows_r1"], m)
                self.assertEqual(card["work_queue"], 0)
                total_staging = sum(card.values())
                self.assertLessEqual(total_staging, m + n + 2)
                # No trace event may enumerate referrer/candidate pairs.
                trace = cleanup.access_trace()
                for event in trace:
                    payload = event["key_json"] or ""
                    self.assertNotIn(str(m * n), payload)
            finally:
                cleanup.discard_staging()

            # Removing all N candidates cascades to all M referrers, and the
            # support table never holds more than one distinct key.
            cleanup = SnapshotCleanup(
                h.catalog, h.paths, h.rules,
                [("ref", i + 1) for i in range(n)],
                h.delivery_dir("hot-all-gone")).run_keeping_staging()
            try:
                kept = h.kept_ids("hot-all-gone")
                self.assertEqual(kept["src"], [])
                self.assertEqual(kept["ref"], [])
                card = cleanup.staging_cardinality()
                self.assertLessEqual(card["support_u0"], 1)
            finally:
                cleanup.discard_staging()
        finally:
            h.cleanup()


class FailureAndValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        schemas = {
            "A": [{"name": "k", "type": "int64"}],
            "B": [{"name": "k", "type": "int64"}],
        }
        rows = {
            "A": [{"row_id": 1, "k": 10}, {"row_id": 2, "k": 11}],
            "B": [{"row_id": 10, "k": 10}, {"row_id": 11, "k": 11}],
        }
        rules = [{"rule_id": "ab", "source_relation": "A",
                  "target_relation": "B", "columns": [["k", "k"]]}]
        self.h = SnapshotHarness(schemas, rows, rules)

    def tearDown(self) -> None:
        self.h.cleanup()

    def test_fault_after_first_staged_table_leaves_nothing(self) -> None:
        before = self.h.source_hashes()
        with self.assertRaises(DeliveryError):
            self.h.run([("B", 10)], name="broken1",
                       inject_fault="after_first_staged_table")
        self.assertFalse(os.path.exists(self.h.delivery_dir("broken1")))
        # No sibling delivery artifact and no temp working directory remain.
        self.assertEqual(
            [name for name in os.listdir(self.h.tmp)
             if name.startswith(".snapshot-clean-")], [])
        self.assertEqual(self.h.source_hashes(), before)

    def test_fault_after_first_output_leaves_nothing(self) -> None:
        before = self.h.source_hashes()
        with self.assertRaises(DeliveryError):
            self.h.run([("B", 10)], name="broken2",
                       inject_fault="after_first_output")
        self.assertFalse(os.path.exists(self.h.delivery_dir("broken2")))
        self.assertEqual(self.h.source_hashes(), before)

    def test_existing_delivery_dir_refused(self) -> None:
        target = self.h.delivery_dir("exists")
        os.makedirs(target)
        with self.assertRaises(DeliveryError):
            self.h.run([], name="exists")

    def test_unknown_quarantine_relation(self) -> None:
        with self.assertRaises(QuarantineError):
            self.h.run([("nope", 1)], name="q1")

    def test_unknown_quarantine_row(self) -> None:
        with self.assertRaises(QuarantineError):
            self.h.run([("A", 999)], name="q2")

    def test_duplicate_quarantine_entry(self) -> None:
        with self.assertRaises(QuarantineError):
            self.h.run([("A", 1), ("A", 1)], name="q3")

    def test_bad_row_and_duplicate_row_id(self) -> None:
        bad_dir = tempfile.mkdtemp(dir=self.h.tmp)
        try:
            bad_path = os.path.join(bad_dir, "A.jsonl")
            with open(bad_path, "w", encoding="utf-8") as fh:
                fh.write('{"row_id": 1, "k": 1}\n')
                fh.write("not-json\n")
            paths = dict(self.h.paths)
            paths["A"] = bad_path
            with self.assertRaises(BadRowError):
                clean_snapshot(self.h.catalog, paths, self.h.rules, [],
                               self.h.delivery_dir("badrow"))
            self.assertFalse(os.path.exists(
                self.h.delivery_dir("badrow")))

            with open(bad_path, "w", encoding="utf-8") as fh:
                fh.write('{"row_id": 1, "k": 1}\n')
                fh.write('{"row_id": 1, "k": 2}\n')
            with self.assertRaises(BadRowError):
                clean_snapshot(self.h.catalog, paths, self.h.rules, [],
                               self.h.delivery_dir("dupid"))
        finally:
            shutil.rmtree(bad_dir, ignore_errors=True)

    def test_invalid_rules(self) -> None:
        base = {
            "source_relation": "A", "target_relation": "B",
            "columns": [["k", "k"]]}
        # unknown relation
        with self.assertRaises(RuleError):
            rules_from_json([dict(base, source_relation="Z")],
                            self.h.catalog)
        # unknown column
        with self.assertRaises(RuleError):
            rules_from_json([dict(base, columns=[["nope", "k"]])],
                            self.h.catalog)
        # mismatched types
        schemas = {
            "A": [{"name": "t", "type": "text"}],
            "B": [{"name": "k", "type": "int64"}],
        }
        cat2 = SchemaCatalog(
            [schema_from_json(rel, cols) for rel, cols in schemas.items()])
        with self.assertRaises(RuleError):
            rules_from_json(
                [{"rule_id": "x", "source_relation": "A",
                  "target_relation": "B", "columns": [["t", "k"]]}], cat2)
        # empty column list / malformed pair / duplicate rule id
        with self.assertRaises(RuleError):
            rules_from_json([dict(base, columns=[])], self.h.catalog)
        with self.assertRaises(RuleError):
            rules_from_json([dict(base, columns=[["k"]])], self.h.catalog)
        with self.assertRaises(RuleError):
            rules_from_json([dict(base, rule_id="d"),
                             dict(base, rule_id="d")], self.h.catalog)

    def test_missing_input_file(self) -> None:
        paths = dict(self.h.paths)
        paths["A"] = os.path.join(self.h.tmp, "missing.jsonl")
        with self.assertRaises(CleanupError):
            clean_snapshot(self.h.catalog, paths, self.h.rules, [],
                           self.h.delivery_dir("missing"))


class DeliveryContentTest(unittest.TestCase):
    def test_row_values_and_row_ids_preserved_verbatim(self) -> None:
        schemas = {
            "T": [{"name": "k", "type": "int64"},
                  {"name": "label", "type": "text"}],
            "U": [{"name": "k", "type": "int64"}],
        }
        rows = {
            "T": [{"row_id": 5, "k": 1, "label": "héllo→x"},
                  {"row_id": 6, "k": 2, "label": ""}],
            "U": [{"row_id": 7, "k": 1}, {"row_id": 8, "k": 2}],
        }
        rules = [{"rule_id": "u_t", "source_relation": "U",
                  "target_relation": "T", "columns": [["k", "k"]]}]
        h = SnapshotHarness(schemas, rows, rules)
        try:
            h.run([("T", 6)], name="out")
            t_path = os.path.join(h.delivery_dir("out"), "T.jsonl")
            self.assertEqual(read_jsonl(t_path), [rows["T"][0]])
            u_rows = read_jsonl(
                os.path.join(h.delivery_dir("out"), "U.jsonl"))
            self.assertEqual(u_rows, [rows["U"][0]])
        finally:
            h.cleanup()


if __name__ == "__main__":
    unittest.main()
