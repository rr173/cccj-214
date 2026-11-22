#!/usr/bin/env python3
"""Reproducible demo of the pre-delivery snapshot quarantine cleanup.

Run::

    python3 -I demo/snapshot_cleanup_demo.py [--workdir DIR]

The demo generates only small JSONL fixtures (no large data is kept) and
walks the contract end to end:

  delivery 1  two same-key targets, only one quarantined: referrers survive;
  delivery 2  both targets quarantined: the referrer and its downstream table
              are eliminated together;
  cycles      a complete mutual-reference cycle survives, while a cycle that
              loses its external support collapses;
  composite   a composite key with a NULL component and one row constrained by
              two rules at once;
  faults      a write failure injected after the first staged table and after
              the first output file leaves no delivery directory and the
              source files byte-identical;
  executor    the delivered JSONL paths are handed to the existing query
              executor unchanged - its query tree and refresh behaviour are
              not modified, only the scan file paths point at the delivery.

Every retained set is independently recomputed by the naive row-by-row
oracle (:mod:`tests.snapshot_oracle`), which shares no cleanup machinery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from join_exec import (  # noqa: E402
    DeliveryError,
    QueryEngine,
    Budget,
    SchemaCatalog,
    SnapshotCleanup,
    clean_snapshot,
    SpillManager,
)
from join_exec.snapshot_cleanup import rules_from_json  # noqa: E402
from join_exec.types import schema_from_json  # noqa: E402
from tests.snapshot_oracle import oracle_clean  # noqa: E402


def write_jsonl(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def heading(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


class DemoWorld:
    """Builds the three-relation fixture used by deliveries 1 and 2."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.src = os.path.join(root, "src")
        os.makedirs(self.src, exist_ok=True)
        self.schemas = {
            "S": [{"name": "k", "type": "int64"}],
            "F": [{"name": "fk", "type": "int64"},
                  {"name": "note", "type": "text"}],
            "L": [{"name": "lf", "type": "int64"}],
        }
        self.rows = {
            # Two subjects sharing key 10 (same-key candidates), one at 20.
            "S": [{"row_id": 1, "k": 10}, {"row_id": 2, "k": 10},
                  {"row_id": 3, "k": 20}],
            # F#10 needs an S@10; F#11 needs S@20; F#12 has a NULL fk.
            "F": [{"row_id": 10, "fk": 10, "note": "needs S@10"},
                  {"row_id": 11, "fk": 20, "note": "needs S@20"},
                  {"row_id": 12, "fk": None, "note": "null fk"}],
            # Ledgers require their fact rows.
            "L": [{"row_id": 100, "lf": 10}, {"row_id": 101, "lf": 11}],
        }
        self.catalog = SchemaCatalog(
            [schema_from_json(rel, cols)
             for rel, cols in self.schemas.items()])
        self.paths: dict = {}
        for rel, rows in self.rows.items():
            path = os.path.join(self.src, rel + ".jsonl")
            write_jsonl(path, rows)
            self.paths[rel] = path
        self.rules = rules_from_json([
            {"rule_id": "f_to_s", "source_relation": "F",
             "target_relation": "S", "columns": [["fk", "k"]]},
            {"rule_id": "l_to_f", "source_relation": "L",
             "target_relation": "F", "columns": [["lf", "row_id"]]},
        ], self.catalog)
        self.source_hashes = {p: sha256(p) for p in self.paths.values()}

    def deliver(self, quarantine: list, name: str) -> dict:
        target = os.path.join(self.root, name)
        summary = clean_snapshot(self.catalog, self.paths, self.rules,
                                 quarantine, target)
        expected = oracle_clean(self.paths, self.rules, quarantine)
        for rel in self.paths:
            got = sorted(r["row_id"] for r in read_jsonl(
                summary["output_paths"][rel]))
            assert got == sorted(expected[rel]), (rel, got, expected[rel])
        return summary


def show_delivery(summary: dict, root: str) -> None:
    for rel, path in summary["output_paths"].items():
        rows = read_jsonl(path)
        ids = [row["row_id"] for row in rows]
        print(f"  {rel}.jsonl -> {os.path.relpath(path, root)}")
        print(f"      kept row_ids {ids} (kept={summary['kept_rows'][rel]}, "
              f"total={summary['total_rows'][rel]})")


def demo_two_deliveries(world: DemoWorld) -> None:
    heading("DELIVERY 1: quarantine only ONE of two same-key targets")
    s1 = world.deliver([("S", 1)], "delivery-one")
    show_delivery(s1, world.root)
    assert sorted(r["row_id"] for r in read_jsonl(
        s1["output_paths"]["F"])) == [10, 11]
    print("  -> S#2 still carries key 10, so F#10 and ledger L#100 survive.")

    heading("DELIVERY 2: quarantine BOTH same-key targets (cascade)")
    s2 = world.deliver([("S", 1), ("S", 2)], "delivery-two")
    show_delivery(s2, world.root)
    assert read_jsonl(s2["output_paths"]["F"]) == [
        {"row_id": 11, "fk": 20, "note": "needs S@20"}]
    assert [r["row_id"] for r in read_jsonl(
        s2["output_paths"]["L"])] == [101]
    print("  -> key 10 loses all support: F#10 is removed and L#100, having "
          "lost its only fact, is removed with it.  F#11/L#101 survive.")


def demo_cycles(root: str) -> None:
    heading("CYCLES: complete cycle survives; unsupported cycle collapses")
    catalog = SchemaCatalog([
        schema_from_json("acct", [
            {"name": "owner", "type": "int64"},
            {"name": "peer", "type": "int64"}]),
        schema_from_json("subject", [{"name": "k", "type": "int64"}]),
    ])
    rows = {
        "subject": [{"row_id": 900, "k": 1}],
        "acct": [
            {"row_id": 1, "owner": 1, "peer": 2},   # mutual pair 1 <-> 2
            {"row_id": 2, "owner": 1, "peer": 1},
            {"row_id": 3, "owner": 99, "peer": 4},  # missing subject owner
            {"row_id": 4, "owner": 1, "peer": 3},
            {"row_id": 5, "owner": 1, "peer": 5},   # supported self-loop
        ],
    }
    paths = {}
    for rel, rel_rows in rows.items():
        path = os.path.join(root, "src", rel + ".jsonl")
        write_jsonl(path, rel_rows)
        paths[rel] = path
    rules = rules_from_json([
        {"rule_id": "owner_subject", "source_relation": "acct",
         "target_relation": "subject", "columns": [["owner", "k"]]},
        {"rule_id": "peer_acct", "source_relation": "acct",
         "target_relation": "acct", "columns": [["peer", "row_id"]]},
    ], catalog)

    full = clean_snapshot(catalog, paths, rules, [],
                          os.path.join(root, "cycle-complete"))
    kept = {rel: [r["row_id"] for r in read_jsonl(p)]
            for rel, p in full["output_paths"].items()}
    print("  no quarantine:        acct kept =", kept["acct"])
    assert kept["acct"] == [1, 2, 5]

    broken = clean_snapshot(catalog, paths, rules, [("subject", 900)],
                            os.path.join(root, "cycle-broken"))
    kept_b = {rel: [r["row_id"] for r in read_jsonl(p)]
              for rel, p in broken["output_paths"].items()}
    print("  subject quarantined:  acct kept =", kept_b["acct"],
          "(empty file still present:",
          os.path.isfile(broken["output_paths"]["acct"]), ")")
    assert kept_b["acct"] == []
    print("  -> the mutual pair and the self-loop keep each other alive, but "
          "removing the external subject removes every account.")


def demo_composite_and_multi_rule(root: str) -> None:
    heading("COMPOSITE NULL KEY + ONE ROW UNDER TWO RULES")
    catalog = SchemaCatalog([
        schema_from_json("parent", [
            {"name": "ka", "type": "int64"}, {"name": "kb", "type": "text"}]),
        schema_from_json("child", [
            {"name": "fa", "type": "int64"}, {"name": "fb", "type": "text"},
            {"name": "aux", "type": "int64"}]),
        schema_from_json("aux", [{"name": "k", "type": "int64"}]),
    ])
    rows = {
        "parent": [{"row_id": 1, "ka": 1, "kb": "x"},
                   {"row_id": 2, "ka": 1, "kb": "y"},
                   {"row_id": 3, "ka": 2, "kb": None}],
        "child": [
            {"row_id": 10, "fa": 1, "fb": "x", "aux": 7},   # both rules pass
            {"row_id": 11, "fa": None, "fb": "x", "aux": 7},  # NULL component
            {"row_id": 12, "fa": 1, "fb": "z", "aux": 7},  # no parent key
            {"row_id": 13, "fa": 1, "fb": "y", "aux": 8},  # aux rule fails
        ],
        "aux": [{"row_id": 7, "k": 7}],
    }
    paths = {}
    for rel, rel_rows in rows.items():
        path = os.path.join(root, "src", rel + ".jsonl")
        write_jsonl(path, rel_rows)
        paths[rel] = path
    rules = rules_from_json([
        {"rule_id": "composite", "source_relation": "child",
         "target_relation": "parent",
         "columns": [["fa", "ka"], ["fb", "kb"]]},
        {"rule_id": "aux_rule", "source_relation": "child",
         "target_relation": "aux", "columns": [["aux", "k"]]},
    ], catalog)
    summary = clean_snapshot(catalog, paths, rules, [],
                             os.path.join(root, "composite"))
    kept = sorted(r["row_id"] for r in read_jsonl(
        summary["output_paths"]["child"]))
    print("  child kept =", kept)
    assert kept == [10]
    print("  -> only child#10 satisfies BOTH rules; the NULL composite, the "
          "unmatched composite and the missing aux support are all removed.")


def demo_faults(world: DemoWorld) -> None:
    heading("FAILURE ATOMICITY: injected write faults leave nothing behind")
    before = dict(world.source_hashes)
    for fault in ("after_first_staged_table", "after_first_output"):
        target = os.path.join(world.root, "fault-" + fault)
        try:
            SnapshotCleanup(world.catalog, world.paths, world.rules,
                            [("S", 1)], target).run(inject_fault=fault)
        except DeliveryError as exc:
            print(f"  injected {fault!r}: {type(exc).__name__}")
        assert not os.path.exists(target), "delivery dir survived a failure"
        assert {p: sha256(p) for p in world.paths.values()} == before
        leftovers = [n for n in os.listdir(world.root)
                     if n.startswith(".snapshot-clean-")]
        assert not leftovers, f"temp working dir leaked: {leftovers}"
        print(f"      delivery dir absent, no temp dir left, "
              f"sources byte-identical")


def demo_hand_off_to_executor(world: DemoWorld, root: str) -> None:
    heading("HAND-OFF: delivered paths run through the existing executor")
    summary = world.deliver([("S", 1), ("S", 2)], "delivery-for-executor")
    # Run the existing executor against the delivered files: the same
    # QueryEngine/HybridJoin used by normal queries, with both scan "file"
    # entries pointing at the delivery directory.  The engine, query shape and
    # refresh logic are untouched - only the input paths move.
    tree = {
        "node": "join", "id": "j1", "type": "inner",
        "left": {"node": "scan", "relation": "F", "alias": "f",
                 "file": summary["output_paths"]["F"]},
        "right": {"node": "scan", "relation": "S", "alias": "s",
                  "file": summary["output_paths"]["S"]},
        "on": [["f.fk", "s.k"]],
    }
    budget = Budget(4 * 1024 * 1024)
    spill = SpillManager(budget, base_dir=os.path.join(root, "spill"))
    engine = QueryEngine(world.catalog, budget, spill)
    try:
        plan = engine.build(tree)
        pairs = [tuple(sorted(
            (row.rids[i], row.rids[i + 1])
            for i in range(0, len(row.rids), 2)))
            for row in engine.run(plan)]
        print("  executor inner-join lineage pairs over the delivery:")
        for pair in pairs:
            print("     ", pair)
        # Delivered F#11 (fk=20) matches delivered S#3 (k=20) exactly once.
        assert pairs == [tuple(sorted([("f", 11), ("s", 3)]))]
    finally:
        spill.cleanup()
    print("  -> the existing executor consumed the delivered JSONL with no "
          "change to its query or refresh behaviour.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()
    import tempfile
    root = args.workdir or tempfile.mkdtemp(prefix="snapshot-cleanup-demo-")
    os.makedirs(root, exist_ok=True)
    print("workdir:", os.path.abspath(root))
    world = DemoWorld(root)
    demo_two_deliveries(world)
    demo_cycles(root)
    demo_composite_and_multi_rule(root)
    demo_faults(world)
    demo_hand_off_to_executor(world, root)
    heading("ALL DEMO CHECKS PASSED")
    print("deliveries and source fixtures kept under:", os.path.abspath(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
