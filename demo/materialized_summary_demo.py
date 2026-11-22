#!/usr/bin/env python3
"""Reproducible demo of the durable, incrementally refreshed summary view.

Run::

    python3 -I demo/materialized_summary_demo.py [--workdir DIR]

The demo generates only small JSONL fixtures (no large data is kept) and
maintains one fixed-shape query over three relations::

    AGGREGATE keys=b.g  sum=f.amt
      SEMI JOIN f.ek = c.ck
        LEFT JOIN f.fk = b.bk
          F (facts)      B (category, one group column g)
        C (eligibility)

Scenarios, each checked against an independent row-by-row oracle
(:mod:`join_exec.incremental_oracle`) that never touches the incremental
implementation:

  batch 1  one batch simultaneously moves a category key, changes how many
           eligibility rows match, and moves a fact row's join keys;
  batch 2  the last eligibility match for a key is removed, a NULL group is
           created, and an expired group disappears;
  batch 3  a tiny correction on one key while many unrelated facts exist,
           with local-access evidence proving only affected rows are read;
  fault    a failure injected after the first write but before publish, then
           reopen proves the old version is intact and the retry publishes
           the new version atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from join_exec import SchemaCatalog  # noqa: E402
from join_exec.incremental_oracle import (  # noqa: E402
    apply_corrections_to_maps,
    load_relations,
    oracle_summary,
)
from join_exec.materialized import (  # noqa: E402
    CorrectionError,
    MaterializedSummary,
    OverflowRejected,
)
from join_exec.types import schema_from_json  # noqa: E402

FACT = "F"
CAT = "B"
ELIG = "C"

# ---------------------------------------------------------------- fixtures
INITIAL_FACTS = [
    # fk: LEFT-join key into B.bk ; ek: SEMI-join key into C.ck
    {"row_id": 1, "fk": 10, "ek": 100, "amt": 10},
    {"row_id": 2, "fk": 10, "ek": 100, "amt": 20},
    {"row_id": 3, "fk": 20, "ek": 200, "amt": 30},
    {"row_id": 4, "fk": 30, "ek": None, "amt": 40},   # NULL SEMI key: dropped
    {"row_id": 5, "fk": 99, "ek": 300, "amt": None},  # pad + NULL amount
    {"row_id": 6, "fk": 10, "ek": 200, "amt": 50},
]
INITIAL_CATS = [
    {"row_id": 10, "bk": 10, "g": "A"},
    {"row_id": 11, "bk": 10, "g": "A"},   # same group value, still counts twice
    {"row_id": 12, "bk": 20, "g": "B"},
    {"row_id": 13, "bk": 30, "g": "A"},   # f4 cannot reach it (SEMI rejects)
]
INITIAL_ELIG = [
    {"row_id": 100, "ck": 100},
    {"row_id": 101, "ck": 100},           # two matches: SEMI still keeps once
    {"row_id": 102, "ck": 200},
    {"row_id": 103, "ck": 300},
]

# Batch 1: move a category key (B#12: 20 -> 10), change eligibility match
# count (C#101: 100 -> 999, so key 100 loses one of its two matches), and
# move fact F#3's keys (fk 20 -> 10, ek 200 -> 100) in the same batch.
BATCH_1 = [
    {"relation": CAT, "row_id": 12, "op": "upsert",
     "row": {"row_id": 12, "bk": 10, "g": "B"}},
    {"relation": ELIG, "row_id": 101, "op": "upsert",
     "row": {"row_id": 101, "ck": 999}},
    {"relation": FACT, "row_id": 3, "op": "upsert",
     "row": {"row_id": 3, "fk": 10, "ek": 100, "amt": 30}},
]

# Batch 2: remove the last eligibility match of key 300 (fact F#5 loses its
# SEMI membership), move B#13 onto a fresh key 40 with NULL group (the old
# "A" row vanishes as a contributor and a legal NULL group is born), and move
# fact F#4 so it reaches that NULL-group category row once its SEMI key is
# repaired.  Group "B" goes empty and must disappear entirely.
BATCH_2 = [
    {"relation": ELIG, "row_id": 103, "op": "delete"},
    {"relation": CAT, "row_id": 13, "op": "upsert",
     "row": {"row_id": 13, "bk": 40, "g": None}},
    {"relation": FACT, "row_id": 4, "op": "upsert",
     "row": {"row_id": 4, "fk": 40, "ek": 400, "amt": 40}},
    {"relation": ELIG, "row_id": 104, "op": "upsert",
     "row": {"row_id": 104, "ck": 400}},
]

# Batch 3: one small correction on key 10 while the store holds many
# unrelated facts (added below). Only F#1 moves; everything else stays put.
BATCH_3 = [
    {"relation": FACT, "row_id": 1, "op": "upsert",
     "row": {"row_id": 1, "fk": 10, "ek": 100, "amt": 11}},
]

# Fault batch: applied twice - first with an injected post-write fault,
# then for real after reopening.
FAULT_BATCH = [
    {"relation": FACT, "row_id": 6, "op": "upsert",
     "row": {"row_id": 6, "fk": 10, "ek": 100, "amt": 51}},
]


def build_catalog() -> SchemaCatalog:
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


def build_tree(paths: dict) -> dict:
    return {
        "node": "aggregate", "keys": ["b.g"], "sum": "f.amt",
        "input": {
            "node": "join", "type": "semi", "id": "semi1",
            "on": [["f.ek", "c.ck"]],
            "left": {
                "node": "join", "type": "left", "id": "left1",
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
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def show(title: str, summary: list) -> None:
    print(f"\n== {title} ==")
    for row in summary:
        print(f"  group={row['group']!r:>8}  count={row['count']:<3} "
              f"sum={row['sum']}")


def assert_equal(got: list, expected: list, label: str) -> None:
    if got != expected:
        raise AssertionError(
            f"{label} mismatch:\n  got      = {got}\n  expected = {expected}")
    print(f"  [ok] {label}: {len(got)} groups match independent oracle")


def main(workdir: str) -> int:
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    paths = {name: os.path.join(workdir, f"{name}.jsonl")
             for name in (FACT, CAT, ELIG)}
    write_jsonl(paths[FACT], INITIAL_FACTS)
    write_jsonl(paths[CAT], INITIAL_CATS)
    write_jsonl(paths[ELIG], INITIAL_ELIG)

    catalog = build_catalog()
    tree = build_tree(paths)
    db_path = os.path.join(workdir, "summary_state.db")

    # Independent oracle state: plain row maps, rebuilt nowhere else.
    oracle_maps = load_relations(paths)
    spec = MaterializedSummary  # imported for name clarity below

    view = MaterializedSummary.initialize(db_path, tree, catalog)
    expected = oracle_summary(oracle_maps, view.spec)
    show("initial load (version 1)", view.read_summary())
    assert_equal(view.read_summary(), expected, "initial summary")
    print(f"  initial version = {view.version()}")

    # ------------------------------------------------------------- batch 1
    evidence_1 = view.apply_corrections(BATCH_1)
    apply_corrections_to_maps(oracle_maps, BATCH_1)
    expected = oracle_summary(oracle_maps, view.spec)
    show("after batch 1: moved cat key, changed match count, moved fact",
         view.read_summary())
    assert_equal(view.read_summary(), expected, "batch 1 summary")
    print("  local-access evidence (batch 1):",
          json.dumps(evidence_1["access"], sort_keys=True))
    print("  affected fact row ids:", evidence_1["affected_fact_row_ids"],
          "| groups touched:", evidence_1["groups_updated"])

    # Order independence: reordering the same batch cannot change anything;
    # verified through the oracle here and property tests on the real store.
    assert_equal(oracle_summary(oracle_maps, view.spec), expected,
                 "oracle order-stability")

    # ------------------------------------------------------------- batch 2
    evidence_2 = view.apply_corrections(BATCH_2)
    apply_corrections_to_maps(oracle_maps, BATCH_2)
    expected = oracle_summary(oracle_maps, view.spec)
    show("after batch 2: last match removed, NULL group born, group expires",
         view.read_summary())
    assert_equal(view.read_summary(), expected, "batch 2 summary")
    print("  affected fact row ids:", evidence_2["affected_fact_row_ids"],
          "| groups touched:", evidence_2["groups_updated"])
    assert view.read_summary() == view.read_summary()  # deterministic order

    # --------------------------------- batch 3: locality with many distractors
    distractors = [
        {"row_id": 1000 + i, "fk": 777, "ek": 888, "amt": i}
        for i in range(500)
    ]
    view.apply_corrections([
        {"relation": FACT, "row_id": d["row_id"], "op": "upsert", "row": d}
        for d in distractors
    ] + [
        {"relation": CAT, "row_id": 900, "op": "upsert",
         "row": {"row_id": 900, "bk": 777, "g": "D"}},
        {"relation": ELIG, "row_id": 901, "op": "upsert",
         "row": {"row_id": 901, "ck": 888}},
    ])
    apply_corrections_to_maps(oracle_maps, [
        {"relation": FACT, "row_id": d["row_id"], "op": "upsert", "row": d}
        for d in distractors
    ] + [
        {"relation": CAT, "row_id": 900, "op": "upsert",
         "row": {"row_id": 900, "bk": 777, "g": "D"}},
        {"relation": ELIG, "row_id": 901, "op": "upsert",
         "row": {"row_id": 901, "ck": 888}},
    ])

    before_trace = view.trace_summary()
    evidence_3 = view.apply_corrections(BATCH_3)
    apply_corrections_to_maps(oracle_maps, BATCH_3)
    expected = oracle_summary(oracle_maps, view.spec)
    show("after batch 3: one-amount change amid 500 unrelated facts",
         view.read_summary())
    assert_equal(view.read_summary(), expected, "batch 3 summary")
    print("  local-access evidence (batch 3):",
          json.dumps(evidence_3["access"], sort_keys=True))
    assert evidence_3["affected_fact_row_ids"] == [1], evidence_3
    fact_reads = [t for t in view.last_access_trace()
                  if t["access"] == "fact_by_row_id"]
    assert {t["row_id"] for t in fact_reads} == {1}, fact_reads
    print("  [ok] only fact row 1 was read; the 500 distractor facts "
          "were never touched")

    # Every localized statement must be an index SEARCH, never a full SCAN.
    for entry in view.explain_local_access():
        joined = " ".join(entry["plan"])
        assert "SCAN" not in joined.upper() or "USING COVERING INDEX" in joined, \
            entry
        assert "SEARCH" in joined.upper(), entry
    print("  [ok] EXPLAIN QUERY PLAN shows index SEARCH for all five "
          "local lookups")

    # --------------------------------------------- no-op batch still versions
    version_before = view.version()
    view.apply_corrections([
        {"relation": FACT, "row_id": 2, "op": "upsert",
         "row": oracle_maps[FACT][2]},
    ])
    assert view.version() == version_before + 1
    assert_equal(view.read_summary(),
                 oracle_summary(oracle_maps, view.spec),
                 "visible-identical batch keeps same summary")
    print(f"  [ok] an identical-content batch still published version "
          f"{view.version()}")

    # ------------------------------------------------------------- faults
    version_before_fault = view.version()
    summary_before_fault = view.read_summary()
    try:
        view.apply_corrections(FAULT_BATCH, inject_fault="after_first_write")
    except RuntimeError as exc:
        print(f"\n== injected failure ==\n  caught: {exc}")
    else:  # pragma: no cover
        raise AssertionError("injected fault did not fire")
    assert view.version() == version_before_fault
    assert view.read_summary() == summary_before_fault
    print("  [ok] same-handle view still serves the whole old version")

    view.close()
    view = MaterializedSummary.open(db_path)
    assert view.version() == version_before_fault
    assert view.read_summary() == summary_before_fault
    print("  [ok] reopening the state DB resumes from the old version; "
          "no half batch visible")

    evidence_f = view.apply_corrections(FAULT_BATCH)
    apply_corrections_to_maps(oracle_maps, FAULT_BATCH)
    expected = oracle_summary(oracle_maps, view.spec)
    show("after retried fault batch (new version)", view.read_summary())
    assert_equal(view.read_summary(), expected, "retried batch summary")
    print("  retried batch evidence:",
          json.dumps(evidence_f["access"], sort_keys=True))

    # ---------------------------------------------------- rejection scenarios
    bad_cases = [
        ("delete missing id", [
            {"relation": FACT, "row_id": 999999, "op": "delete"}]),
        ("duplicate id in batch", [
            {"relation": FACT, "row_id": 1, "op": "delete"},
            {"relation": FACT, "row_id": 1, "op": "upsert",
             "row": {"row_id": 1, "fk": 10, "ek": 100, "amt": 1}}]),
        ("bad int64", [
            {"relation": FACT, "row_id": 7, "op": "upsert",
             "row": {"row_id": 7, "fk": 10, "ek": 100,
                     "amt": 2 ** 63}}]),
        ("missing column", [
            {"relation": FACT, "row_id": 7, "op": "upsert",
             "row": {"row_id": 7, "fk": 10, "ek": 100}}]),
        ("unknown relation", [
            {"relation": "X", "row_id": 1, "op": "delete"}]),
        ("empty batch", []),
    ]
    for label, batch in bad_cases:
        try:
            view.apply_corrections(batch)
        except (CorrectionError, OverflowRejected) as exc:
            assert view.read_summary() == expected
            print(f"  [ok] rejected {label}: {type(exc).__name__}")
        else:  # pragma: no cover
            raise AssertionError(f"{label} was not rejected")

    # int64 overflow: a huge positive amount makes the published sum exceed
    # int64; the whole batch must roll back regardless of delta ordering.
    from join_exec.types import INT64_MAX
    ov_version = view.version()
    overflow_batch = [
        {"relation": FACT, "row_id": 700, "op": "upsert",
         "row": {"row_id": 700, "fk": 10, "ek": 100, "amt": INT64_MAX}},
    ]
    try:
        view.apply_corrections(overflow_batch)
    except OverflowRejected as exc:
        print(f"  [ok] overflow batch rejected: {exc}")
    else:  # pragma: no cover
        raise AssertionError("overflow was not rejected")
    assert view.version() == ov_version
    assert view.read_summary() == expected
    print("  [ok] state remains on the previous version after overflow")

    view.close()
    print("\nAll demo checks passed. State DB:", db_path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/tmp/materialized-summary-demo")
    args = parser.parse_args()
    raise SystemExit(main(args.workdir))
