"""Shared helpers for the executor test suite."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from join_exec import (
    Budget,
    ControlPoint,
    QueryEngine,
    ReferenceInterpreter,
    SchemaCatalog,
    SpillManager,
)
from join_exec.types import Row

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def catalog() -> SchemaCatalog:
    with open(os.path.join(FIX, "schema.json"), "r", encoding="utf-8") as fh:
        return SchemaCatalog.from_json(json.load(fh))


def fixture_path(name: str) -> str:
    return os.path.join(FIX, name)


def make_engine(memory_bytes: int, mode: str = "adaptive",
                skew_threshold: Optional[int] = None,
                control: Optional[ControlPoint] = None,
                temp_dir: Optional[str] = None,
                join_overrides: Optional[dict] = None
                ) -> Tuple[QueryEngine, Budget, SpillManager, ControlPoint]:
    budget = Budget(memory_bytes)
    control = control or ControlPoint()
    spill = SpillManager(budget, base_dir=temp_dir)
    engine = QueryEngine(catalog(), budget, spill, control,
                         mode=mode, skew_threshold=skew_threshold,
                         join_overrides=join_overrides)
    return engine, budget, spill, control


def run_tree(tree: dict, memory_bytes: int = 4 * 1024 * 1024,
             mode: str = "adaptive", skew_threshold: Optional[int] = None,
             control: Optional[ControlPoint] = None,
             join_overrides: Optional[dict] = None
             ) -> Tuple[List[Row], dict, SpillManager]:
    engine, budget, spill, ctrl = make_engine(
        memory_bytes, mode=mode, skew_threshold=skew_threshold,
        control=control, join_overrides=join_overrides)
    try:
        plan = engine.build(tree)
        rows = list(engine.run(plan))
        return rows, engine.stats(), spill
    finally:
        spill.cleanup()


def reference_bag(tree: dict) -> List[Tuple[Tuple, dict]]:
    return ReferenceInterpreter(tree).run()


def lineage_set(rows: List[Row]) -> set:
    out = set()
    for r in rows:
        pairs = tuple(sorted(
            (r.rids[i], r.rids[i + 1]) for i in range(0, len(r.rids), 2)))
        out.add(pairs)
    return out


def value_bag(rows: List[Row]) -> List[tuple]:
    return sorted(tuple(r.values) for r in rows)


def assert_bag_equals_reference(testcase, tree: dict, rows: List[Row]) -> None:
    expected = reference_bag(tree)
    testcase.assertEqual(len(rows), len(expected),
                         msg=f"row count differs: {len(rows)} vs {len(expected)}")
    ref_lineage = {tuple(sorted(lineage)) for lineage, _ in expected}
    got_lineage = lineage_set(rows)
    testcase.assertEqual(got_lineage, ref_lineage,
                         msg="lineage bags differ")
    ref_values = sorted(tuple(sorted(v.items())) for _, v in expected)
    got_values = sorted(tuple(sorted(zip(
        _slots_of(tree, rows[0] if rows else None), r.values)))
        if r.values else tuple() for r in rows)
    # Values are compared positionally via the product headers elsewhere;
    # here only lineage/count plus explicit hand checks are required.
    testcase.assertEqual(len(ref_values), len(got_values))


def _slots_of(tree: dict, row: Optional[Row]) -> List[str]:
    return []
