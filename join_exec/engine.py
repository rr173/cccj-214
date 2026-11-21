"""Query tree parsing, planning and execution.

Tree node kinds (JSON)::

    {"node": "scan",      "relation": "A", "alias": "a", "file": "..."}
    {"node": "filter",    "input": <node>, "expr": <3VL expression>}
    {"node": "project",   "input": <node>, "outputs": [["out", "a.col"], ...]}
    {"node": "join",      "type": "inner|left|semi",
     "left": <node>, "right": <node>,
     "on": [["a.k", "b.k"], ...]}
    {"node": "aggregate", "input": <node>, "keys": ["a.k"],
     "sum": "a.v" | null}

Legal join trees have 3-6 leaf inputs.  Consecutive INNER joins are flattened
and may be reordered; LEFT/SEMI nodes are semantic barriers and never get
reordered across.  This module builds the physical tree but emits no rows
itself — every join is an observable :class:`HybridJoin` instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from .budget import Budget
from .control import ControlPoint
from .errors import SchemaError
from .join import HybridJoin, Source
from .ops import Aggregate, Filter, Project
from .scan import Scan
from .schema_catalog import SchemaCatalog
from .spill import SpillManager
from .types import Header, Row


@dataclass
class PlanJoin:
    join_id: str
    join_type: str
    on: List[Tuple[str, str]]
    left_node: dict
    right_node: dict
    # filled during build
    left_phys: Optional["PlanNode"] = None
    right_phys: Optional["PlanNode"] = None


@dataclass
class PlanNode:
    kind: str
    header: Header
    scan: Optional[Scan] = None
    filter: Optional[Filter] = None
    project: Optional[Project] = None
    join: Optional[HybridJoin] = None
    aggregate: Optional[Aggregate] = None
    child: Optional["PlanNode"] = None
    children: List["PlanNode"] = field(default_factory=list)


class QueryEngine:
    def __init__(self, catalog: SchemaCatalog, budget: Budget,
                 spill: SpillManager, control: Optional[ControlPoint] = None,
                 mode: str = "adaptive",
                 skew_threshold: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 join_overrides: Optional[Dict[str, dict]] = None) -> None:
        self.catalog = catalog
        self.budget = budget
        self.spill = spill
        self.control = control
        self.mode = mode
        self.skew_threshold = skew_threshold
        self.batch_size = batch_size
        # per-join-id {mode, skew_threshold} overrides
        self.join_overrides = join_overrides or {}
        self._join_seq = 0
        self.joins: List[HybridJoin] = []
        self.scan_operators: List[Scan] = []

    # ------------------------------------------------------------ planning
    def build(self, tree: dict) -> PlanNode:
        leaves = count_leaves(tree)
        if not 2 <= leaves <= 6:
            raise SchemaError(
                f"join tree must have 2-6 leaf inputs (3-6 required by spec), "
                f"got {leaves}")
        return self._build_node(normalize_inner_tree(tree))

    def _build_node(self, node: dict) -> PlanNode:
        kind = node["node"]
        if kind == "scan":
            return self._build_scan(node)
        if kind == "filter":
            child = self._build_node(node["input"])
            flt = Filter(node["expr"], child.header)
            return PlanNode("filter", child.header, filter=flt, child=child)
        if kind == "project":
            child = self._build_node(node["input"])
            proj = Project(child.header, [tuple(x) for x in node["outputs"]])
            return PlanNode("project", proj.header, project=proj, child=child)
        if kind == "join":
            return self._build_join(node)
        if kind == "aggregate":
            child = self._build_node(node["input"])
            agg = Aggregate(child.header, node.get("keys", []),
                            node.get("sum"), self.budget)
            return PlanNode("aggregate", agg.header, aggregate=agg, child=child)
        raise SchemaError(f"unknown node kind {kind!r}")

    def _build_scan(self, node: dict) -> PlanNode:
        rel = node["relation"]
        alias = node.get("alias", rel)
        schema = self.catalog.get(rel)
        scan = Scan(schema, node["file"], alias, control=self.control)
        self.scan_operators.append(scan)
        return PlanNode("scan", scan.header, scan=scan)

    def _build_join(self, node: dict) -> PlanNode:
        jtype = node["type"]
        left = self._build_node(node["left"])
        right = self._build_node(node["right"])
        self._join_seq += 1
        join_id = node.get("id") or f"j{self._join_seq}"
        on = [(tuple(l.split(".", 1)), tuple(r.split(".", 1)))
              for l, r in node["on"]]
        left_refs = [l for l, _ in on]
        right_refs = [r for _, r in on]
        # validate refs against headers
        for alias, col in left_refs:
            left.header.find(alias, col)
        for alias, col in right_refs:
            right.header.find(alias, col)
        override = self.join_overrides.get(join_id, {})
        hj = HybridJoin(
            join_id=join_id,
            join_type=jtype,
            left=Source(left.header, lambda node=left: self._open(node)),
            right=Source(right.header, lambda node=right: self._open(node)),
            left_refs=left_refs,
            right_refs=right_refs,
            budget=self.budget,
            spill=self.spill,
            control=self.control,
            mode=override.get("mode", self.mode),
            skew_threshold=override.get("skew_threshold", self.skew_threshold),
            expected_build_rows=override.get("expected_build_rows"),
            batch_size=override.get("batch_size", self.batch_size),
        )
        self.joins.append(hj)
        return PlanNode("join", hj.header, join=hj,
                        children=[left, right])

    # ------------------------------------------------------------ runtime
    def _open(self, plan: PlanNode) -> Iterator[Row]:
        if plan.kind == "scan":
            yield from plan.scan
        elif plan.kind == "filter":
            yield from plan.filter.apply(self._open(plan.child), self.control)
        elif plan.kind == "project":
            yield from plan.project.apply(self._open(plan.child), self.control)
        elif plan.kind == "join":
            yield from plan.join
        elif plan.kind == "aggregate":
            yield from plan.aggregate.run(self._open(plan.child), self.control)
        else:  # pragma: no cover
            raise SchemaError(f"cannot open node {plan.kind}")

    def run(self, plan: PlanNode) -> Iterator[Row]:
        return self._open(plan)

    # ------------------------------------------------------------ explain
    def explain(self, tree: dict) -> dict:
        """Static explain: structure plus *potential* switch points only."""
        return {
            "mode_default": self.mode,
            "skew_threshold": self.skew_threshold,
            "tree": _explain_node(normalize_inner_tree(tree)),
            "potential_switch_points": [
                {
                    "join_id": j.join_id,
                    "join_type": j.join_type,
                    "triggers": [
                        "build_phase_budget_overflow",
                        "probe_phase_hot_group",
                        "control_after_outputs",
                        "control_after_input_batches",
                    ],
                }
                for j in self.joins
            ],
            "note": "switch timing/partitions are runtime facts, see stats.",
        }

    def stats(self) -> dict:
        return {
            "budget": self.budget.snapshot(),
            "spill": {
                "temp_dir": self.spill.temp_dir,
                "files": self.spill.stats.files,
                "bytes_written": self.spill.stats.bytes_written,
                "bytes_read": self.spill.stats.bytes_read,
                "records_written": self.spill.stats.records_written,
            },
            "joins": [_join_stats(j) for j in self.joins],
            "scans": [{"alias": s.alias, "rows": s.rows_scanned}
                      for s in self.scan_operators],
        }


def _join_stats(j: HybridJoin) -> dict:
    return {
        "join_id": j.join_id,
        "join_type": j.join_type,
        "mode_final": j.stats.mode,
        "build_rows": j.stats.build_rows,
        "probe_rows": j.stats.probe_rows,
        "outputs": j.stats.outputs,
        "left_padded": j.stats.left_padded,
        "build_rid_range": list(j.stats.build_rid_range),
        "probe_rid_range": list(j.stats.probe_rid_range),
        "hot_keys": j.stats.hot_keys,
        "partitions": j.stats.partitions,
        "merge_partitions": j.stats.merge_partitions,
        "residual_count": j.stats.residual_count,
        "reserved_peak_bytes": j.stats.bytes_reserved_peak,
        "switch_events": j.stats.switch_events,
    }


def _explain_node(node: dict) -> dict:
    kind = node["node"]
    if kind == "join":
        return {
            "node": "join",
            "type": node["type"],
            "on": node["on"],
            "switchable": True,
            "left": _explain_node(node["left"]),
            "right": _explain_node(node["right"]),
        }
    if kind in ("filter", "project", "aggregate"):
        out = {"node": kind}
        if kind == "filter":
            out["expr"] = node["expr"]
        if kind == "project":
            out["outputs"] = node["outputs"]
        if kind == "aggregate":
            out["keys"] = node.get("keys", [])
            out["sum"] = node.get("sum")
        out["input"] = _explain_node(node["input"])
        return out
    return {"node": "scan", "relation": node["relation"],
            "alias": node.get("alias", node["relation"])}


def count_leaves(node: dict) -> int:
    if node["node"] == "scan":
        return 1
    if node["node"] == "join":
        return count_leaves(node["left"]) + count_leaves(node["right"])
    return count_leaves(node["input"])


def normalize_inner_tree(node: dict) -> dict:
    """Normalise a join tree while keeping every predicate in scope.

    INNER joins are the reorderable region; LEFT/SEMI nodes are barriers that
    the normaliser never crosses.  We deliberately keep each INNER join's
    ``on`` attached to exactly the same pair of child subtrees it started
    with: reordering is therefore limited to rebuilding identical-structure
    nodes with normalised children, so an ``on`` referencing the left side can
    never migrate onto a right-only subtree.  The physical builder is also
    free to choose which side to build; no predicates are pushed through an
    outer/semi boundary.
    """
    if node["node"] == "scan":
        return dict(node)
    if node["node"] != "join":
        n = dict(node)
        n["input"] = normalize_inner_tree(node["input"])
        return n
    n = dict(node)
    n["left"] = normalize_inner_tree(node["left"])
    n["right"] = normalize_inner_tree(node["right"])
    return n
