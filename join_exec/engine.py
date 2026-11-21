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
    node_id: str = ""
    tree_node: Optional[dict] = None
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
        # Active provenance session (None for plain runs).
        self.provenance = None
        # Pre-order node ids keyed by tree-node object identity.
        self._node_id_map: Dict[int, str] = {}

    # ------------------------------------------------------------ planning
    def build(self, tree: dict) -> PlanNode:
        leaves = count_leaves(tree)
        if not 2 <= leaves <= 6:
            raise SchemaError(
                f"join tree must have 2-6 leaf inputs (3-6 required by spec), "
                f"got {leaves}")
        normalized = normalize_inner_tree(tree)
        self._node_id_map = _assign_node_ids(normalized)
        return self._build_node(normalized)

    def _build_node(self, node: dict) -> PlanNode:
        kind = node["node"]
        node_id = self._node_id_map[id(node)]
        if kind == "scan":
            return self._build_scan(node, node_id)
        if kind == "filter":
            child = self._build_node(node["input"])
            flt = Filter(node["expr"], child.header)
            return PlanNode("filter", child.header, node_id=node_id,
                            tree_node=node, filter=flt, child=child)
        if kind == "project":
            child = self._build_node(node["input"])
            proj = Project(child.header, [tuple(x) for x in node["outputs"]])
            return PlanNode("project", proj.header, node_id=node_id,
                            tree_node=node, project=proj, child=child)
        if kind == "join":
            return self._build_join(node, node_id)
        if kind == "aggregate":
            child = self._build_node(node["input"])
            agg = Aggregate(child.header, node.get("keys", []),
                            node.get("sum"), self.budget)
            return PlanNode("aggregate", agg.header, node_id=node_id,
                            tree_node=node, aggregate=agg, child=child)
        raise SchemaError(f"unknown node kind {kind!r}")

    def _build_scan(self, node: dict, node_id: str) -> PlanNode:
        rel = node["relation"]
        alias = node.get("alias", rel)
        schema = self.catalog.get(rel)
        scan = Scan(schema, node["file"], alias, control=self.control)
        self.scan_operators.append(scan)
        return PlanNode("scan", scan.header, node_id=node_id,
                        tree_node=node, scan=scan)

    def _build_join(self, node: dict, node_id: str) -> PlanNode:
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
        if self.provenance is not None:
            hj.proof_sink = self.provenance.join_sink(node_id)
        self.joins.append(hj)
        return PlanNode("join", hj.header, node_id=node_id,
                        tree_node=node, join=hj,
                        children=[left, right])

    # ------------------------------------------------------------ runtime
    def _open(self, plan: PlanNode) -> Iterator[Row]:
        if self.provenance is not None:
            yield from self._open_pv(plan)
            return
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

    def _open_pv(self, plan: PlanNode) -> Iterator[Row]:
        """Same physical operators, wrapped so every emitted row gets a
        proof.  Business values and row counts are untouched."""
        pv = self.provenance
        if plan.kind == "scan":
            for row in plan.scan:
                pv.record_scan(plan.node_id, row)
                yield row
        elif plan.kind == "filter":
            role = "filter_pass"
            for row in plan.filter.apply(
                    self._open_pv(plan.child), self.control):
                pv.record_passthrough(plan.node_id, "filter", row, role)
                yield row
        elif plan.kind == "project":
            for row in plan.project.apply(
                    self._open_pv(plan.child), self.control):
                pv.record_passthrough(plan.node_id, "project", row, "project")
                yield row
        elif plan.kind == "join":
            # The join reports matches/pads/semi events itself via its sink;
            # its children's proofs are produced when their streams open.
            yield from plan.join
        elif plan.kind == "aggregate":
            yield from self._open_pv_aggregate(plan)
        else:  # pragma: no cover
            raise SchemaError(f"cannot open node {plan.kind}")

    def _open_pv_aggregate(self, plan: PlanNode) -> Iterator[Row]:
        pv = self.provenance
        agg = plan.aggregate
        node = plan.tree_node
        sum_ref = node.get("sum")
        key_refs = list(node.get("keys", []))
        pv.prepare_aggregate(plan.node_id, sum_ref, key_refs)
        sum_idx = agg._sum_idx  # noqa: SLF001 - same-package instrumentation
        agg.emit_empty_group = True
        agg.observer = pv.make_agg_observer(plan.node_id, sum_idx)
        rows = agg.run(self._open_pv(plan.child), self.control)
        rows = pv.annotate_aggregate_rows(
            plan.node_id, rows, len(key_refs), sum_ref is not None)
        for row in rows:
            yield row

    def run(self, plan: PlanNode) -> Iterator[Row]:
        if self.provenance is None:
            return self._open(plan)
        return self._run_provenance(plan)

    def _run_provenance(self, plan: PlanNode) -> Iterator[Row]:
        """Streaming provenance run.

        Rows are yielded as produced.  The proof commit happens only when
        the consumer drains the whole stream; closing/cancelling the
        generator aborts the run and removes all proof artifacts.
        """
        pv = self.provenance
        try:
            for row in self._open_pv(plan):
                pv.record_output(row, plan.node_id)
                yield row
        except GeneratorExit:
            pv.abort()
            raise
        except BaseException:
            pv.abort()
            raise
        else:
            pv.commit()

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
        stats = {
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
        if self.provenance is not None:
            stats["provenance_run_id"] = self.provenance.run_id
        return stats


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


def _assign_node_ids(tree: dict) -> Dict[int, str]:
    """Pre-order stable node ids; explicit join ids are preserved."""
    out: Dict[int, str] = {}
    counter = {"n": 0}

    def walk(node: dict) -> None:
        if node["node"] == "join":
            out[id(node)] = node.get("id") or f"n{counter['n']}"
        else:
            out[id(node)] = f"n{counter['n']}"
        counter["n"] += 1
        if node["node"] == "join":
            walk(node["left"])
            walk(node["right"])
        elif node["node"] != "scan":
            walk(node["input"])

    walk(tree)
    return out


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
