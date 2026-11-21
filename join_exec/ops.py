"""Three-valued filter, projection and terminal aggregation.

Three-valued logic follows SQL: comparisons with NULL yield UNKNOWN, AND/OR
combine TRUE/FALSE/UNKNOWN truth-table style, and only rows evaluating to
TRUE pass the filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from .budget import Budget
from .control import ControlPoint
from .errors import RunCancelled, SchemaError
from .types import Header, Row, checked_add_int64

TRUE, FALSE, UNKNOWN = True, False, None

_CMP = ("eq", "ne", "lt", "le", "gt", "ge")


def _compare(op: str, left: Any, right: Any) -> Optional[bool]:
    if left is None or right is None:
        return UNKNOWN
    # Type mismatch -> UNKNOWN (SQL-style), never an implicit cast.
    if isinstance(left, bool) or isinstance(right, bool):  # pragma: no cover
        return UNKNOWN
    if type(left) is not type(right):
        return UNKNOWN
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "lt":
        return left < right
    if op == "le":
        return left <= right
    if op == "gt":
        return left > right
    if op == "ge":
        return left >= right
    raise SchemaError(f"unknown comparison {op!r}")  # pragma: no cover


def _not(v: Optional[bool]) -> Optional[bool]:
    if v is UNKNOWN:
        return UNKNOWN
    return not v


def _and(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is FALSE or b is FALSE:
        return FALSE
    if a is UNKNOWN or b is UNKNOWN:
        return UNKNOWN
    return TRUE


def _or(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is TRUE or b is TRUE:
        return TRUE
    if a is UNKNOWN or b is UNKNOWN:
        return UNKNOWN
    return FALSE


@dataclass
class Filter:
    """Expression tree node.

    Leaf: ``{"op": <cmp>, "left": "alias.col", "right": "alias.col"|value}``.
    Branch: ``{"op": "and"|"or"|"not", "args": [...]}``.
    """

    spec: dict
    header: Header

    def __post_init__(self) -> None:
        self._validate(self.spec)

    def _validate(self, node: dict) -> None:
        op = node["op"]
        if op in ("and", "or"):
            for a in node["args"]:
                self._validate(a)
        elif op == "not":
            self._validate(node["arg"])
        elif op in _CMP:
            self.header.find_ref(node["left"])
            r = node["right"]
            if isinstance(r, str) and "." in r:
                self.header.find_ref(r)
        else:
            raise SchemaError(f"unknown filter op {op!r}")

    def _eval_operand(self, operand: Any, row: Row) -> Any:
        if isinstance(operand, str) and "." in operand:
            return row.values[self.header.find_ref(operand)]
        if isinstance(operand, dict) and "literal" in operand:
            return operand["literal"]
        return operand

    def eval(self, row: Row) -> Optional[bool]:
        node = self.spec
        op = node["op"]
        if op in ("and", "or"):
            vals = [Filter(a, self.header).eval(row) for a in node["args"]]
            acc = vals[0]
            combine = _and if op == "and" else _or
            for v in vals[1:]:
                acc = combine(acc, v)
            return acc
        if op == "not":
            return _not(Filter(node["arg"], self.header).eval(row))
        lv = self._eval_operand(node["left"], row)
        rv = self._eval_operand(node["right"], row)
        return _compare(op, lv, rv)

    def apply(self, rows: Iterator[Row], control: Optional[ControlPoint] = None
              ) -> Iterator[Row]:
        passed = 0
        rejected = 0
        for row in rows:
            if control is not None and control.is_cancelled():
                raise RunCancelled(control.reason or "filter cancelled")
            if self.eval(row) is TRUE:
                passed += 1
                yield row
            else:
                rejected += 1
        self.rows_passed = passed
        self.rows_rejected = rejected


@dataclass
class Project:
    header_in: Header
    outputs: List[Tuple[str, str]]  # (alias of output, source ref "a.col")

    def __post_init__(self) -> None:
        self.header = Header(tuple((out_alias, ref.split(".", 1)[1])
                                   for out_alias, ref in self.outputs))
        self._idx = [self.header_in.find_ref(ref) for _, ref in self.outputs]

    def apply(self, rows: Iterator[Row], control: Optional[ControlPoint] = None
              ) -> Iterator[Row]:
        for row in rows:
            if control is not None and control.is_cancelled():
                raise RunCancelled(control.reason or "project cancelled")
            values = [row.values[i] for i in self._idx]
            yield Row(values, row.rids)


class Aggregate:
    """Terminal per-key count/sum.

    Output schema slots are named ``key_<n>`` plus ``count`` and ``sum_<field>``.
    ``sum`` over a group with no non-NULL input yields NULL.  Both count and
    sum are checked int64.
    """

    def __init__(self, header: Header, key_refs: Sequence[str],
                 sum_ref: Optional[str], budget: Budget,
                 group_alias: str = "agg") -> None:
        self.header_in = header
        self.key_refs = list(key_refs)
        self.sum_ref = sum_ref
        self.budget = budget
        self.group_alias = group_alias
        self._key_idx = [header.find_ref(r) for r in key_refs]
        self._sum_idx = header.find_ref(sum_ref) if sum_ref else None
        slots = tuple((group_alias, f"key_{i}") for i in range(len(self._key_idx)))
        slots = slots + ((group_alias, "count"),)
        if sum_ref:
            slots = slots + ((group_alias, f"sum_{sum_ref.replace('.', '_')}"),)
        self.header = Header(slots)
        # group key -> [count, sum, has_non_null, reserved_bytes]
        self._groups: dict = {}
        self._owner = "aggregate"

    def run(self, rows: Iterator[Row], control: Optional[ControlPoint] = None
            ) -> List[Row]:
        out: List[Row] = []
        try:
            for row in rows:
                if control is not None and control.is_cancelled():
                    raise RunCancelled(control.reason or "aggregate cancelled")
                key = tuple(row.values[i] for i in self._key_idx)
                if key not in self._groups:
                    est = 96 + 16 * len(key)
                    self.budget.reserve(est, owner=self._owner)
                    self._groups[key] = [0, 0, False, est]
                g = self._groups[key]
                g[0] = checked_add_int64(g[0], 1)
                if self._sum_idx is not None:
                    v = row.values[self._sum_idx]
                    if v is not None:
                        g[1] = checked_add_int64(g[1], v)
                        g[2] = True
            for key, (cnt, total, has_non_null, _est) in self._groups.items():
                values = list(key) + [cnt]
                if self.sum_ref:
                    values.append(total if has_non_null else None)
                out.append(Row(values, []))
        finally:
            for g in self._groups.values():
                self.budget.release(g[3], owner=self._owner)
            self._groups.clear()
        return out
