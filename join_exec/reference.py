"""Independent nested-loop reference interpreter.

This is the test oracle.  It deliberately shares *no* execution machinery
with the product: no partitions, no hash tables, no matched-state objects,
no spill code.  Rows are read as plain Python dicts straight from JSONL and
joins are evaluated by exhaustive nested loops over the original rows,
exactly as the relational semantics dictate.

Bag semantics are preserved by materialising every matching combination as a
tuple of (alias, row_id) lineage plus values, never deduplicating by business
value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .errors import BadRowError

NULL = None


# --------------------------------------------------------------------- rows
@dataclass(frozen=True)
class RefRow:
    alias: str
    rid: int
    values: Dict[str, Any]

    def get(self, ref: str) -> Any:
        alias, col = ref.split(".", 1)
        if alias != self.alias:
            raise BadRowError(f"ref {ref} not in row alias {self.alias}")
        return self.values.get(col)


def load_raw(path: str, alias: str) -> List[RefRow]:
    rows: List[RefRow] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(RefRow(alias, obj["row_id"],
                               {k: v for k, v in obj.items() if k != "row_id"}))
    return rows


# A running tuple is (lineage tuple, values dict keyed by alias.col).
Tuple = Tuple[Tuple[Tuple[str, int], ...], Dict[str, Any]]


def _merge(left: Tuple, right: RefRow) -> Tuple:
    lineage, values = left
    merged = dict(values)
    for col, v in right.values.items():
        merged[f"{right.alias}.{col}"] = v
    return lineage + ((right.alias, right.rid),), merged


def _scan_tuple(row: RefRow) -> Tuple:
    values = {f"{row.alias}.{c}": v for c, v in row.values.items()}
    return ((row.alias, row.rid),), values


# ------------------------------------------------------------- 3VL filter
def _cmp(op: str, a: Any, b: Any) -> Optional[bool]:
    if a is None or b is None:
        return None
    if type(a) is not type(b):
        return None
    return {
        "eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b,
        "gt": a > b, "ge": a >= b,
    }[op]


def _not(v: Optional[bool]) -> Optional[bool]:
    return None if v is None else not v


def _and(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _or(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


def _eval(node: dict, values: Dict[str, Any]) -> Optional[bool]:
    op = node["op"]
    if op == "and":
        acc = _eval(node["args"][0], values)
        for a in node["args"][1:]:
            acc = _and(acc, _eval(a, values))
        return acc
    if op == "or":
        acc = _eval(node["args"][0], values)
        for a in node["args"][1:]:
            acc = _or(acc, _eval(a, values))
        return acc
    if op == "not":
        return _not(_eval(node["arg"], values))

    def operand(x: Any) -> Any:
        if isinstance(x, str) and "." in x:
            return values.get(x)
        if isinstance(x, dict) and "literal" in x:
            return x["literal"]
        return x

    return _cmp(op, operand(node["left"]), operand(node["right"]))


# ----------------------------------------------------------------- planner
class ReferenceInterpreter:
    def __init__(self, tree: dict, schemas: Optional[dict] = None) -> None:
        self.tree = tree
        # Optional {relation: [column, ...]} used when a scan file is empty,
        # so LEFT padding still knows the right-side column names.
        self.schemas = schemas or {}

    def run(self) -> List[Tuple]:
        return self._eval_node(self.tree)

    def _eval_node(self, node: dict) -> List[Tuple]:
        kind = node["node"]
        if kind == "scan":
            alias = node.get("alias", node["relation"])
            return [_scan_tuple(r) for r in load_raw(node["file"], alias)]
        if kind == "filter":
            return [t for t in self._eval_node(node["input"])
                    if _eval(node["expr"], t[1]) is True]
        if kind == "project":
            out = []
            for lineage, values in self._eval_node(node["input"]):
                projected = {f"{out_alias}.{ref.split('.', 1)[1]}": values[ref]
                             for out_alias, ref in node["outputs"]}
                out.append((lineage, projected))
            return out
        if kind == "aggregate":
            return self._aggregate(node)
        if kind == "join":
            return self._join(node)
        raise BadRowError(f"reference: unknown node {kind}")

    def _join(self, node: dict) -> List[Tuple]:
        lefts = self._eval_node(node["left"])
        rights_raw = self._raw_right_rows(node["right"])
        jtype = node["type"]
        on = node["on"]
        out: List[Tuple] = []
        for lt in lefts:
            matched = False
            for rrow in rights_raw:
                if all(self._key_equal(lt, rrow, lref, rref)
                       for lref, rref in on):
                    merged = _merge(lt, rrow)
                    if jtype == "semi":
                        out.append(lt)  # at most once: break immediately
                        matched = True
                        break
                    out.append(merged)
                    matched = True
            if not matched and jtype == "left":
                out.append(self._pad(lt, node["right"]))
        return out

    def _raw_right_rows(self, node: dict) -> List[RefRow]:
        """Right subtree evaluated independently; only leaf rows needed for
        equality matching, but filters/project must still be honoured."""
        tuples = self._eval_node(node)
        rows: List[RefRow] = []
        for lineage, values in tuples:
            # rebuild a pseudo RefRow per output tuple using its rightmost rid
            rid = lineage[-1][1]
            alias = lineage[-1][0]
            stripped = {k.split(".", 1)[1]: v for k, v in values.items()
                        if k.split(".", 1)[0] == alias}
            rows.append(RefRow(alias, rid, stripped))
        return rows

    def _key_equal(self, lt: Tuple, rrow: RefRow, lref: str, rref: str) -> bool:
        lv = lt[1].get(lref)
        rv = rrow.get(rref)
        if lv is None or rv is None:
            return False  # NULL never matches, including NULL = NULL
        return lv == rv and type(lv) is type(rv)

    def _pad(self, lt: Tuple, right_node: dict) -> Tuple:
        lineage, values = lt
        padded = dict(values)
        for alias, cols in self._output_columns(right_node):
            for c in cols:
                padded[f"{alias}.{c}"] = None
        return lineage, padded

    def _output_columns(self, node: dict) -> List[Tuple[str, List[str]]]:
        kind = node["node"]
        if kind == "scan":
            alias = node.get("alias", node["relation"])
            relation = node["relation"]
            path = node["file"]
            columns: Optional[List[str]] = None
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        columns = [c for c in json.loads(line)
                                   if c != "row_id"]
                        break
            if columns is None:
                # Empty file: prefer an explicitly provided schema.
                declared = self.schemas.get(relation)
                if declared is None:
                    raise BadRowError(
                        f"reference: cannot infer columns for empty scan "
                        f"{relation!r}; pass schemas= for LEFT padding")
                columns = list(declared)
            return [(alias, columns)]
        if kind == "project":
            return [(out_alias, [ref.split(".", 1)[1]])
                    for out_alias, ref in node["outputs"]]
        if kind == "join":
            return self._output_columns(node["left"]) + \
                self._output_columns(node["right"])
        return self._output_columns(node["input"])

    def _aggregate(self, node: dict) -> List[Tuple]:
        rows = self._eval_node(node["input"])
        keys = node.get("keys", [])
        sum_ref = node.get("sum")
        groups: Dict[Tuple[Any, ...], List[Any]] = {}
        order: List[Tuple[Any, ...]] = []
        for lineage, values in rows:
            key = tuple(values.get(k) for k in keys)
            if key not in groups:
                groups[key] = [0, 0, False]
                order.append(key)
            g = groups[key]
            g[0] += 1
            if sum_ref is not None:
                v = values.get(sum_ref)
                if v is not None:
                    g[1] += v
                    g[2] = True
        out = []
        for key in order:
            cnt, total, has_non_null = groups[key]
            values: Dict[str, Any] = {}
            for i, k in enumerate(keys):
                values[f"agg.key_{i}"] = key[i]
            values["agg.count"] = cnt
            if sum_ref is not None:
                values[f"agg.sum_{sum_ref.replace('.', '_')}"] = (
                    total if has_non_null else None)
            out.append(((), values))
        return out
