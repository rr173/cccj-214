"""Independent proof oracle.

This module is the test-side authority for what provenance *should* contain.
It deliberately shares no code with the product provenance machinery: it does
not import :mod:`join_exec.provenance`, :mod:`join_exec.proof_store`, the
spill/partition code or any switch state.  Inputs are read as plain JSONL
dictionaries and expected witnesses are enumerated row by row, independently
of how the executor chose to run (memory or disk).

For every query-tree node it enumerates, independently:

* the business output tuples (values + ``(alias, row_id)`` sources);
* the expected proof kind and the exact set of contributing source
  combinations;
* for aggregates, the exact count inputs and the non-NULL sum inputs.

Tests then compare the durable proof documents (read back through the public
:class:`~join_exec.proof_store.ProofReader`) against these expectations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Source = Tuple[str, int]
# An evaluated tuple: (sources, values keyed by alias.col).
TupleT = Tuple[Tuple[Source, ...], Dict[str, Any]]


def _canonical(obj: Any) -> str:
    """Independent canonical JSON (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _content_id(prefix: str, parts: List[Any]) -> str:
    digest = hashlib.sha256(_canonical(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:40]}"


def row_proof_ref(node_id: str, kind: str,
                  sources: Tuple[Source, ...]) -> str:
    # Mirrors the product's content addressing independently.
    return _content_id("rp", ["row", node_id, kind, sorted(sources)])


def aggregate_proof_ref(node_id: str, metric: str,
                        group_key: Tuple[Any, ...]) -> str:
    return _content_id("ap", ["agg", node_id, metric, list(group_key)])


def canonical_json(obj: Any) -> str:
    return _canonical(obj)


def load_relation(path: str, alias: str) -> List[TupleT]:
    out: List[TupleT] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = obj["row_id"]
            values = {f"{alias}.{k}": v
                      for k, v in obj.items() if k != "row_id"}
            out.append((((alias, rid),), values))
    return out


def _cmp(op: str, a: Any, b: Any) -> Optional[bool]:
    if a is None or b is None:
        return None
    if type(a) is not type(b):
        return None
    return {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b,
            "gt": a > b, "ge": a >= b}[op]


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


def _not(v: Optional[bool]) -> Optional[bool]:
    return None if v is None else not v


def _eval_expr(node: dict, values: Dict[str, Any]) -> Optional[bool]:
    op = node["op"]
    if op == "and":
        acc = _eval_expr(node["args"][0], values)
        for a in node["args"][1:]:
            acc = _and(acc, _eval_expr(a, values))
        return acc
    if op == "or":
        acc = _eval_expr(node["args"][0], values)
        for a in node["args"][1:]:
            acc = _or(acc, _eval_expr(a, values))
        return acc
    if op == "not":
        return _not(_eval_expr(node["arg"], values))

    def operand(x: Any) -> Any:
        if isinstance(x, str) and "." in x:
            return values.get(x)
        if isinstance(x, dict) and "literal" in x:
            return x["literal"]
        return x

    return _cmp(op, operand(node["left"]), operand(node["right"]))


def stable_node_ids(tree: dict) -> Dict[int, str]:
    """Same pre-order id rule the engine uses, computed independently."""
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


@dataclass
class ExpectedProof:
    proof_ref: str
    kind: str
    node_id: str
    sources: Tuple[Source, ...]
    # for semi: the full (left + each right) match source tuples
    match_sources: Tuple[Tuple[Source, ...], ...] = ()
    # aggregate
    metric: Optional[str] = None
    group_key: Tuple[Any, ...] = ()
    contributors: Tuple[Tuple[Source, ...], ...] = ()
    sum_values: Tuple[Any, ...] = ()
    null_contributors: int = 0
    value: Any = None


class ProofOracle:
    """Enumerates expected proofs for a whole tree from raw JSONL inputs."""

    def __init__(self, tree: dict) -> None:
        self.tree = tree
        self.ids = stable_node_ids(tree)
        #: node_id -> list[ExpectedProof] in business emission order
        self.expected: Dict[str, List[ExpectedProof]] = {}
        #: proof_ref -> ExpectedProof
        self.by_ref: Dict[str, ExpectedProof] = {}
        # top-level output node id and ordered output refs
        self.root_id = self.ids[id(tree)]
        self.root_refs: List[str] = []
        self._evaluate(tree)

    # ------------------------------------------------------------- eval
    def _evaluate(self, node: dict) -> List[TupleT]:
        nid = self.ids[id(node)]
        kind = node["node"]
        if kind == "scan":
            alias = node.get("alias", node["relation"])
            tuples_ = load_relation(node["file"], alias)
            for sources, values in tuples_:
                self._add_row(nid, "scan", "scan", sources)
            return tuples_
        if kind == "filter":
            child = self._evaluate(node["input"])
            kept = [(s, v) for s, v in child
                    if _eval_expr(node["expr"], v) is True]
            for sources, _ in kept:
                self._add_row(nid, "filter", "filter", sources)
            return kept
        if kind == "project":
            child = self._evaluate(node["input"])
            out: List[TupleT] = []
            for sources, values in child:
                projected = {f"{a}.{r.split('.', 1)[1]}": values[r]
                             for a, r in node["outputs"]}
                out.append((sources, projected))
                self._add_row(nid, "project", "project", sources)
            return out
        if kind == "join":
            return self._eval_join(node, nid)
        if kind == "aggregate":
            return self._eval_aggregate(node, nid)
        raise ValueError(f"unknown node {kind}")

    def _add_row(self, node_id: str, node_kind: str, kind: str,
                 sources: Tuple[Source, ...]) -> str:
        ref = row_proof_ref(node_id, kind, sources)
        exp = ExpectedProof(proof_ref=ref, kind=kind, node_id=node_id,
                            sources=tuple(sorted(sources)))
        self.expected.setdefault(node_id, []).append(exp)
        self.by_ref[ref] = exp
        return ref

    def _key_equal(self, lt: TupleT, rt: TupleT, lref: str,
                   rref: str) -> bool:
        lv = lt[1].get(lref)
        rv = rt[1].get(rref)
        if lv is None or rv is None:
            return False
        return lv == rv and type(lv) is type(rv)

    def _eval_join(self, node: dict, nid: str) -> List[TupleT]:
        lefts = self._evaluate(node["left"])
        rights = self._evaluate(node["right"])
        jtype = node["type"]
        on = node["on"]
        out: List[TupleT] = []
        for ls, lv in lefts:
            matches = [rt for rt in rights
                       if all(self._key_equal((ls, lv), rt, l, r)
                              for l, r in on)]
            if jtype == "semi":
                if matches:
                    ref = self._add_row(nid, "join", "semi", ls)
                    exp = self.by_ref[ref]
                    object.__setattr__(
                        exp, "match_sources",
                        tuple(sorted(tuple(sorted(ls + rs))
                                     for rs, _ in matches)))
                    out.append((ls, lv))
                continue
            for rs, rv in matches:
                merged_values = dict(lv)
                merged_values.update(rv)
                sources = tuple(sorted(ls + rs))
                if jtype == "inner" or jtype == "left":
                    self._add_row(nid, "join", "match", sources)
                    out.append((sources, merged_values))
            if not matches and jtype == "left":
                # Pad: left sources only, no fabricated right row id.
                self._add_row(nid, "join", "pad", ls)
                padded = dict(lv)
                for rs, rv in matches:  # empty by construction
                    pass
                out.append((ls, padded))
        return out

    def _eval_aggregate(self, node: dict, nid: str) -> List[TupleT]:
        rows = self._evaluate(node["input"])
        keys = node.get("keys", [])
        sum_ref = node.get("sum")
        groups: Dict[Tuple[Any, ...], List[TupleT]] = {}
        order: List[Tuple[Any, ...]] = []
        for t in rows:
            key = tuple(t[1].get(k) for k in keys)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(t)
        if not order and not keys:
            # SQL scalar aggregation over empty input: one 0/NULL group.
            order = [()]
            groups[()] = []
        out: List[TupleT] = []
        for key in order:
            members = groups[key]
            count_value = len(members)
            non_null = [(s, v) for s, v in members
                        if sum_ref is None or v.get(sum_ref) is not None]
            sum_value = (sum(v[sum_ref] for _, v in non_null)
                         if sum_ref is not None and non_null else None)
            null_count = (len(members) - len(non_null)
                          if sum_ref is not None else 0)
            count_ref = aggregate_proof_ref(nid, "count", key)
            self.by_ref[count_ref] = ExpectedProof(
                proof_ref=count_ref, kind="count", node_id=nid,
                sources=(), metric="count", group_key=key,
                contributors=tuple(sorted(s for s, _ in members)),
                value=count_value)
            self.expected.setdefault(nid, []).append(
                self.by_ref[count_ref])
            if sum_ref is not None:
                sum_id = aggregate_proof_ref(nid, "sum", key)
                self.by_ref[sum_id] = ExpectedProof(
                    proof_ref=sum_id, kind="sum", node_id=nid,
                    sources=(), metric="sum", group_key=key,
                    contributors=tuple(sorted(s for s, _ in non_null)),
                    sum_values=tuple(v[sum_ref] for _, v in non_null),
                    null_contributors=null_count, value=sum_value)
                self.expected.setdefault(nid, []).append(
                    self.by_ref[sum_id])
            values: Dict[str, Any] = {}
            for i, k in enumerate(keys):
                values[f"agg.key_{i}"] = key[i]
            values["agg.count"] = count_value
            if sum_ref is not None:
                values[f"agg.sum_{sum_ref.replace('.', '_')}"] = sum_value
            out.append(((), values))
            self.root_refs.append(count_ref)
        return out

    # ------------------------------------------------------------- queries
    def expected_refs_for_node(self, node_id: str) -> List[str]:
        return [e.proof_ref for e in self.expected.get(node_id, [])]

    def all_refs(self) -> List[str]:
        return list(self.by_ref.keys())


def expected_run_id(tree: dict) -> str:
    digest = hashlib.sha256(
        canonical_json(tree).encode("utf-8")).hexdigest()[:16]
    return f"run-{digest}"
