"""Independent naive oracle for the pre-delivery snapshot cleanup.

This oracle shares *no* machinery with :mod:`join_exec.snapshot_cleanup`: it
keeps every relation as an in-memory ``row_id -> row dict`` map and recomputes
the retained set from the raw input rows in plain Python.

The algorithm is deliberately the simplest correct one: start with every row,
then repeatedly remove any retained source row whose required reference has
no retained target match, until a full pass changes nothing.  The result is
the greatest fixpoint of the required-reference contract.  Tests compare the
exact retained ``row_id`` set per relation against the SQLite implementation,
not just row counts.

Equality follows the executor's join semantics: any NULL key component means
the key can never match (NULL does not match NULL), and int/text values must
compare with identical types.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from join_exec.snapshot_cleanup import ReferenceRule


def load_jsonl(path: str) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows[obj["row_id"]] = obj
    return rows


def _strict_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    return type(a) is type(b) and a == b


def _key_of(row: Mapping[str, Any], columns: Sequence[str]
            ) -> Tuple[Any, ...]:
    return tuple(row[c] for c in columns)


def _key_matches(src_key: Tuple[Any, ...],
                 tgt_key: Tuple[Any, ...]) -> bool:
    if any(component is None for component in src_key):
        return False
    if any(component is None for component in tgt_key):
        return False
    return len(src_key) == len(tgt_key) and all(
        _strict_equal(a, b) for a, b in zip(src_key, tgt_key))


def oracle_clean(input_paths: Mapping[str, str],
                 rules: Sequence[ReferenceRule],
                 quarantine: Sequence[Tuple[str, int]]
                 ) -> Dict[str, Set[int]]:
    """Return the maximal retained ``row_id`` set per relation."""
    relations: Dict[str, Dict[int, dict]] = {
        rel: load_jsonl(path) for rel, path in input_paths.items()}

    alive: Dict[str, Set[int]] = {
        rel: set(rows) for rel, rows in relations.items()}

    for rel, rid in quarantine:
        if rel not in relations or rid not in relations[rel]:
            raise KeyError(f"unknown quarantine row {rel}#{rid}")
        alive[rel].discard(rid)

    # Index target rows by rule and key, rebuilt each round from the current
    # alive sets.  This is intentionally O(rounds * rows): the point of this
    # oracle is obvious correctness, not performance.
    changed = True
    while changed:
        changed = False
        target_index: Dict[str, Dict[Tuple[Any, ...], List[int]]] = {}
        for rule in rules:
            index: Dict[Tuple[Any, ...], List[int]] = {}
            tgt_cols = [pair[1] for pair in rule.columns]
            for rid in alive[rule.target_relation]:
                row = relations[rule.target_relation][rid]
                key = _key_of(row, tgt_cols)
                index.setdefault(key, []).append(rid)
            target_index[rule.rule_id] = index

        for rule in rules:
            src_cols = [pair[0] for pair in rule.columns]
            supported: Set[int] = set()
            for rid in alive[rule.source_relation]:
                row = relations[rule.source_relation][rid]
                src_key = _key_of(row, src_cols)
                if any(_key_matches(src_key, tgt_key)
                       for tgt_key in target_index[rule.rule_id]):
                    supported.add(rid)
            before = len(alive[rule.source_relation])
            alive[rule.source_relation] &= supported
            if len(alive[rule.source_relation]) != before:
                changed = True

    return {rel: set(rids) for rel, rids in alive.items()}
