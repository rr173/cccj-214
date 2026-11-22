"""Independent row-by-row oracle for the maintained summary shape.

This oracle shares *no* code with :mod:`join_exec.materialized`: it keeps
relations as plain ``row_id -> dict`` maps and enumerates, for every fact row,
the exact relational result of

    SEMI( LEFT(F, B ON f.fk = b.bk), C ON f.ek = c.ck )

before grouping into ``COUNT(*)`` / ``SUM(f.amt)``.  Both the demo and the
property tests use it as ground truth for the incremental implementation.

NULL rules follow the one-shot executor's reference interpreter: NULL join
keys never match (including NULL = NULL), a LEFT join emits one NULL-padded
row when nothing matched, and the SEMI join retains a fact at most once no
matter how many eligibility rows match.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _strict_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    return type(a) is type(b) and a == b


def oracle_summary(
    relations: Mapping[str, Dict[int, dict]],
    spec,
) -> List[Dict[str, Any]]:
    """Compute the full summary by enumerating every fact row from scratch."""
    facts = relations[spec.fact_rel]
    cats = relations[spec.cat_rel]
    eligs = relations[spec.elig_rel]

    groups: Dict[Any, List[Any]] = {}

    def add(group: Any, amount: Any) -> None:
        state = groups.setdefault(group, [0, 0, 0])
        state[0] += 1  # COUNT(*) counts every joined row, NULL amount included
        if amount is not None:
            state[1] += 1
            state[2] += amount

    for fact in facts.values():
        # SEMI first: existence of at least one eligibility row on the key.
        ek = fact[spec.fact_ek]
        if ek is None:
            continue
        if not any(_strict_equal(ek, c[spec.elig_key])
                   for c in eligs.values()):
            continue

        fk = fact[spec.fact_fk]
        amount = fact[spec.sum_col]
        if fk is None:
            add(None, amount)  # NULL LEFT key never matches -> single pad
            continue
        matched = [b for b in cats.values()
                   if _strict_equal(fk, b[spec.cat_key])]
        if not matched:
            add(None, amount)  # unmatched LEFT fact -> one padded row
            continue
        # Every matching category row contributes once, including rows whose
        # group value repeats; duplicate rows are distinct dict instances.
        for b in matched:
            add(b[spec.group_col], amount)

    out: List[Dict[str, Any]] = []
    for group, (cnt, nonnull, total) in groups.items():
        out.append({"group": group, "count": cnt,
                    "sum": total if nonnull else None})
    out.sort(key=lambda d: _sort_key(d["group"]))
    return out


def _sort_key(value: Any):
    if value is None:
        return (0, 0)
    if isinstance(value, bool):  # pragma: no cover - int64/text only
        return (1, int(value))
    if isinstance(value, int):
        return (1, value)
    return (2, value)


def apply_corrections_to_maps(
    relations: Dict[str, Dict[int, dict]],
    corrections: Sequence[Mapping[str, Any]],
) -> None:
    """Mutate plain row maps with the same correction contract as the view."""
    seen: set = set()
    for c in corrections:
        rel = c["relation"]
        rid = c["row_id"]
        if (rel, rid) in seen:
            raise ValueError(f"duplicate correction for {rel}#{rid}")
        seen.add((rel, rid))
        if rel not in relations:
            raise ValueError(f"unknown relation {rel!r}")
        if c["op"] == "delete":
            if rid not in relations[rel]:
                raise ValueError(f"cannot delete missing row {rel}#{rid}")
            del relations[rel][rid]
        elif c["op"] == "upsert":
            row = c["row"]
            if row["row_id"] != rid:
                raise ValueError("row_id mismatch between correction and row")
            relations[rel][rid] = dict(row)
        else:  # pragma: no cover - callers validate first
            raise ValueError(f"unknown op {c['op']!r}")


def load_relations(paths: Mapping[str, str]
                   ) -> Dict[str, Dict[int, dict]]:
    """Load JSONL files into row maps, exactly like the reference oracle does."""
    import json

    out: Dict[str, Dict[int, dict]] = {name: {} for name in paths}
    for name, path in paths.items():
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                out[name][obj["row_id"]] = obj
    return out
