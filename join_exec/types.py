"""Core row/header data structures, memory accounting and key helpers.

Memory strategy
---------------
The shared :class:`~join_exec.budget.MemoryBudget` accounts *logical* bytes of
data held by operators (hash-table entries, buffered rows, aggregate state).
Spill files live on disk: only a small fixed read/write buffer is charged.
The benchmark separately records real RSS so the logical accounting can be
cross-checked against the operating system.

Integer strategy
----------------
Every int field is validated against int64 on read.  Sum and count perform
checked int64 arithmetic and raise :class:`OverflowError` on overflow rather
than silently producing a value outside int64.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .errors import BadRowError, SchemaError

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
NULL = None
# Sentinel meaning "this join key contains a NULL component; never matches".
NULL_KEY = "\x00null-key-sentinel"
VALID_TYPES = ("int64", "text")


def check_int64(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRowError(f"expected int64, got {type(value).__name__}: {value!r}")
    if value < INT64_MIN or value > INT64_MAX:
        raise BadRowError(f"int64 value out of range: {value}")
    return value


def checked_add_int64(a: int, b: int) -> int:
    r = a + b
    if r < INT64_MIN or r > INT64_MAX:
        raise OverflowError(f"int64 overflow: {a} + {b}")
    return r


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # "int64" | "text"

    def __post_init__(self) -> None:
        if self.dtype not in VALID_TYPES:
            raise SchemaError(f"unsupported column type {self.dtype!r}")


@dataclass(frozen=True)
class Schema:
    relation: str
    columns: Tuple[Column, ...]

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise SchemaError(f"relation {self.relation!r} has no column {name!r}")


def schema_from_json(rel: str, spec: Sequence[dict]) -> Schema:
    return Schema(
        relation=rel,
        columns=tuple(Column(c["name"], c["type"]) for c in spec),
    )


@dataclass(frozen=True)
class Header:
    """Ordered list of (relation_alias, column_name) slots in a row."""

    slots: Tuple[Tuple[str, str], ...]

    def index(self, ref: Tuple[str, str]) -> int:
        for i, slot in enumerate(self.slots):
            if slot == ref:
                raise KeyError
        # unreachable helper kept explicit
        raise KeyError(ref)

    def find(self, alias: str, col: str) -> int:
        for i, (a, c) in enumerate(self.slots):
            if a == alias and c == col:
                return i
        raise SchemaError(f"field {alias}.{col} is not in scope; have {self.slots}")

    def find_ref(self, ref: str) -> int:
        try:
            alias, col = ref.split(".", 1)
        except ValueError as exc:
            raise SchemaError(f"field reference must be alias.column, got {ref!r}") from exc
        return self.find(alias, col)

    def union(self, other: "Header") -> "Header":
        return Header(self.slots + other.slots)


class Row:
    """One materialised tuple.

    ``rids`` is the flat lineage of contributing base rows as
    ``(alias, rid, alias, rid, ...)``.  Rows produced by joins carry the rids
    of every contributing base row so tests can assert exact source-lineage
    pairs and the disk-phase matched-state machinery can identify rows.

    Values are kept in a tuple: rows are only ever rebuilt (never mutated),
    and tuples avoid the per-row list overhead, which matters at 1M rows.

    Side-channel metadata (never part of business values, filters, keys or
    aggregates):

    * ``is_pad`` marks a LEFT-join padding row (the right-side slots are
      fabricated NULLs).  It lets the provenance layer distinguish a pad
      from a real match whose right values happen to all be NULL.
    * ``proof_ref`` carries the stable, content-addressed provenance
      reference for this output row.  It is attached by the provenance
      session only; plain runs leave it ``None``.
    """

    __slots__ = ("values", "rids", "size", "is_pad", "proof_ref")

    def __init__(self, values: Sequence[Any], rids: Sequence[Any],
                 size: Optional[int] = None,
                 is_pad: bool = False,
                 proof_ref: Optional[str] = None) -> None:
        self.values: Tuple[Any, ...] = tuple(values)
        self.rids: Tuple[Any, ...] = tuple(rids)
        self.size: int = size if size is not None else estimate_row_size(self.values)
        self.is_pad: bool = is_pad
        self.proof_ref: Optional[str] = proof_ref

    def rid_of(self, alias: str) -> int:
        rids = self.rids
        for i in range(0, len(rids), 2):
            if rids[i] == alias:
                return rids[i + 1]
        raise SchemaError(f"row has no rid for alias {alias!r}")

    def sources(self) -> Tuple[Tuple[str, int], ...]:
        """The contributing base rows as a canonical, sorted tuple."""
        return tuple(sorted(
            (self.rids[i], self.rids[i + 1])
            for i in range(0, len(self.rids), 2)))

    def copy(self) -> "Row":
        return Row(self.values, self.rids, self.size,
                   is_pad=self.is_pad, proof_ref=self.proof_ref)

    def to_jsonable(self) -> dict:
        paired = [list(self.rids[i:i + 2])
                  for i in range(0, len(self.rids), 2)]
        return {"values": list(self.values), "rids": paired}


def estimate_row_size(values: Sequence[Any]) -> int:
    """Conservative logical size of a row's payload in bytes."""
    total = 48  # Row object + list baseline
    for v in values:
        if v is None:
            total += 8
        elif isinstance(v, bool):  # pragma: no cover - bools are not a column type
            total += 28
        elif isinstance(v, int):
            total += 32
        elif isinstance(v, str):
            total += 58 + 2 * len(v)
        else:  # pragma: no cover - rejected at parse time
            total += 64
    total += 16 + 16 * max(1, len(values))  # rid tuple contribution
    return total


#: Extra bytes charged per hash-table entry (dict slot + key object overhead).
HT_ENTRY_OVERHEAD = 72
#: Rows per input batch for batch-boundary control points.
BATCH_SIZE = 64
#: Fixed bytes charged for an open spill file handle / buffer.
SPILL_HANDLE_BYTES = 2 * 1024


def make_join_key(row: Row, header: Header, refs: Sequence[Tuple[str, str]]) -> Any:
    """Return a hashable join key, or NULL_KEY if any component is NULL."""
    parts: List[Any] = []
    for alias, col in refs:
        v = row.values[header.find(alias, col)]
        if v is None:
            return NULL_KEY
        parts.append(v)
    return tuple(parts)


def tag_key(key: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Make heterogeneous keys sortable: ints and texts must not compare raw."""
    out: List[Any] = []
    for v in key:
        if isinstance(v, bool):  # pragma: no cover
            out.append((1, int(v)))
        elif isinstance(v, int):
            out.append((0, v))
        elif isinstance(v, str):
            out.append((1, v))
        else:  # pragma: no cover
            out.append((2, str(v)))
    return tuple(out)


def encode_spill_record(record: dict) -> bytes:
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode_spill_record(line: bytes) -> dict:
    try:
        return json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BadRowError(f"corrupt spill record: {exc}") from exc
