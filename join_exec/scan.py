"""Scan operator and JSONL parsing with schema validation."""

from __future__ import annotations

import json
from typing import Iterator, List, Optional, Tuple

from .control import ControlPoint
from .errors import BadRowError, RunCancelled
from .types import Header, Row, Schema, check_int64


class Scan:
    """Streaming JSONL scan.

    Input lines must be JSON objects containing ``row_id`` plus every declared
    column.  ``row_id`` values are unique within the relation (guaranteed by
    the input contract and checked defensively).
    """

    def __init__(self, schema: Schema, path: str, alias: str,
                 control: Optional[ControlPoint] = None) -> None:
        self.schema = schema
        self.path = path
        self.alias = alias
        self.control = control
        self.header = Header(tuple((alias, c.name) for c in schema.columns))
        self.rows_scanned = 0

    def __iter__(self) -> Iterator[Row]:
        # A scan must be safely re-openable: disk joins re-run whole subtrees
        # and re-read spill sources, so duplicate-id detection is local to a
        # single pass rather than living on the operator instance.
        seen_rids = set()
        try:
            fh = open(self.path, "r", encoding="utf-8")
        except OSError as exc:
            raise BadRowError(f"cannot open input {self.path}: {exc}") from exc
        with fh:
            for lineno, line in enumerate(fh, start=1):
                if self.control is not None and self.control.is_cancelled():
                    raise RunCancelled(self.control.reason or "scan cancelled")
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BadRowError(
                        f"{self.path}:{lineno}: invalid JSON: {exc}"
                    ) from exc
                yield self._decode(obj, lineno, seen_rids)

    def _decode(self, obj: dict, lineno: int, seen_rids: set) -> Row:
        if not isinstance(obj, dict) or "row_id" not in obj:
            raise BadRowError(f"{self.path}:{lineno}: missing row_id")
        rid = check_int64(obj["row_id"])
        if rid in seen_rids:
            raise BadRowError(
                f"{self.path}:{lineno}: duplicate row_id {rid} in {self.schema.relation}"
            )
        seen_rids.add(rid)
        values: List = []
        for col in self.schema.columns:
            if col.name not in obj:
                raise BadRowError(
                    f"{self.path}:{lineno}: missing column {col.name!r}"
                )
            v = obj[col.name]
            if v is None:
                values.append(None)
            elif col.dtype == "int64":
                values.append(check_int64(v))
            elif col.dtype == "text":
                if not isinstance(v, str):
                    raise BadRowError(
                        f"{self.path}:{lineno}: column {col.name!r} expects text"
                    )
                values.append(v)
        self.rows_scanned += 1
        return Row(values, (self.alias, rid))
