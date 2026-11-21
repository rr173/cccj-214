"""Spill space management and external-sort primitives.

A :class:`SpillManager` owns a temporary directory, counts every byte written
and guarantees file cleanup on close (including error/cancel paths).  Spill
records are one JSON object per line.  ``rids`` is a flat lineage array
``[alias, rid, alias, rid, ...]``::

    {"kind": "row",      "values": [...], "rids": ["a", 1, "b", 10]}
    {"kind": "residual", "values": [...], "rids": [...],
     "suppress": [["b", 10], ...]}

External sort is built on the standard library only (``heapq.merge``); chunk
sorting is plain in-memory ``sorted`` over budget-bounded chunks.
"""

from __future__ import annotations

import heapq
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

from .budget import Budget
from .errors import SpillError
from .types import (
    Row,
    SPILL_HANDLE_BYTES,
    decode_spill_record,
    encode_spill_record,
)


def row_to_record(row: Row) -> dict:
    return {"kind": "row", "values": list(row.values),
            "rids": list(row.rids)}


def residual_to_record(row: Row, suppress: Sequence[Tuple[str, int]]) -> dict:
    return {
        "kind": "residual",
        "values": list(row.values),
        "rids": list(row.rids),
        "suppress": [list(s) for s in suppress],
    }


def record_to_row(rec: dict) -> Tuple[Row, Optional[List[Tuple[str, int]]]]:
    row = Row(rec["values"], rec["rids"])
    if rec.get("kind") == "residual":
        return row, [tuple(x) for x in rec.get("suppress", [])]
    return row, None


@dataclass
class SpillStats:
    files: int = 0
    bytes_written: int = 0
    bytes_read: int = 0
    records_written: int = 0
    temp_dir: Optional[str] = None


class SpillFile:
    """One append-then-read JSONL spill file with byte accounting."""

    def __init__(self, path: str, manager: "SpillManager", tag: str = "") -> None:
        self.path = path
        self.manager = manager
        self.tag = tag
        self._fh = None
        self.closed = False
        self._charged = False
        self._read_charged = False

    def open_for_write(self) -> "SpillFile":
        try:
            self._fh = open(self.path, "wb", buffering=1 << 16)
        except OSError as exc:
            raise SpillError(f"cannot open spill file {self.path}: {exc}") from exc
        # Handles are OS resources; account them best-effort so an extremely
        # small logical budget cannot block spilling itself (data rows remain
        # hard-charged everywhere else).
        try:
            self.manager.budget.reserve(SPILL_HANDLE_BYTES,
                                        owner=f"spill:{self.tag}")
            self._charged = True
        except Exception:
            self._charged = False
        return self

    def write_record(self, record: dict) -> None:
        payload = encode_spill_record(record)
        try:
            self._fh.write(payload)
        except OSError as exc:
            raise SpillError(f"spill write failed: {exc}") from exc
        self.manager.stats.bytes_written += len(payload)
        self.manager.stats.records_written += 1

    def write_row(self, row: Row) -> None:
        self.write_record(row_to_record(row))

    def finalize_write(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except OSError as exc:
                raise SpillError(f"spill flush failed: {exc}") from exc
            finally:
                self._fh.close()
                self._fh = None
                if self._charged:
                    self.manager.budget.release(SPILL_HANDLE_BYTES,
                                                owner=f"spill:{self.tag}")
                    self._charged = False

    def read(self) -> Iterator[Tuple[Row, Optional[List[Tuple[str, int]]]]]:
        if self._fh is not None:
            self.finalize_write()
        try:
            fh = open(self.path, "rb", buffering=1 << 16)
        except OSError as exc:
            raise SpillError(f"cannot reopen spill {self.path}: {exc}") from exc
        try:
            self.manager.budget.reserve(SPILL_HANDLE_BYTES,
                                        owner=f"spill:{self.tag}")
            self._read_charged = True
        except Exception:
            self._read_charged = False
        try:
            for line in fh:
                if not line.strip():
                    continue
                self.manager.stats.bytes_read += len(line)
                rec = decode_spill_record(line)
                yield record_to_row(rec)
        finally:
            fh.close()
            if self._read_charged:
                self.manager.budget.release(SPILL_HANDLE_BYTES,
                                            owner=f"spill:{self.tag}")
                self._read_charged = False

    def cleanup(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
            if self._charged:
                self.manager.budget.release(SPILL_HANDLE_BYTES,
                                            owner=f"spill:{self.tag}")
                self._charged = False
        if not self.closed and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        self.closed = True


class SpillManager:
    def __init__(self, budget: Budget, base_dir: Optional[str] = None,
                 prefix: str = "joinexec-") -> None:
        self.budget = budget
        self.prefix = prefix
        if base_dir is not None:
            try:
                os.makedirs(base_dir, exist_ok=True)
            except OSError as exc:
                raise SpillError(
                    f"cannot create spill base dir {base_dir}: {exc}") from exc
        try:
            self.temp_dir = tempfile.mkdtemp(prefix=prefix, dir=base_dir)
        except OSError as exc:
            raise SpillError(f"cannot create temp dir: {exc}") from exc
        self.stats = SpillStats(temp_dir=self.temp_dir)
        self._files: List[SpillFile] = []

    def new_file(self, tag: str = "part") -> SpillFile:
        fd, path = tempfile.mkstemp(prefix=f"{tag}-", suffix=".jsonl", dir=self.temp_dir)
        os.close(fd)
        sf = SpillFile(path, self, tag=tag)
        self._files.append(sf)
        self.stats.files += 1
        return sf.open_for_write()

    def external_sort(self, rows: Iterable[Row], key_fn: Callable[[Row], Any],
                      owner: str, mem_bytes: int) -> SpillFile:
        """Sort ``rows`` by ``key_fn`` into a spill file using bounded chunks."""
        out = self.new_file(tag="sorted")
        chunk: List[Row] = []
        chunk_bytes = 0
        chunk_paths: List[str] = []

        def flush_chunk() -> None:
            nonlocal chunk, chunk_bytes
            if not chunk:
                return
            chunk.sort(key=key_fn)
            cf = self.new_file(tag="sortchunk")
            for r in chunk:
                cf.write_row(r)
            cf.finalize_write()
            chunk_paths.append(cf.path)
            chunk = []
            chunk_bytes = 0

        try:
            for row in rows:
                chunk.append(row)
                chunk_bytes += row.size
                if chunk_bytes >= max(1, mem_bytes // 2):
                    flush_chunk()
            flush_chunk()
        except SpillError:
            raise
        if len(chunk_paths) <= 1:
            if chunk_paths:
                # already fully sorted in one chunk file
                out.finalize_write()
                single = SpillFile(chunk_paths[0], self, tag="sorted")
                self._files.append(single)
                out.cleanup()
                return single
            out.finalize_write()
            return out

        # k-way merge into out
        handles = []
        iterators = []
        try:
            for p in chunk_paths:
                fh = open(p, "rb", buffering=1 << 16)
                handles.append(fh)
                iterators.append(_file_row_iter(fh, self))
            for row in heapq.merge(*iterators, key=key_fn):
                out.write_row(row)
        except OSError as exc:
            raise SpillError(f"external sort merge failed: {exc}") from exc
        finally:
            for fh in handles:
                fh.close()
        out.finalize_write()
        for p in chunk_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        return out

    def cleanup(self) -> None:
        for sf in self._files:
            sf.cleanup()
        try:
            if os.path.isdir(self.temp_dir):
                os.rmdir(self.temp_dir)
        except OSError:
            pass


def _file_row_iter(fh, manager: SpillManager) -> Iterator[Row]:
    for line in fh:
        if not line.strip():
            continue
        manager.stats.bytes_read += len(line)
        rec = decode_spill_record(line)
        row, _ = record_to_row(rec)
        yield row
