"""Durable, content-addressed result-provenance store.

The store is a *separate deliverable* from business results: writing a proof
never changes row values, filters or aggregates.  It is designed around three
hard constraints:

1. **No whole-result materialisation.**  Proof events are streamed into fixed
   JSONL staging shards; nothing is accumulated per proof in memory (a single
   count/sum proof may have millions of contributions).
2. **Bounded memory.**  Commit groups events with an external sort (bounded
   in-memory chunks + ``heapq.merge``), and distinct-source counting spills
   through the same mechanism once a capped set overflows the shared budget.
3. **Atomic, recognisable completeness.**  Staging data is rewritten into
   final sorted data/index files; the manifest (whose presence means
   "complete") is published last with an atomic rename.  Any error or
   cancellation removes the whole run directory, so a partial index can
   never be mistaken for a complete proof set.

Proof references are SHA-256 content ids derived from the node id, the proof
kind and the contributing ``(alias, row_id)`` sources — never from temp
directory names, iteration order or process addresses.  The same logical
source therefore yields the same reference on memory and disk executions and
on re-runs.

On-disk layout of one run directory::

    staging/ev/shard-XX.jsonl      streamed proof events (commit input)
    staging/out/shard-XX.jsonl     streamed output-row -> proof mapping
    staging/distinct-spill.jsonl   overflow tuples for distinct counting
    staging/sort-*.jsonl           bounded external-sort chunks
    data/shard-XX.jsonl            committed proof documents, pid sorted
    index/shard-XX.jsonl           pid -> (offset, length, counts)
    index/shard-XX.dir.jsonl       sparse byte-offset directory
    mapping/shard-XX.jsonl         committed output mapping, pid sorted
    mapping/shard-XX.dir.jsonl     sparse directory for the mapping shard
    manifest.json                  present only after a successful commit
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .budget import Budget
from .errors import ProofError, RunCancelled
from .types import SPILL_HANDLE_BYTES

PROOF_FORMAT = "join-exec-proof/1"
DEFAULT_SHARDS = 16
SORT_CHUNK_TARGET_LINES = 256
INDEX_SAMPLE_EVERY = 64
#: Logical bytes charged per distinct-set entry (dict slot + tuple objects).
DISTINCT_ENTRY_COST = 72


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text used for content-addressed ids."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def content_id(prefix: str, parts: Sequence[Any]) -> str:
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:40]}"


def row_proof_ref(node_id: str, kind: str,
                  sources: Sequence[Tuple[str, int]]) -> str:
    return content_id("rp", ["row", node_id, kind, sorted(sources)])


def aggregate_proof_ref(node_id: str, metric: str,
                        group_key: Sequence[Any]) -> str:
    return content_id("ap", ["agg", node_id, metric, list(group_key)])


def event_sort_key(line: bytes) -> Tuple[str, int]:
    """Order staging events by proof id, registration before witnesses."""
    rec = json.loads(line)
    return rec["pid"], rec.get("ord", 1)


def mapping_sort_key(line: bytes) -> Tuple[str, int]:
    rec = json.loads(line)
    return rec.get("row_proof_ref", f"seq-{rec['seq']}"), rec["seq"]


# --------------------------------------------------------------------- stats
@dataclass
class ProofRunStats:
    run_id: str
    shards: int
    proof_count: int = 0
    witness_count: int = 0
    source_occurrences: int = 0
    distinct_sources: int = 0
    output_rows: int = 0
    bytes_written: int = 0
    bytes_read: int = 0
    files: int = 0
    nodes: Dict[str, dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.nodes is None:
            self.nodes = {}

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "shards": self.shards,
            "proof_count": self.proof_count,
            "witness_count": self.witness_count,
            "source_occurrences": self.source_occurrences,
            "distinct_sources": self.distinct_sources,
            "output_rows": self.output_rows,
            "bytes_written": self.bytes_written,
            "bytes_read": self.bytes_read,
            "files": self.files,
            "nodes": self.nodes,
        }


# ------------------------------------------------------------- distinct set
class _BoundedDistinct:
    """Per-scope distinct-value counter under the shared memory budget.

    Small scopes stay in capped Python sets.  When a scope cannot reserve its
    next entry it (and only it) spills: subsequent values go to a JSONL spill
    file and its exact distinct count is finished by an external-sort pass at
    commit time.  No scope is ever fully materialised past the budget.
    """

    def __init__(self, budget: Budget, spill_path: str, owner: str) -> None:
        self.budget = budget
        self.spill_path = spill_path
        self.owner = owner
        self._sets: Dict[str, set] = {}
        self._reserved: Dict[str, int] = {}
        self._spilled: set = set()
        self._fh = None

    def add(self, scope: str, value: Any) -> None:
        value = tuple(value) if isinstance(value, list) else value
        if scope in self._spilled:
            self._spill_write(scope, value)
            return
        s = self._sets.get(scope)
        if s is None:
            s = set()
            self._sets[scope] = s
            self._reserved[scope] = 0
        if value in s:
            return
        try:
            self.budget.reserve(DISTINCT_ENTRY_COST, owner=self.owner)
        except Exception:
            # Scope overflows the shared budget: move it entirely to disk.
            self._spill_scope(scope)
            self._spill_write(scope, value)
            return
        s.add(value)
        self._reserved[scope] += DISTINCT_ENTRY_COST

    def _spill_scope(self, scope: str) -> None:
        for v in self._sets.pop(scope):
            self._spill_write(scope, v)
        n = self._reserved.pop(scope, 0)
        if n:
            self.budget.release(n, owner=self.owner)
        self._spilled.add(scope)

    def _spill_write(self, scope: str, value: Any) -> None:
        if self._fh is None:
            try:
                self._fh = open(self.spill_path, "wb", buffering=1 << 16)
            except OSError as exc:
                raise ProofError(f"cannot open distinct spill: {exc}") from exc
        payload = (canonical_json({"s": scope, "v": value}) + "\n").encode()
        try:
            self._fh.write(payload)
        except OSError as exc:
            raise ProofError(f"distinct spill write failed: {exc}") from exc

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def release_all(self) -> None:
        for scope, n in list(self._reserved.items()):
            if n:
                self.budget.release(n, owner=self.owner)
        self._reserved.clear()
        self._sets.clear()
        self.close()

    def in_memory_counts(self) -> Dict[str, int]:
        return {scope: len(s) for scope, s in self._sets.items()}

    def spilled_scopes(self) -> List[str]:
        return sorted(self._spilled)

    def spilled_counts(self, mem_bytes: int,
                       read_counter: Callable[[int], None]) -> Dict[str, int]:
        """Exact distinct counts for spilled scopes via external sort."""
        self.close()
        if not self._spilled or not os.path.exists(self.spill_path):
            return {}
        work_dir = os.path.dirname(self.spill_path)
        sorted_path = _external_sort_file(
            self.spill_path, work_dir, mem_bytes, self.budget, self.owner,
            read_counter)
        counts: Dict[str, int] = {}
        last_scope: Optional[str] = None
        last_line = None
        with open(sorted_path, "rb") as fh:
            for line in fh:
                read_counter(len(line))
                if line == last_line:
                    continue
                last_line = line
                scope = json.loads(line)["s"]
                if scope != last_scope:
                    counts[scope] = 1
                    last_scope = scope
                else:
                    counts[scope] += 1
        try:
            os.remove(sorted_path)
        except OSError:
            pass
        return counts


# ------------------------------------------------------------- external sort
#: Maximum simultaneously open chunk readers during a proof external sort.
#: Beyond this, runs are merged in rounds so file-descriptor pressure stays
#: bounded even when the shared budget forces very small chunks.
MAX_SORT_RUNS = 8
#: Floor for one sort chunk's live bytes so tiny budgets do not produce a
#: run-per-line storm.  Chunk data lives on disk between flushes; like the
#: spill subsystem, only open file handles are budget-charged, and the
#: byte target still bounds actual live memory.
MIN_SORT_CHUNK_BYTES = 16 * 1024


def _write_sorted_run(items: List[Any], work_dir: str,
                      key_fn: Optional[Callable[[bytes], Any]]) -> str:
    items.sort(key=(lambda item: item[0]) if key_fn is not None else None)
    fd, path = _mkstemp(work_dir, "sort-", ".jsonl")
    os.close(fd)
    # Intermediate runs are private staging artifacts; only final published
    # files need fsync durability, so runs use plain buffered writes.
    with open(path, "wb", buffering=1 << 16) as fh:
        for item in items:
            fh.write(item[1] if key_fn is not None else item)
    return path


def _merge_runs(run_paths: List[str], work_dir: str,
                key_fn: Optional[Callable[[bytes], Any]]) -> str:
    """Merge at most MAX_SORT_RUNS sorted files into one new run."""
    handles = [open(p, "rb", buffering=1 << 16) for p in run_paths]
    fd, out_path = _mkstemp(work_dir, "sortm-", ".jsonl")
    os.close(fd)
    try:
        def keyed(line: bytes) -> Any:
            return key_fn(line) if key_fn is not None else line

        streams = [((keyed(line), line) for line in h)
                   if key_fn is not None else h for h in handles]
        with open(out_path, "wb", buffering=1 << 16) as out:
            for item in heapq.merge(*streams):
                out.write(item[1] if key_fn is not None else item)
    except OSError as exc:
        raise ProofError(f"external sort merge failed: {exc}") from exc
    finally:
        for fh in handles:
            fh.close()
        for p in run_paths:
            try:
                os.remove(p)
            except OSError:
                pass
    return out_path


def _external_sort_file(input_path: str, work_dir: str, mem_bytes: int,
                        budget: Budget, owner: str,
                        read_counter: Callable[[int], None],
                        key_fn: Optional[Callable[[bytes], Any]] = None
                        ) -> str:
    """Sort the JSONL lines of ``input_path`` using bounded chunks.

    Returns the path of a sorted file (the input itself when it fits one
    chunk).  Live chunk memory is hard-bounded by a byte target derived from
    the shared budget (with a floor so tiny budgets do not create a
    run-per-line storm).  Chunk contents are on disk between flushes; as in
    the spill subsystem, data bytes are not budget-reserved — only open
    handles are, best-effort — while the byte target bounds real memory.
    Runs merge in rounds of at most :data:`MAX_SORT_RUNS`.
    """
    target = max(MIN_SORT_CHUNK_BYTES, mem_bytes)
    run_paths: List[str] = []
    chunk: List[Any] = []
    chunk_bytes = 0

    def keyed(line: bytes) -> Any:
        return key_fn(line) if key_fn is not None else line

    def wrap(line: bytes) -> Any:
        return (keyed(line), line) if key_fn is not None else line

    def flush() -> None:
        nonlocal chunk, chunk_bytes
        if not chunk:
            return
        run_paths.append(_write_sorted_run(chunk, work_dir, key_fn))
        chunk = []
        chunk_bytes = 0
        if len(run_paths) >= MAX_SORT_RUNS:
            merged = _merge_runs(run_paths, work_dir, key_fn)
            run_paths.clear()
            run_paths.append(merged)

    try:
        with open(input_path, "rb") as fh:
            for line in fh:
                read_counter(len(line))
                chunk.append(wrap(line))
                chunk_bytes += len(line)
                if chunk_bytes >= target:
                    flush()
        flush()
    except OSError as exc:
        raise ProofError(f"external sort read failed: {exc}") from exc

    if len(run_paths) <= 1:
        return run_paths[0] if run_paths else input_path
    return _merge_runs(run_paths, work_dir, key_fn)


def _mkstemp(work_dir: str, prefix: str, suffix: str) -> Tuple[int, str]:
    try:
        return tempfile.mkstemp(
            prefix=prefix, suffix=suffix, dir=work_dir)
    except OSError as exc:
        raise ProofError(f"cannot create temp file in {work_dir}: {exc}") from exc


# ----------------------------------------------------------------- the store
class ProofStore:
    """Owns the durable proof directory and one run's write/commit lifecycle."""

    def __init__(self, base_dir: str, budget: Budget,
                 shards: int = DEFAULT_SHARDS) -> None:
        self.base_dir = base_dir
        self.budget = budget
        self.shards = shards
        try:
            os.makedirs(base_dir, exist_ok=True)
        except OSError as exc:
            raise ProofError(
                f"cannot create proof directory {base_dir}: {exc}") from exc
        #: Test hook: called as ``write_fault(path)`` before every write;
        #: raise OSError to simulate disk exhaustion/failure.
        self.write_fault: Optional[Callable[[str], None]] = None
        self._run_dir: Optional[str] = None
        self._run_id: Optional[str] = None
        self._event_fhs: Dict[int, Any] = {}
        self._out_fhs: Dict[int, Any] = {}
        self._stats: Optional[ProofRunStats] = None
        self._distinct: Optional[_BoundedDistinct] = None
        self._closed = False

    # ------------------------------------------------------------ lifecycle
    @property
    def run_dir(self) -> str:
        if self._run_dir is None:
            raise ProofError("proof run has not been started")
        return self._run_dir

    def begin_run(self, run_id: str) -> str:
        run_dir = os.path.join(self.base_dir, run_id)
        if os.path.exists(os.path.join(run_dir, "manifest.json")):
            raise ProofError(
                f"proof run {run_id!r} already committed; choose a fresh run id")
        # Re-create staging cleanly; a previous failed attempt leaves no
        # manifest but may leave debris.
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
        for sub in ("staging/ev", "staging/out", "data", "index", "mapping"):
            self._makedirs(os.path.join(run_dir, sub))
        self._run_dir = run_dir
        self._run_id = run_id
        self._stats = ProofRunStats(run_id=run_id, shards=self.shards)
        self._distinct = _BoundedDistinct(
            self.budget,
            os.path.join(run_dir, "staging", "distinct-spill.jsonl"),
            owner="proof-distinct")
        return run_dir

    def _makedirs(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            raise ProofError(f"cannot create {path}: {exc}") from exc

    def abort(self) -> None:
        """Remove every artifact of this run; never leave a manifest."""
        if self._distinct is not None:
            self._distinct.release_all()
        for fh in list(self._event_fhs.values()) + list(self._out_fhs.values()):
            try:
                fh.close()
            except OSError:
                pass
        self._event_fhs.clear()
        self._out_fhs.clear()
        if self._run_dir is not None and os.path.isdir(self._run_dir):
            shutil.rmtree(self._run_dir, ignore_errors=True)
        self._closed = True

    # ---------------------------------------------------------------- writes
    def _shard_of(self, pid: str) -> int:
        return int(hashlib.sha256(pid.encode("utf-8")).hexdigest(), 16) \
            % self.shards

    def _open_shard(self, fhs: Dict[int, Any], sub: str, tag: str,
                    shard: int) -> Any:
        fh = fhs.get(shard)
        if fh is not None:
            return fh
        path = os.path.join(self.run_dir, "staging", sub,
                            f"{tag}-{shard:02x}.jsonl")
        try:
            fh = open(path, "wb", buffering=1 << 16)
        except OSError as exc:
            raise ProofError(f"cannot open proof shard {path}: {exc}") from exc
        # Handles are best-effort charged, like spill handles: proof writing
        # itself must never be blocked by an exhausted logical budget.
        try:
            self.budget.reserve(SPILL_HANDLE_BYTES, owner=f"proof:{tag}")
        except Exception:
            pass
        fhs[shard] = fh
        self._stats.files += 1
        return fh

    def _write_line(self, fh: Any, path: str, payload: bytes) -> None:
        if self.write_fault is not None:
            try:
                self.write_fault(path)
            except OSError as exc:
                raise ProofError(str(exc)) from exc
        try:
            fh.write(payload)
        except OSError as exc:
            raise ProofError(f"proof write failed ({path}): {exc}") from exc
        self._stats.bytes_written += len(payload)

    def write_event(self, pid: str, event: dict) -> None:
        shard = self._shard_of(pid)
        fh = self._open_shard(self._event_fhs, "ev", "ev", shard)
        path = fh.name
        record = dict(event)
        record["pid"] = pid
        # Explicit ordering within one proof: registration first, then
        # witnesses/contributions.  Text sorting alone cannot guarantee that.
        record["ord"] = 0 if event["t"] == "r" else 1
        self._write_line(
            fh, path,
            (canonical_json(record) + "\n").encode("utf-8"))

    def write_output_mapping(self, record: dict) -> None:
        pid = record.get("row_proof_ref") or f"seq-{record['seq']}"
        shard = self._shard_of(pid)
        fh = self._open_shard(self._out_fhs, "out", "out", shard)
        self._write_line(
            fh, fh.name,
            (canonical_json(record) + "\n").encode("utf-8"))
        self._stats.output_rows += 1

    def note_source(self, pid: str, source: Tuple[str, int]) -> None:
        """Account one base-source occurrence inside a proof."""
        self._distinct.add(pid, list(source))
        self._stats.source_occurrences += 1

    # ---------------------------------------------------------------- commit
    def commit(self, node_descriptors: Dict[str, dict],
               query_digest: str) -> dict:
        """Freeze all events into sorted proof documents and publish.

        On any failure the run directory is removed and the error is
        re-raised; no manifest is ever published for a failed commit.
        """
        assert self._stats is not None and self._distinct is not None
        try:
            return self._commit(node_descriptors, query_digest)
        except (ProofError, RunCancelled, OSError):
            self.abort()
            raise

    def _commit(self, node_descriptors: Dict[str, dict],
                query_digest: str) -> dict:
        for fh in list(self._event_fhs.values()) + list(self._out_fhs.values()):
            try:
                fh.flush()
                os.fsync(fh.fileno())
                fh.close()
            except OSError as exc:
                raise ProofError(f"proof flush failed: {exc}") from exc
        self._event_fhs.clear()
        self._out_fhs.clear()

        mem_bytes = max(1, self.budget.limit_bytes // 8)
        node_rollup: Dict[str, dict] = {}

        def count_read(n: int) -> None:
            self._stats.bytes_read += n

        for shard in range(self.shards):
            ev_path = os.path.join(
                self.run_dir, "staging", "ev", f"ev-{shard:02x}.jsonl")
            if not os.path.exists(ev_path):
                continue
            sorted_path = _external_sort_file(
                ev_path, os.path.join(self.run_dir, "staging"),
                mem_bytes, self.budget, "proof-sort", count_read,
                key_fn=event_sort_key)
            self._commit_shard(shard, sorted_path, node_rollup, count_read)
            if sorted_path != ev_path:
                try:
                    os.remove(sorted_path)
                except OSError:
                    pass

        # Exact distinct counts (spilled scopes finished by external sort).
        in_mem = self._distinct.in_memory_counts()
        spilled = self._distinct.spilled_counts(mem_bytes, count_read)
        distinct_counts = {**in_mem, **spilled}
        self._stats.distinct_sources = distinct_counts.get("__global__", 0)
        self._annotate_distinct_counts(distinct_counts)

        # Output mapping shards: sort and publish with a sparse directory.
        self._commit_mappings(mem_bytes, count_read)
        self._distinct.release_all()

        # Node descriptors: explicit aggregate metadata plus the observed
        # per-node rollup (proof/witness/source counts and bytes).
        observed_nodes = {
            nid: dict(roll) for nid, roll in node_rollup.items()}
        for nid, desc in node_descriptors.items():
            observed_nodes.setdefault(nid, {}).update(desc)
        self._stats.nodes = observed_nodes
        manifest = {
            "format": PROOF_FORMAT,
            "run_id": self._run_id,
            "query_digest": query_digest,
            "shards": self.shards,
            "complete": True,
            "stats": self._stats.snapshot(),
            "nodes": observed_nodes,
            "node_rollup": node_rollup,
        }
        tmp = os.path.join(self.run_dir, "manifest.json.tmp")
        final = os.path.join(self.run_dir, "manifest.json")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, final)
        except OSError as exc:
            raise ProofError(f"cannot publish manifest: {exc}") from exc
        self._cleanup_staging()
        self._closed = True
        return manifest

    def _commit_shard(self, shard: int, sorted_path: str,
                      node_rollup: Dict[str, dict],
                      count_read: Callable[[int], None]) -> None:
        """Stream sorted events into framed proof documents.

        Only the events of the proof currently being finalised are held; a
        proof with millions of contributors therefore needs constant
        per-proof memory.  Exact per-proof and global distinct source counts
        go through the budget-bounded :class:`_BoundedDistinct` tracker.
        """
        data_path = os.path.join(self.run_dir, "data",
                                 f"shard-{shard:02x}.jsonl")
        index_path = os.path.join(self.run_dir, "index",
                                  f"shard-{shard:02x}.jsonl")
        dir_path = os.path.join(self.run_dir, "index",
                                f"shard-{shard:02x}.dir.jsonl")
        global_scope = "__global__"
        current_pid: Optional[str] = None
        register: Optional[dict] = None
        n_wit = 0
        n_sources = 0
        per_proof_distinct: set = set()
        doc_start = 0
        index_lines: List[dict] = []
        data_fh = self._open_final(data_path)
        index_fh = self._open_final(index_path)
        dir_fh = self._open_final(dir_path)
        try:
            with open(sorted_path, "rb") as fh:
                last_line = None
                for raw in fh:
                    count_read(len(raw))
                    if raw == last_line:
                        continue  # duplicate event lines are idempotent
                    last_line = raw
                    rec = json.loads(raw)
                    pid = rec["pid"]
                    if current_pid is not None and pid != current_pid:
                        self._finalize_proof(
                            data_fh, data_path, current_pid,
                            register, n_wit, len(per_proof_distinct),
                            doc_start, index_lines, node_rollup)
                        register = None
                        n_wit = 0
                        per_proof_distinct = set()
                    current_pid = pid
                    t = rec["t"]
                    if t == "r":
                        if register is None:
                            doc_start = data_fh.tell()
                            self._write_frame(data_fh, data_path,
                                              "h", rec)
                        # Repeated registrations only backfill the metric
                        # value; the header was already streamed, so keep
                        # the first frame (its value is finalised at read
                        # time from the aggregate output mapping).
                        register = rec
                    elif t == "w":
                        self._write_frame(data_fh, data_path,
                                          "w", rec["w"])
                        n_wit += 1
                        for s in rec["w"].get("sources", []):
                            src = tuple(s)
                            per_proof_distinct.add(src)
                            self._distinct.add(pid, src)
                            self._distinct.add(global_scope, src)
                            n_sources += 1
                    elif t == "c":
                        self._write_frame(data_fh, data_path,
                                          "c", rec["c"])
                        n_wit += 1
                        for s in rec["c"].get("sources", []):
                            src = tuple(s)
                            per_proof_distinct.add(src)
                            self._distinct.add(pid, src)
                            self._distinct.add(global_scope, src)
                            n_sources += 1
            if current_pid is not None:
                self._finalize_proof(
                    data_fh, data_path, current_pid,
                    register, n_wit, len(per_proof_distinct),
                    doc_start, index_lines, node_rollup)
            data_fh.flush()
            os.fsync(data_fh.fileno())
            self._stats.source_occurrences += n_sources
            index_offset = 0
            for i, entry in enumerate(index_lines):
                if i % INDEX_SAMPLE_EVERY == 0:
                    dir_fh.write((canonical_json(
                        {"pid": entry["p"], "off": index_offset})
                        + "\n").encode())
                line = (canonical_json(entry) + "\n").encode("utf-8")
                index_fh.write(line)
                index_offset += len(line)
                self._stats.bytes_written += len(line)
            index_fh.flush()
            dir_fh.flush()
            os.fsync(index_fh.fileno())
            os.fsync(dir_fh.fileno())
        finally:
            data_fh.close()
            index_fh.close()
            dir_fh.close()
            self._stats.files += 3

    def _write_frame(self, fh: Any, path: str, frame: str,
                     payload: Any) -> None:
        record = {"f": frame}
        if frame == "h":
            # payload is the registration event
            record.update({
                "proof_ref": payload["pid"],
                "node": payload["node"],
                "node_kind": payload.get("node_kind", payload["kind"]),
                "kind": payload["kind"],
                "group_key": payload.get("group_key"),
                "metric_value": payload.get("value"),
                "null_contributors": payload.get("null_contributors"),
            })
        else:
            record["d"] = payload
        self._write_line(fh, path,
                         (canonical_json(record) + "\n").encode("utf-8"))

    def _finalize_proof(self, data_fh: Any, data_path: str,
                        pid: str, register: Optional[dict],
                        n_wit: int, n_distinct: int, doc_start: int,
                        index_lines: List[dict],
                        node_rollup: Dict[str, dict]) -> int:
        if register is None:
            raise ProofError(f"proof {pid} has events but no registration")
        node_id = register["node"]
        kind = register["kind"]
        trailer = {"f": "t", "kind": kind,
                   "no_right_match": kind == "pad"}
        self._write_line(data_fh, data_path,
                         (canonical_json(trailer) + "\n").encode("utf-8"))
        end = data_fh.tell()
        index_lines.append({
            "p": pid, "off": doc_start, "len": end - doc_start,
            "w": n_wit, "s": n_distinct,
        })
        self._stats.proof_count += 1
        self._stats.witness_count += n_wit
        roll = node_rollup.setdefault(node_id, {
            "proofs": 0, "witnesses": 0, "source_occurrences": 0,
            "distinct_sources": 0})
        roll["proofs"] += 1
        roll["witnesses"] += n_wit
        roll["source_occurrences"] += n_wit
        roll["distinct_sources"] += n_distinct
        return end

    def _open_final(self, path: str) -> Any:
        try:
            return open(path, "wb", buffering=1 << 16)
        except OSError as exc:
            raise ProofError(f"cannot open committed file {path}: {exc}") from exc

    def _annotate_distinct_counts(self,
                                  distinct_counts: Dict[str, int]) -> None:
        """Rewrite the per-proof index ``s`` field where the streaming
        in-memory counter had to spill, using the exact sorted count."""
        spilled_pids = {scope for scope in distinct_counts
                        if scope != "__global__"}
        if not spilled_pids:
            return
        for shard in range(self.shards):
            index_path = os.path.join(self.run_dir, "index",
                                      f"shard-{shard:02x}.jsonl")
            if not os.path.exists(index_path):
                continue
            corrected: List[bytes] = []
            with open(index_path, "rb") as fh:
                for raw in fh:
                    entry = json.loads(raw)
                    if entry["p"] in spilled_pids:
                        entry["s"] = distinct_counts[entry["p"]]
                    corrected.append(
                        (canonical_json(entry) + "\n").encode())
            tmp = index_path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.writelines(corrected)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, index_path)

    def _commit_mappings(self, mem_bytes: int,
                         count_read: Callable[[int], None]) -> None:
        for shard in range(self.shards):
            out_path = os.path.join(
                self.run_dir, "staging", "out", f"out-{shard:02x}.jsonl")
            if not os.path.exists(out_path):
                continue
            sorted_path = _external_sort_file(
                out_path, os.path.join(self.run_dir, "staging"),
                mem_bytes, self.budget, "proof-map-sort", count_read,
                key_fn=mapping_sort_key)
            dest = os.path.join(self.run_dir, "mapping",
                                f"shard-{shard:02x}.jsonl")
            dir_path = os.path.join(self.run_dir, "mapping",
                                    f"shard-{shard:02x}.dir.jsonl")
            fh = self._open_final(dest)
            dfh = self._open_final(dir_path)
            offset = 0
            i = 0
            try:
                with open(sorted_path, "rb") as src:
                    last = None
                    for raw in src:
                        count_read(len(raw))
                        if raw == last:
                            continue
                        last = raw
                        if i % INDEX_SAMPLE_EVERY == 0:
                            dfh.write((canonical_json(
                                {"pid": json.loads(raw).get(
                                    "row_proof_ref", ""),
                                 "off": offset}) + "\n").encode())
                        fh.write(raw)
                        offset += len(raw)
                        i += 1
                fh.flush()
                dfh.flush()
                os.fsync(fh.fileno())
                os.fsync(dfh.fileno())
            finally:
                fh.close()
                dfh.close()
                self._stats.files += 2
            if sorted_path != out_path:
                try:
                    os.remove(sorted_path)
                except OSError:
                    pass

    def _cleanup_staging(self) -> None:
        staging = os.path.join(self.run_dir, "staging")
        shutil.rmtree(staging, ignore_errors=True)


# ------------------------------------------------------------------ reader
class ProofReader:
    """Reads committed proofs.  Works after the executor and its temp/spill
    directories are gone — only the durable proof directory is needed."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        manifest_path = os.path.join(run_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise ProofError(
                f"no complete proof manifest in {run_dir}; "
                f"the run did not commit successfully")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        if not self.manifest.get("complete"):
            raise ProofError(f"manifest in {run_dir} is not marked complete")
        self.shards = self.manifest["shards"]
        self._dir_cache: Dict[Tuple[str, int], List[dict]] = {}

    @classmethod
    def open(cls, base_dir: str, run_id: str) -> "ProofReader":
        return cls(os.path.join(base_dir, run_id))

    def stats(self) -> dict:
        return dict(self.manifest["stats"])

    def node_rollup(self) -> dict:
        return dict(self.manifest["node_rollup"])

    def _shard_of(self, pid: str) -> int:
        return int(hashlib.sha256(pid.encode("utf-8")).hexdigest(), 16) \
            % self.shards

    def _load_dir(self, sub: str, shard: int) -> List[dict]:
        key = (sub, shard)
        if key not in self._dir_cache:
            path = os.path.join(self.run_dir, sub,
                                f"shard-{shard:02x}.dir.jsonl")
            entries: List[dict] = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        entries.append(json.loads(line))
            self._dir_cache[key] = entries
        return self._dir_cache[key]

    def _index_entry(self, proof_ref: str) -> Tuple[int, dict]:
        import bisect
        shard = self._shard_of(proof_ref)
        index_path = os.path.join(self.run_dir, "index",
                                  f"shard-{shard:02x}.jsonl")
        if not os.path.exists(index_path):
            raise KeyError(proof_ref)
        directory = self._load_dir("index", shard)
        pids = [d["pid"] for d in directory]
        pos = bisect.bisect_right(pids, proof_ref) - 1
        start = directory[pos]["off"] if pos >= 0 else 0
        with open(index_path, "r", encoding="utf-8") as fh:
            fh.seek(start)
            entry = None
            for line in fh:
                rec = json.loads(line)
                if rec["p"] == proof_ref:
                    entry = rec
                    break
                if rec["p"] > proof_ref:
                    break
        if entry is None:
            raise KeyError(proof_ref)
        return shard, entry

    def iter_proof_frames(self, proof_ref: str) -> Iterator[dict]:
        """Stream the raw frames of one proof without assembling it.

        A high-fanout count/sum proof can therefore be consumed with
        constant reader-side memory.  Frames are ``h`` (header), ``w``
        (witness), ``c`` (contribution) and ``t`` (trailer).
        """
        shard, entry = self._index_entry(proof_ref)
        data_path = os.path.join(self.run_dir, "data",
                                 f"shard-{shard:02x}.jsonl")
        with open(data_path, "rb") as fh:
            fh.seek(entry["off"])
            remaining = entry["len"]
            while remaining > 0:
                raw = fh.readline()
                if not raw:
                    raise ProofError(f"truncated proof frame for {proof_ref}")
                remaining -= len(raw)
                yield json.loads(raw)

    def get_proof(self, proof_ref: str) -> dict:
        """Assemble one proof document from its frames.

        Convenience API; for proofs whose witness/contribution count may be
        large prefer :meth:`iter_proof_frames`.
        """
        doc: Dict[str, Any] = {}
        witnesses: List[dict] = []
        contributions: List[dict] = []
        for frame in self.iter_proof_frames(proof_ref):
            f = frame["f"]
            if f == "h":
                doc = {
                    "proof_ref": frame["proof_ref"],
                    "node": frame["node"],
                    "node_kind": frame["node_kind"],
                    "kind": frame["kind"],
                }
                if frame["kind"] in ("count", "sum"):
                    doc["group_key"] = frame.get("group_key") or []
                    doc["metric_value"] = frame.get("metric_value")
                    if frame["kind"] == "sum":
                        doc["excluded_null_contributors"] = \
                            frame.get("null_contributors", 0)
            elif f == "w":
                witnesses.append(frame["d"])
            elif f == "c":
                contributions.append(frame["d"])
            elif f == "t":
                if frame.get("no_right_match"):
                    doc["no_right_match"] = True
        kind = doc.get("kind")
        if kind in ("count", "sum"):
            doc["contributions"] = contributions
        else:
            doc["witnesses"] = witnesses
        return doc

    def iter_proofs(self) -> Iterator[dict]:
        """Stream every proof document in shard/pid order (assembled)."""
        for shard in range(self.shards):
            index_path = os.path.join(self.run_dir, "index",
                                      f"shard-{shard:02x}.jsonl")
            if not os.path.exists(index_path):
                continue
            with open(index_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    yield self.get_proof(json.loads(line)["p"])

    def iter_index(self, token: Optional[str] = None,
                   page_size: int = 128
                   ) -> Tuple[List[dict], Optional[str]]:
        """Page through index entries (stable order, no row-order promise)."""
        shard, line_no = _parse_token(token)
        out: List[dict] = []
        next_token: Optional[str] = None
        while shard < self.shards and len(out) < page_size:
            index_path = os.path.join(self.run_dir, "index",
                                      f"shard-{shard:02x}.jsonl")
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        if i < line_no:
                            continue
                        out.append(json.loads(line))
                        if len(out) >= page_size:
                            consumed = i + 1
                            next_token = f"{shard}:{consumed}"
                            break
            if next_token is not None:
                break
            shard += 1
            line_no = 0
        return out, next_token

    def iter_output_mappings(self, token: Optional[str] = None,
                             page_size: int = 128
                             ) -> Tuple[List[dict], Optional[str]]:
        shard, line_no = _parse_token(token)
        out: List[dict] = []
        next_token: Optional[str] = None
        while shard < self.shards and len(out) < page_size:
            path = os.path.join(self.run_dir, "mapping",
                                f"shard-{shard:02x}.jsonl")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        if i < line_no:
                            continue
                        out.append(json.loads(line))
                        if len(out) >= page_size:
                            next_token = f"{shard}:{i + 1}"
                            break
            if next_token is not None:
                break
            shard += 1
            line_no = 0
        return out, next_token


def _parse_token(token: Optional[str]) -> Tuple[int, int]:
    if not token:
        return 0, 0
    shard_s, _, line_s = token.partition(":")
    return int(shard_s), int(line_s or "0")
