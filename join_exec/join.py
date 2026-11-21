"""Hybrid hash join with a real, runtime memory -> disk transition.

One operator owns both execution modes and the switch between them.  The
switch is never a static plan choice made before the first row:

* build-phase pressure (budget overflow, hot-key threshold, batch control)
  moves the partially built hash table into partition spill files and routes
  the rest of both inputs there;
* probe-phase pressure (a hot matching group, or an "after N outputs"
  control point) keeps the already emitted pairs, records the current left
  row together with the right row ids already delivered as a *residual*
  record, then partitions the remaining work.

Both paths call the same :meth:`HybridJoin._switch` method.  Residual
suppression makes replayed matches idempotent: a left row's matched state
follows it into exactly one partition, so LEFT JOIN pads only after the whole
side is proven unmatched and SEMI/INNER never duplicate.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .budget import Budget
from .control import ControlPoint
from .errors import JoinExecError, MemoryLimitExceeded, RunCancelled
from .spill import SpillManager, residual_to_record, row_to_record
from .types import (
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    HT_ENTRY_OVERHEAD,
    Header,
    Row,
    make_join_key,
    NULL_KEY,
    tag_key,
)

BATCH_SIZE = DEFAULT_BATCH_SIZE
MAX_PARTITIONS = 256
TOP_HOT_KEYS = 10

class Source:
    """Anything iterable of Rows that also carries an output Header."""

    def __init__(self, header: Header, rows: Callable[[], Iterator[Row]]) -> None:
        self.header = header
        self._rows = rows

    def __iter__(self) -> Iterator[Row]:
        return self._rows()


@dataclass
class JoinStats:
    join_id: str
    join_type: str
    left_alias: str
    right_alias: str
    mode: str = "memory"
    build_rows: int = 0
    probe_rows: int = 0
    outputs: int = 0
    left_padded: int = 0
    build_rid_range: Tuple[Optional[int], Optional[int]] = (None, None)
    probe_rid_range: Tuple[Optional[int], Optional[int]] = (None, None)
    switch_events: List[dict] = field(default_factory=list)
    hot_keys: List[Tuple[str, int]] = field(default_factory=list)
    partitions: int = 0
    merge_partitions: List[int] = field(default_factory=list)
    residual_count: int = 0
    bytes_reserved_peak: int = 0


class HybridJoin:
    """Adaptive equi-join over multi-column keys.

    Parameters
    ----------
    join_id:
        Stable identifier used by control points and stats.
    join_type:
        ``"inner" | "left" | "semi"``.
    left_refs / right_refs:
        Parallel ``(alias, column)`` key component lists.  A NULL in any
        component yields the non-matching NULL_KEY on both sides.
    mode:
        ``"adaptive"`` (switch when triggered), ``"force_memory"`` (never
        switch; budget errors are fatal) or ``"force_disk"`` (switch through
        the same code path before reading the first build row).
    skew_threshold:
        When a build-side key group reaches this many rows, or a probe group
        of at least this size is about to be expanded, the automatic switch
        fires.  ``None`` disables size-based triggering.
    """

    def __init__(self, join_id: str, join_type: str,
                 left: Source, right: Source,
                 left_refs: Sequence[Tuple[str, str]],
                 right_refs: Sequence[Tuple[str, str]],
                 budget: Budget, spill: SpillManager,
                 control: Optional[ControlPoint] = None,
                 mode: str = "adaptive",
                 skew_threshold: Optional[int] = None,
                 expected_build_rows: Optional[int] = None,
                 batch_size: int = BATCH_SIZE) -> None:
        if join_type not in ("inner", "left", "semi"):
            raise JoinExecError(f"unsupported join type {join_type!r}")
        if len(left_refs) != len(right_refs) or not left_refs:
            raise JoinExecError("join key column lists must be non-empty and aligned")
        if mode not in ("adaptive", "force_memory", "force_disk"):
            raise JoinExecError(f"bad mode {mode!r}")
        self.join_id = join_id
        self.join_type = join_type
        self.left = left
        self.right = right
        self.left_refs = tuple(left_refs)
        self.right_refs = tuple(right_refs)
        self.budget = budget
        self.spill = spill
        self.control = control
        self.mode = mode
        self.skew_threshold = skew_threshold
        self._expected_build_rows = expected_build_rows or 0
        self.batch_size = max(1, batch_size or BATCH_SIZE)
        self.header = left.header.union(right.header)
        self.stats = JoinStats(join_id, join_type,
                               left_refs[0][0], right_refs[0][0])
        self._owner = f"join:{join_id}"
        # Optional provenance sink.  When set, every emitted match/pad is
        # reported here *in addition* to yielding the business row; the sink
        # never alters values, predicates or the number of outputs.
        self.proof_sink: Optional["object"] = None
        self._reserved = 0
        # memory-phase state
        self._ht: Dict[Any, List[Row]] = {}
        self._group_count: Dict[Any, int] = {}
        self._phase = "init"
        self._pending_batch_switch = False
        self._npart = 1
        self._disk_group_count: Dict[Any, int] = {}

    # ------------------------------------------------------------------ utils
    def _reserve(self, n: int) -> None:
        self.budget.reserve(n, owner=self._owner)
        self._reserved += n
        self.stats.bytes_reserved_peak = max(self.stats.bytes_reserved_peak,
                                             self._reserved)

    def _release(self, n: int) -> None:
        self.budget.release(n, owner=self._owner)
        self._reserved -= n

    def _check_cancel(self) -> None:
        if self.control is not None and self.control.is_cancelled():
            raise RunCancelled(self.control.reason or "run cancelled")

    def _lkey(self, row: Row) -> Any:
        return make_join_key(row, self.left.header, self.left_refs)

    def _rkey(self, row: Row) -> Any:
        return make_join_key(row, self.right.header, self.right_refs)

    def _combine(self, lrow: Row, rrow: Row) -> Row:
        out = Row(tuple(lrow.values) + tuple(rrow.values),
                  tuple(lrow.rids) + tuple(rrow.rids))
        if self.proof_sink is not None:
            self.proof_sink.note_match(self.join_id, out)
        return out

    def _note_semi(self, lrow: Row, rrow: Row) -> None:
        if self.proof_sink is not None:
            # A SEMI output still carries one proof covering the left row
            # plus *every* real matching right row, so all matches are
            # reported even though the join yields the left row only once.
            self.proof_sink.note_semi_match(self.join_id, lrow, rrow)

    def _pad(self, lrow: Row) -> Row:
        width = len(self.right.header.slots)
        out = Row(tuple(lrow.values) + (None,) * width, lrow.rids,
                  is_pad=True)
        if self.proof_sink is not None:
            self.proof_sink.note_pad(self.join_id, out)
        return out

    def _rid(self, row: Row, alias: str) -> int:
        return row.rid_of(alias)

    # ------------------------------------------------------------------ run
    def __iter__(self) -> Iterator[Row]:
        if self.mode == "force_disk":
            # Same switch implementation, triggered before any row is read.
            yield from self._run_disk_from_start(reason="forced_disk")
            return
        yield from self._run_memory()

    # ------------------------------------------------- memory-phase pipeline
    def _run_memory(self) -> Iterator[Row]:
        self._phase = "build"
        right_iter = iter(self.right)
        try:
            for row in right_iter:
                self._check_cancel()
                key = self._rkey(row)
                if key is NULL_KEY:
                    # NULL keys can never match; still charged in memory build
                    # accounting but held separately so hot stats stay clean.
                    pass
                cost = row.size + HT_ENTRY_OVERHEAD
                try:
                    self._reserve(cost)
                except MemoryLimitExceeded:
                    if self.mode == "force_memory":
                        raise
                    # Build-phase pressure switch: hand off this row too.
                    yield from self._switch_build_phase(right_iter, row,
                                                        reason="memory_pressure")
                    return
                self._ht.setdefault(key, []).append(row)
                if key is not NULL_KEY:
                    self._group_count[key] = self._group_count.get(key, 0) + 1
                self.stats.build_rows += 1
                self._note_rid_range(self.stats, "build",
                                     self._rid(row, self.right_refs[0][0]))
                # Batch boundaries align with BATCH_SIZE on both build and
                # probe, so "switch after N input batches" is consistent.
                batch_due = self.stats.build_rows % self.batch_size == 0
                if self._auto_skew_build(key) or (batch_due and self._batch_switch()):
                    if self.mode == "force_memory":
                        raise JoinExecError(
                            f"switch requested for {self.join_id} but mode=force_memory")
                    yield from self._switch_build_phase(right_iter, None,
                                                        reason="skew_or_control")
                    return
        except RunCancelled:
            self._discard_ht()
            raise

        # Build complete; probe phase.
        self._phase = "probe"
        self._record_hot_keys()
        self._pending_batch_switch = False
        left_iter = iter(self.left)
        try:
            for lrow in left_iter:
                self._check_cancel()
                self.stats.probe_rows += 1
                self._note_rid_range(self.stats, "probe",
                                     self._rid(lrow, self.left_refs[0][0]))
                if self.stats.probe_rows % self.batch_size == 0 and self._batch_switch():
                    self._pending_batch_switch = True
                key = self._lkey(lrow)
                if key is NULL_KEY:
                    if self.join_type == "left":
                        self.stats.left_padded += 1
                        yield self._pad(lrow)
                    continue
                matches = self._ht.get(key)
                if not matches:
                    if self.join_type == "left":
                        self.stats.left_padded += 1
                        yield self._pad(lrow)
                    continue
                if self._pending_batch_switch or self._auto_skew_probe(len(matches)):
                    self._pending_batch_switch = False
                    if self.mode == "force_memory":
                        raise JoinExecError(
                            f"switch requested for {self.join_id} but mode=force_memory")
                    yield from self._switch_probe_phase(
                        left_iter, lrow, key, suppress=set(),
                        reason="skew_or_control")
                    return
                yield from self._probe_match(lrow, matches, left_iter, key)
        except RunCancelled:
            self._discard_ht()
            raise
        self._discard_ht()

    def _probe_match(self, lrow: Row, matches: List[Row],
                     left_iter: Iterator[Row], key: Any) -> Iterator[Row]:
        emitted_right_rids: set = set()
        if self.join_type == "semi":
            # One output for this left row, but every real right match is
            # reported to the proof sink so its proof lists all witnesses.
            self.stats.outputs += 1
            for rrow in matches:
                self._note_semi(lrow, rrow)
            yield lrow
            return
        for rrow in matches:
            emitted_right_rids.add(self._rid(rrow, self.right_refs[0][0]))
            self.stats.outputs += 1
            yield self._combine(lrow, rrow)
            if self.control is not None and self.control.note_output(self.join_id):
                # Mid-group switch: residual keeps this left row's matched state.
                if self.mode == "force_memory":
                    raise JoinExecError(
                        f"switch requested for {self.join_id} but mode=force_memory")
                yield from self._switch_probe_phase(
                    left_iter, lrow, key, suppress=emitted_right_rids,
                    reason="after_output_control")
                return
        else:
            # Full group expanded in memory: nothing left for this row.
            if self.join_type == "left":
                pass  # matched, no pad
            return

    def _auto_skew_build(self, key: Any) -> bool:
        if self.skew_threshold is None or key is NULL_KEY:
            return False
        return self._group_count.get(key, 0) >= self.skew_threshold

    def _auto_skew_probe(self, group_size: int) -> bool:
        if self.skew_threshold is None:
            return False
        return group_size >= self.skew_threshold

    def _batch_switch(self) -> bool:
        return bool(self.control and self.control.note_input_batch(self.join_id))

    # ------------------------------------------------------------- switches
    def _switch_build_phase(self, right_iter: Iterator[Row],
                            pending_right: Optional[Row], reason: str
                            ) -> Iterator[Row]:
        """Switch while still building; no outputs exist yet."""
        # Size partitions from the still-reserved build volume, then snapshot
        # and release the partial hash table so spill handles share budget.
        npart = self._partition_count()
        partial = list(self._ht.items())
        self._discard_ht()
        writers = self._make_partition_writers(npart)
        # 1. flush partial hash table
        for key, rows in partial:
            for r in rows:
                self._write_build(writers, key, r)
        # 2. route the row that triggered the switch and the rest of right
        if pending_right is not None:
            key = self._rkey(pending_right)
            self._reserve_then_spill_build(writers, key, pending_right)
            self.stats.build_rows += 1
            self._note_rid_range(self.stats, "build",
                                 self._rid(pending_right, self.right_refs[0][0]))
        for row in right_iter:
            self._check_cancel()
            key = self._rkey(row)
            self._reserve_then_spill_build(writers, key, row)
            self.stats.build_rows += 1
            self._note_rid_range(self.stats, "build",
                                 self._rid(row, self.right_refs[0][0]))
        # 3. route all of left
        for lrow in self.left:
            self._check_cancel()
            key = self._lkey(lrow)
            self._write_left(writers, key, lrow, suppress=None)
            self.stats.probe_rows += 1
            self._note_rid_range(self.stats, "probe",
                                 self._rid(lrow, self.left_refs[0][0]))
        yield from self._finish_switch_and_run(writers, reason, residual=None)

    def _switch_probe_phase(self, left_iter: Iterator[Row], current_left: Row,
                            current_key: Any, suppress: set, reason: str
                            ) -> Iterator[Row]:
        """Switch while probing; ``current_left`` may already have outputs."""
        npart = self._partition_count()
        partial = list(self._ht.items())
        self._discard_ht()
        writers = self._make_partition_writers(npart)
        # 1. flush the complete build side into partitions
        for key, rows in partial:
            for r in rows:
                self._write_build(writers, key, r)
        # 2. residual record for the in-flight left row (if anything emitted)
        residual = None
        if suppress:
            residual = (current_left, set(suppress))
            self.stats.residual_count += 1
            self._write_left(writers, current_key, current_left,
                             suppress=set(suppress))
        else:
            # switch before this row produced anything: route it normally
            self._write_left(writers, current_key, current_left, suppress=None)
        # 3. remaining left rows
        for lrow in left_iter:
            self._check_cancel()
            key = self._lkey(lrow)
            self._write_left(writers, key, lrow, suppress=None)
            self.stats.probe_rows += 1
            self._note_rid_range(self.stats, "probe",
                                 self._rid(lrow, self.left_refs[0][0]))
        yield from self._finish_switch_and_run(writers, reason, residual=residual)

    def _run_disk_from_start(self, reason: str) -> Iterator[Row]:
        npart = self._partition_count()
        writers = self._make_partition_writers(npart)
        for row in self.right:
            self._check_cancel()
            key = self._rkey(row)
            self._write_build(writers, key, row)
            self.stats.build_rows += 1
            self._note_rid_range(self.stats, "build",
                                 self._rid(row, self.right_refs[0][0]))
        for lrow in self.left:
            self._check_cancel()
            key = self._lkey(lrow)
            self._write_left(writers, key, lrow, suppress=None)
            self.stats.probe_rows += 1
            self._note_rid_range(self.stats, "probe",
                                 self._rid(lrow, self.left_refs[0][0]))
        yield from self._finish_switch_and_run(writers, reason, residual=None)

    # ------------------------------------------------------- partition IO
    def _partition_count(self) -> int:
        """Choose fan-out so one average partition fits a quarter-budget.

        Two effects are handled:

        * a build-phase switch may fire early (small current reservation), so
          when the full build cardinality is known we also size from the
          average per-row cost times the number of build rows still expected;
        * the result is always at least large enough that one partition's
          worth of *build* rows fits the hash-path share, otherwise every
          partition would fall back to external sort-merge anyway.
        """
        from .types import SPILL_HANDLE_BYTES
        target_partition_bytes = max(1, self.budget.limit_bytes // 4)
        candidates = [4]
        if self._reserved > 0:
            candidates.append(
                (self._reserved + target_partition_bytes - 1)
                // target_partition_bytes)
        # Extrapolate total build volume from rows seen so far.
        if self.stats.build_rows > 0:
            avg = self._reserved / max(1, self.stats.build_rows)
            candidates.append(int(
                avg * self._expected_build_rows / target_partition_bytes) + 1)
        target = max(candidates)
        handle_cost = 2 * SPILL_HANDLE_BYTES
        max_by_handles = max(
            1, (self.budget.limit_bytes - 2 * SPILL_HANDLE_BYTES)
            // max(1, handle_cost))
        return min(MAX_PARTITIONS, target, max_by_handles)

    def _part_index(self, key: Tuple[Any, ...]) -> int:
        return zlib.crc32(repr(tag_key(key)).encode("utf-8")) % self._npart

    def _make_partition_writers(self, npart: int) -> dict:
        self._npart = npart
        self.stats.partitions = npart
        w: Dict[str, Any] = {"build": [], "left": [], "build_null": None,
                             "left_null": None}
        for i in range(npart):
            w["build"].append(self.spill.new_file(tag=f"b{i}"))
            w["left"].append(self.spill.new_file(tag=f"l{i}"))
        w["build_null"] = self.spill.new_file(tag="bnull")
        w["left_null"] = self.spill.new_file(tag="lnull")
        return w

    def _write_build(self, writers: dict, key: Any, row: Row) -> None:
        if key is NULL_KEY:
            writers["build_null"].write_row(row)
        else:
            writers["build"][self._part_index(key)].write_row(row)
            self._disk_group_count[key] = self._disk_group_count.get(key, 0) + 1

    def _reserve_then_spill_build(self, writers: dict, key: Any, row: Row) -> None:
        # In build-phase switch the row that failed reservation goes straight
        # to disk instead of being charged.
        self._write_build(writers, key, row)

    def _write_left(self, writers: dict, key: Any, row: Row,
                    suppress: Optional[set]) -> None:
        if key is NULL_KEY:
            # NULL components never match; residuals are impossible here.
            writers["left_null"].write_row(row)
            return
        sf = writers["left"][self._part_index(key)]
        if suppress:
            sf.write_record(residual_to_record(
                row, [(self.right_refs[0][0], rid) for rid in suppress]))
        else:
            sf.write_record(row_to_record(row))

    def _finish_switch_and_run(self, writers: dict, reason: str,
                               residual: Optional[tuple]) -> Iterator[Row]:
        for group in ("build", "left"):
            for sf in writers[group]:
                sf.finalize_write()
        writers["build_null"].finalize_write()
        writers["left_null"].finalize_write()
        event = {
            "join_id": self.join_id,
            "reason": reason,
            "phase_at_switch": self._phase,
            "outputs_before_switch": self.stats.outputs,
            "build_rows_seen": self.stats.build_rows,
            "probe_rows_seen": self.stats.probe_rows,
            "build_rid_range": list(self.stats.build_rid_range),
            "probe_rid_range": list(self.stats.probe_rid_range),
            "partitions": self.stats.partitions,
            "residual": None if residual is None else {
                "left_rid": self._rid(residual[0], self.left_refs[0][0]),
                "suppressed_right_rids": sorted(residual[1]),
            },
            "spill_bytes_written_so_far": self.spill.stats.bytes_written,
        }
        self.stats.switch_events.append(event)
        self.stats.mode = "disk"
        self._phase = "disk"
        top = sorted(self._disk_group_count.items(), key=lambda kv: -kv[1])[:TOP_HOT_KEYS]
        if top:
            self.stats.hot_keys = [(self._key_label(k), n) for k, n in top]
        yield from self._run_partitions(writers)
        # NULL-key left rows can never match: left join pads them exactly once.
        if self.join_type == "left":
            for lrow, _ in writers["left_null"].read():
                self.stats.left_padded += 1
                yield self._pad(lrow)

    # ---------------------------------------------------- partition runners
    def _run_partitions(self, writers: dict) -> Iterator[Row]:
        for p in range(self.stats.partitions):
            self._check_cancel()
            build_file = writers["build"][p]
            left_file = writers["left"][p]
            # Size both sides by streaming once without retaining rows.
            need = 0
            left_count = 0
            for r, _ in left_file.read():
                left_count += 1
            if left_count == 0:
                continue
            for r, _ in build_file.read():
                need += r.size + HT_ENTRY_OVERHEAD
            share = max(1, self.budget.available())
            if need <= share:
                # Build side fits: one hash table, left streamed in batches.
                build_rows = [r for r, _ in build_file.read()]
                yield from self._run_partition_hash(build_rows, left_file)
            else:
                # Oversized (e.g. a single hot key): external sort-merge so
                # neither side is materialised and there are no quadratic
                # re-scans.
                self.stats.merge_partitions.append(p)
                yield from self._run_partition_merge(build_file, left_file)

    def _run_partition_hash(self, build_rows: List[Row],
                            left_file) -> Iterator[Row]:
        ht: Dict[Any, List[Row]] = {}
        reserved = 0
        try:
            for r in build_rows:
                cost = r.size + HT_ENTRY_OVERHEAD
                self._reserve(cost)
                reserved += cost
                ht.setdefault(self._rkey(r), []).append(r)
            # Left side streamed from spill: never whole-partition in memory.
            for lrow, suppress in left_file.read():
                self._check_cancel()
                key = self._lkey(lrow)
                suppress_rids = self._suppress_set(suppress)
                matched_here = bool(suppress_rids)
                if key in ht:
                    if self.join_type == "semi":
                        for rrow in ht[key]:
                            rrid = self._rid(rrow, self.right_refs[0][0])
                            if rrid in suppress_rids:
                                continue
                            self._note_semi(lrow, rrow)
                            matched_here = True
                        if matched_here:
                            self.stats.outputs += 1
                            yield lrow
                    else:
                        for rrow in ht[key]:
                            rrid = self._rid(rrow, self.right_refs[0][0])
                            if rrid in suppress_rids:
                                continue
                            self.stats.outputs += 1
                            yield self._combine(lrow, rrow)
                            matched_here = True
                if not matched_here and self.join_type == "left":
                    self.stats.left_padded += 1
                    yield self._pad(lrow)
        finally:
            self._release(reserved)

    def _run_partition_merge(self, build_file, left_file) -> Iterator[Row]:
        """External sort-merge join for an oversized partition.

        Both sides are externally sorted on the tagged join key (bounded
        chunks + heap merge) and merged with proper key alignment: when the
        left key is smaller the left group has no right match (pad for LEFT,
        drop for INNER/SEMI); when the right key is smaller the right group
        is skipped; equal keys are joined group by group.

        Equal-key groups are processed in budget-bounded blocks.  A right
        key group larger than ``rcap`` is re-streamed from the sorted right
        spill file per left block (key-local, never a full-table scan).
        Residual suppression travels per left rid; a LEFT row is padded
        exactly once after its whole right group is exhausted; SEMI emits
        each left row at most once.
        """
        from .types import tag_key
        lkey_fn = lambda r: tag_key(self._lkey(r))  # noqa: E731
        rkey_fn = lambda r: tag_key(self._rkey(r))  # noqa: E731
        mem = max(1, self.budget.limit_bytes // 8)
        rcap = max(1, self.budget.limit_bytes // 4)

        suppress_by_lrid: Dict[int, set] = {}

        def left_rows():
            for lrow, sup in left_file.read():
                if sup:
                    suppress_by_lrid[
                        self._rid(lrow, self.left_refs[0][0])
                    ] = self._suppress_set(sup)
                yield lrow

        left_sorted = self.spill.external_sort(
            left_rows(), lkey_fn, self._owner, mem)
        right_sorted = self.spill.external_sort(
            (r for r, _ in build_file.read()), rkey_fn, self._owner, mem)
        li = iter(row for row, _ in left_sorted.read())
        ri = iter(row for row, _ in right_sorted.read())
        try:
            lcur = next(li)
        except StopIteration:
            return
        rcur = next(ri, None)
        matched_rids: set = set()

        def right_rows_for_key(key):
            """Re-stream rows of one key from the sorted right spill file."""
            for row, _ in right_sorted.read():
                rk = rkey_fn(row)
                if rk == key:
                    yield row
                elif rk > key:
                    break

        while lcur is not None:
            self._check_cancel()
            lkey = lkey_fn(lcur)

            # Gather the whole equal-key LEFT group, then split into blocks.
            group_rows: List[Row] = []
            while lcur is not None and lkey_fn(lcur) == lkey:
                group_rows.append(lcur)
                lcur = next(li, None)

            # Align the right cursor to this left key.
            while rcur is not None and rkey_fn(rcur) < lkey:
                rcur = next(ri, None)

            if rcur is None or rkey_fn(rcur) > lkey:
                # No right rows for this key: LEFT pads, INNER/SEMI drop.
                if self.join_type == "left":
                    for lrow in group_rows:
                        lrid = self._rid(lrow, self.left_refs[0][0])
                        if (lrow.rids not in matched_rids
                                and lrid not in suppress_by_lrid):
                            self.stats.left_padded += 1
                            yield self._pad(lrow)
                continue

            # Right group matches this key. Buffer up to rcap; detect oversize.
            right_group: List[Row] = []
            rbytes = 0
            oversize = False
            while rcur is not None and rkey_fn(rcur) == lkey:
                right_group.append(rcur)
                rbytes += rcur.size
                rcur = next(ri, None)
                if rbytes >= rcap and rcur is not None and rkey_fn(rcur) == lkey:
                    oversize = True
                    break
            if oversize:
                while rcur is not None and rkey_fn(rcur) == lkey:
                    rcur = next(ri, None)

            # Process the left group in bounded blocks against right data.
            block: List[Row] = []
            bbytes = 0
            for lrow in group_rows + [None]:
                if lrow is not None:
                    block.append(lrow)
                    bbytes += lrow.size
                if lrow is None or bbytes >= mem:
                    if not oversize:
                        yield from self._emit_chunk(
                            block, right_group, suppress_by_lrid,
                            matched_rids)
                    else:
                        for rrow in right_rows_for_key(lkey):
                            yield from self._emit_chunk(
                                block, [rrow], suppress_by_lrid,
                                matched_rids)
                    block = []
                    bbytes = 0

            # LEFT padding only for rows still unmatched after the full group.
            if self.join_type == "left":
                for lrow in group_rows:
                    lrid = self._rid(lrow, self.left_refs[0][0])
                    if (lrow.rids not in matched_rids
                            and lrid not in suppress_by_lrid):
                        self.stats.left_padded += 1
                        yield self._pad(lrow)

    def _emit_chunk(self, left_block, right_chunk, suppress_by_lrid,
                    matched_rids) -> Iterator[Row]:
        for lrow in left_block:
            self._check_cancel()
            lrid = self._rid(lrow, self.left_refs[0][0])
            suppress = suppress_by_lrid.get(lrid, set())
            # Identity of an input bag element is its FULL lineage tuple.
            # After an upstream join, one base rid appears in many distinct
            # input rows (different right rids); SEMI must keep each bag
            # element, so dedup cannot key on the base rid alone.
            identity = lrow.rids
            matched = identity in matched_rids or bool(suppress)
            for rrow in right_chunk:
                rrid = self._rid(rrow, self.right_refs[0][0])
                if rrid in suppress:
                    continue
                if self.join_type == "semi":
                    if not matched:
                        matched = True
                        matched_rids.add(identity)
                        self.stats.outputs += 1
                        yield lrow
                    # Every real (unsuppressed) match is a proof witness,
                    # even though only one left row is emitted.
                    self._note_semi(lrow, rrow)
                    continue
                matched = True
                matched_rids.add(identity)
                self.stats.outputs += 1
                yield self._combine(lrow, rrow)

    # ------------------------------------------------------------- helpers
    def _suppress_set(self, suppress: Optional[list]) -> set:
        if not suppress:
            return set()
        # suppress records are (alias, rid); alias is informational, key on rid
        return {rid for _alias, rid in suppress}

    def _discard_ht(self) -> None:
        if not self._ht:
            self._reserved = 0
            return
        total = 0
        for rows in self._ht.values():
            for r in rows:
                total += r.size + HT_ENTRY_OVERHEAD
        self._ht.clear()
        self._group_count.clear()
        self._release(total)
        # _reserved should now be zero; normalise in case of bookkeeping drift
        if self._reserved < 0:  # pragma: no cover
            self._reserved = 0

    def _record_hot_keys(self) -> None:
        top = sorted(self._group_count.items(), key=lambda kv: -kv[1])[:TOP_HOT_KEYS]
        self.stats.hot_keys = [(self._key_label(k), n) for k, n in top]

    @staticmethod
    def _key_label(key: Any) -> str:
        return repr(list(key)) if isinstance(key, tuple) else str(key)

    @staticmethod
    def _note_rid_range(stats: JoinStats, side: str, rid: int) -> None:
        attr = "build_rid_range" if side == "build" else "probe_rid_range"
        lo, hi = getattr(stats, attr)
        lo = rid if lo is None else min(lo, rid)
        hi = rid if hi is None else max(hi, rid)
        setattr(stats, attr, (lo, hi))
