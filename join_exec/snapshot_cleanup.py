"""Pre-delivery quarantine cleanup for offline relation snapshots.

Given the relation schemas, one JSONL file per relation, a quarantine list of
``(relation, row_id)`` pairs and a set of *required reference* rules, this
module produces a delivery directory holding the maximal subset of the input
rows in which no retained row has an unmet required reference.

Contract recap
--------------
A required reference rule names a source relation and a target relation and
pairs source columns with target columns by position (single-column or
composite keys; paired columns must share a type).  A retained source row must
find, for every rule that applies to it, at least one retained target row with
an equal key.  Equality follows the engine's join semantics: any NULL key
component can never match (NULL does not match NULL).  Target keys are not
unique: removing one target row only invalidates referrers when *no* other
retained target row carries the same key, and equal rows are never merged.

Rules may be self-referential and may form cycles.  The delivered set is the
greatest fixpoint of the contract, not just any orphan-free set:

* two accounts that require each other, with no other missing support, are
  both retained;
* if one of them additionally requires a quarantined subject with no
  substitute, both are removed;
* a row with no outgoing required references is never removed merely because
  other rows point at it.

The result does not depend on input row order, file load order or rule order.

Staging strategy (SQLite, constant-bounded Python memory)
---------------------------------------------------------
Whole relations, the full pending set and the whole referrer set of one hot
key are never materialised in Python.  Per relation ``r`` the staging DB
holds one row table (the original columns plus a status column) with one
index per rule whose source is ``r``; per rule ``u`` it holds one compact
*target-key support table* ``(key columns, alive_count)``.  No row pair is
ever stored: a many-to-many fan-out of M referrers and N same-key candidates
occupies O(M + N) rows, never O(M*N).

Elimination propagates along lost keys only.  A durable work queue holds the
row_ids that just lost support; each dequeued row decrements the support
counts of exactly the target keys it carries, and only when a key's count
reaches zero are the referrers on that key removed via a single indexed
``UPDATE``.  Unrelated keys and rows are never rescanned, so propagation does
not degenerate into per-round full relation scans.

Failure semantics
-----------------
All staging and output files live inside a temporary working directory
created next to the (not yet existing) delivery directory.  The delivery
directory itself is created last by a single atomic rename; on any error the
working directory is removed and the delivery directory never appears, so a
downstream consumer can never observe an incomplete delivery.  Input files
are opened read-only and never modified.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import JoinExecError, SchemaError
from .scan import Scan
from .schema_catalog import SchemaCatalog
from .types import Schema

# ----------------------------------------------------------------- errors


class CleanupError(JoinExecError):
    """Base class for snapshot cleanup contract/setup errors."""


class QuarantineError(CleanupError):
    """A quarantine entry names an unknown relation/row or is malformed."""


class RuleError(CleanupError):
    """A required-reference rule is malformed or violates the schema."""


class DeliveryError(CleanupError):
    """The delivery location is unusable or a delivery write failed."""


# ------------------------------------------------------------------ spec


@dataclass(frozen=True)
class ReferenceRule:
    """One validated required-reference rule.

    ``columns`` holds the positional column pairs ``(source_col, target_col)``;
    paired columns have identical declared types.
    """

    rule_id: str
    source_relation: str
    target_relation: str
    columns: Tuple[Tuple[str, str], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "source_relation": self.source_relation,
            "target_relation": self.target_relation,
            "columns": [list(pair) for pair in self.columns],
        }

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any], index: int,
                  catalog: SchemaCatalog) -> "ReferenceRule":
        if not isinstance(spec, Mapping):
            raise RuleError(f"rule #{index} must be an object")
        rule_id = spec.get("rule_id")
        if rule_id is None:
            rule_id = f"rule_{index}"
        if not isinstance(rule_id, str) or not rule_id:
            raise RuleError(f"rule #{index}: rule_id must be a non-empty string")
        src = spec.get("source_relation")
        tgt = spec.get("target_relation")
        if not isinstance(src, str) or not isinstance(tgt, str):
            raise RuleError(
                f"rule {rule_id!r}: source_relation and target_relation "
                "must be strings")
        try:
            src_schema = catalog.get(src)
            tgt_schema = catalog.get(tgt)
        except SchemaError as exc:
            raise RuleError(str(exc)) from exc
        pairs = spec.get("columns")
        if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)) \
                or not pairs:
            raise RuleError(
                f"rule {rule_id!r}: columns must be a non-empty list of "
                "[source_column, target_column] pairs")
        resolved: List[Tuple[str, str]] = []
        seen_src: set = set()
        seen_tgt: set = set()

        def column_type(schema: Schema, col: str) -> str:
            # row_id is the implicit int64 primary key of every relation and
            # is a legal rule column (e.g. child.parent_id = parent.row_id).
            if col == "row_id":
                return "int64"
            return schema.column(col).dtype

        for j, pair in enumerate(pairs):
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) \
                    or len(pair) != 2:
                raise RuleError(
                    f"rule {rule_id!r}: column pair #{j} must be a "
                    "[source_column, target_column] pair")
            s_col, t_col = pair
            if not isinstance(s_col, str) or not isinstance(t_col, str):
                raise RuleError(
                    f"rule {rule_id!r}: column names must be strings")
            try:
                s_type = column_type(src_schema, s_col)
                t_type = column_type(tgt_schema, t_col)
            except SchemaError as exc:
                raise RuleError(str(exc)) from exc
            if s_type != t_type:
                raise RuleError(
                    f"rule {rule_id!r}: paired columns {src}.{s_col} "
                    f"({s_type}) and {tgt}.{t_col} ({t_type}) have "
                    "different types")
            if s_col in seen_src or t_col in seen_tgt:
                raise RuleError(
                    f"rule {rule_id!r}: a column may appear only once per rule")
            seen_src.add(s_col)
            seen_tgt.add(t_col)
            resolved.append((s_col, t_col))
        return cls(rule_id=rule_id, source_relation=src,
                   target_relation=tgt, columns=tuple(resolved))


def rules_from_json(spec: Any, catalog: SchemaCatalog
                    ) -> Tuple[ReferenceRule, ...]:
    if not isinstance(spec, Sequence) or isinstance(spec, (str, bytes)):
        raise RuleError("rules must be a list of rule objects")
    rules: List[ReferenceRule] = []
    seen_ids: set = set()
    for i, raw in enumerate(spec):
        rule = ReferenceRule.from_dict(raw, i, catalog)
        if rule.rule_id in seen_ids:
            raise RuleError(f"duplicate rule_id {rule.rule_id!r}")
        seen_ids.add(rule.rule_id)
        rules.append(rule)
    return tuple(rules)


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# ------------------------------------------------------------- the cleanup


class SnapshotCleanup:
    """Builds one maximal, reference-closed snapshot delivery.

    Use :func:`clean_snapshot` for the one-call entry point.  The class is
    exposed so tests can inject staging faults and inspect the locality
    evidence (:meth:`access_trace`, :meth:`explain_local_lookups`).
    """

    ALIVE = 0
    DEAD = 1
    #: number of work rows pulled per local propagation drain step
    WORK_BATCH = 2048
    #: input rows streamed into SQLite per executemany batch.  Bounded so a
    #: huge relation is never buffered whole in Python, but large enough to
    #: amortize the executemany round-trip on bulk staging.
    LOAD_BATCH = 4096

    def __init__(self, catalog: SchemaCatalog,
                 input_paths: Mapping[str, str],
                 rules: Sequence[ReferenceRule],
                 quarantine: Sequence[Tuple[str, int]],
                 delivery_dir: str) -> None:
        self.catalog = catalog
        self.input_paths = dict(input_paths)
        self.rules = tuple(rules)
        self.quarantine = tuple(quarantine)
        self.delivery_dir = os.path.abspath(delivery_dir)
        self._tmp_root: Optional[str] = None
        self._db_path: Optional[str] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._work_out_dir: Optional[str] = None
        # Per-relation column name -> positional index in the schema.
        self._col_index: Dict[str, Dict[str, int]] = {
            s.relation: {c.name: i for i, c in enumerate(s.columns)}
            for s in catalog.all()
        }
        # Rules grouped by source / target relation (declaration order kept).
        self._rules_by_source: Dict[str, List[ReferenceRule]] = {
            s.relation: [] for s in catalog.all()}
        self._rules_by_target: Dict[str, List[ReferenceRule]] = {
            s.relation: [] for s in catalog.all()}
        for rule in self.rules:
            self._rules_by_source[rule.source_relation].append(rule)
            self._rules_by_target[rule.target_relation].append(rule)
        # Stable, collision-free staging identifiers: catalog declaration
        # order for relations, rule list position for rules.  Never derive SQL
        # identifiers from user-supplied names.
        self._rel_tokens: Dict[str, str] = {
            s.relation: f"r{i}" for i, s in enumerate(catalog.all())}
        self._rule_tokens: Dict[str, str] = {
            rule.rule_id: f"u{i}" for i, rule in enumerate(self.rules)}
        # Filled in after a run.
        self._kept_counts: Dict[str, int] = {}
        self._total_counts: Dict[str, int] = {}
        # Locality events buffered between phase-boundary flushes.  Keyed by
        # (event, rule, relation, key) with an occurrence count so a long
        # elimination chain aggregates instead of inserting a row per hop.
        self._trace_buffer: Dict[Tuple[Any, ...], int] = {}

    # ------------------------------------------------------------ tables
    def _row_table(self, rel: str) -> str:
        return f"rows_{self._rel_tokens[rel]}"

    def _support_table(self, rule: ReferenceRule) -> str:
        return f"support_{self._rule_tokens[rule.rule_id]}"

    def _rel_token(self, rel: str) -> str:
        return self._rel_tokens[rel]

    def _key_columns(self, rule: ReferenceRule, side: str) -> List[str]:
        return [pair[0 if side == "source" else 1] for pair in rule.columns]

    # ------------------------------------------------------------- run
    def run(self, inject_fault: Optional[str] = None) -> Dict[str, Any]:
        """Execute the full cleanup and publish the delivery directory.

        ``inject_fault`` is test-only.  ``"after_first_staged_table"`` fails
        once the first relation has been fully staged; ``"after_first_output"``
        fails after the first delivery table file has been written.  In both
        cases the delivery directory must not exist afterwards.
        """
        if inject_fault not in (
                None, "after_first_staged_table", "after_first_output"):
            raise ValueError(f"unknown fault {inject_fault!r}")
        self._validate_setup()
        parent = os.path.dirname(self.delivery_dir) or "."
        os.makedirs(parent, exist_ok=True)
        self._tmp_root = tempfile.mkdtemp(
            prefix=".snapshot-clean-", dir=parent)
        try:
            self._db_path = os.path.join(self._tmp_root, "staging.db")
            self._conn = self._connect(self._db_path)
            self._create_staging()
            self._stage_relations(inject_fault)
            self._build_source_indexes()
            self._flush_trace()
            self._populate_support()
            self._mark_quarantine()
            self._flush_trace()
            self._find_initial_gaps()
            self._flush_trace()
            self._propagate()
            self._flush_trace()
            self._work_out_dir = os.path.join(self._tmp_root, "delivery")
            os.makedirs(self._work_out_dir)
            self._write_delivery(inject_fault)
            self._flush_trace()
            # Single atomic publication point.
            os.rename(self._work_out_dir, self.delivery_dir)
        except BaseException:
            self._close_conn()
            shutil.rmtree(self._tmp_root, ignore_errors=True)
            # Defensive: a delivery dir must never survive a failure.
            if os.path.exists(self.delivery_dir):
                shutil.rmtree(self.delivery_dir, ignore_errors=True)
            raise
        self._close_conn()
        shutil.rmtree(self._tmp_root, ignore_errors=True)
        self._tmp_root = None
        return self.result_summary()

    def run_keeping_staging(self) -> "SnapshotCleanup":
        """Run every phase but leave staging open (tests/evidence only).

        The delivery directory is still published atomically; callers must
        close the instance with :meth:`discard_staging`, which removes the
        temporary working directory.  Inputs are never modified.
        """
        self._validate_setup()
        parent = os.path.dirname(self.delivery_dir) or "."
        os.makedirs(parent, exist_ok=True)
        self._tmp_root = tempfile.mkdtemp(
            prefix=".snapshot-clean-", dir=parent)
        try:
            self._db_path = os.path.join(self._tmp_root, "staging.db")
            self._conn = self._connect(self._db_path)
            self._create_staging()
            self._stage_relations(None)
            self._build_source_indexes()
            self._flush_trace()
            self._populate_support()
            self._mark_quarantine()
            self._flush_trace()
            self._find_initial_gaps()
            self._flush_trace()
            self._propagate()
            self._flush_trace()
            self._work_out_dir = os.path.join(self._tmp_root, "delivery")
            os.makedirs(self._work_out_dir)
            self._write_delivery(None)
            self._flush_trace()
            os.rename(self._work_out_dir, self.delivery_dir)
        except BaseException:
            self._close_conn()
            shutil.rmtree(self._tmp_root, ignore_errors=True)
            if os.path.exists(self.delivery_dir):
                shutil.rmtree(self.delivery_dir, ignore_errors=True)
            raise
        return self

    def discard_staging(self) -> None:
        """Close the staging connection and remove the temp working dir."""
        self._close_conn()
        if self._tmp_root is not None:
            shutil.rmtree(self._tmp_root, ignore_errors=True)
            self._tmp_root = None

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    @staticmethod
    def _connect(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, isolation_level=None)
        # The staging database lives inside the throwaway working directory;
        # the atomic directory rename is the durability boundary, so staging
        # uses a memory journal and NORMAL sync to avoid fsyncing every work
        # queue step.  A crash still leaves only the removed working
        # directory behind — never an observable partial delivery.
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    # ----------------------------------------------------- validation
    def _validate_setup(self) -> None:
        if os.path.exists(self.delivery_dir):
            raise DeliveryError(
                f"delivery directory already exists: {self.delivery_dir}")
        relations = {s.relation for s in self.catalog.all()}
        missing = sorted(relations - set(self.input_paths))
        if missing:
            raise CleanupError(
                f"no input file provided for relation(s): {missing}")
        for rel in relations:
            path = self.input_paths[rel]
            if not os.path.isfile(path):
                raise CleanupError(f"input file not found for {rel}: {path}")
        # Quarantine validation against the catalog; row existence is checked
        # while streaming the input so files are never read twice up front.
        seen_q: set = set()
        for entry in self.quarantine:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise QuarantineError(
                    "quarantine entries must be (relation, row_id) pairs")
            rel, rid = entry
            if not isinstance(rel, str) or rel not in relations:
                raise QuarantineError(
                    f"quarantine entry names unknown relation {rel!r}")
            if isinstance(rid, bool) or not isinstance(rid, int):
                raise QuarantineError(
                    f"quarantine row_id for {rel!r} must be an int64")
            if (rel, rid) in seen_q:
                raise QuarantineError(
                    f"duplicate quarantine entry {rel}#{rid}")
            seen_q.add((rel, rid))

    # ----------------------------------------------------- DDL/staging
    def _create_staging(self) -> None:
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        # Durable work queue: rows that just lost support, keyed so the same
        # row is processed at most once.
        conn.execute(
            "CREATE TABLE work_queue ("
            " rel_token TEXT NOT NULL, row_id INTEGER NOT NULL,"
            " PRIMARY KEY (rel_token, row_id)) WITHOUT ROWID")
        # Locality evidence: localized propagation accesses, aggregated per
        # distinct (event, rule, relation, key) with an occurrence count so a
        # long elimination chain does not insert one trace row per hop.
        conn.execute(
            "CREATE TABLE access_trace ("
            " trace_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " event TEXT NOT NULL, rule_id TEXT,"
            " rel_token TEXT, key_json TEXT, count INTEGER NOT NULL)")
        for schema in self.catalog.all():
            rel = schema.relation
            table = self._row_table(rel)
            col_sql = ", ".join(
                f"{_qident(c.name)} BLOB" for c in schema.columns)
            conn.execute(
                f"CREATE TABLE {table} ("
                f"row_id INTEGER PRIMARY KEY, {col_sql},"
                " status INTEGER NOT NULL DEFAULT 0)")
            # Source-side partial indexes are built by _build_source_indexes
            # only after the bulk staging load, so inserting input rows never
            # pays index maintenance.
        # One compact support table per rule: distinct target key -> number of
        # ALIVE target rows carrying it.  No referrer/candidate pairs exist.
        # The tables are populated by _populate_support only after every
        # relation row has been staged.
        for rule in self.rules:
            tgt_cols = self._key_columns(rule, "target")
            key_sql = ", ".join(_qident(c) for c in tgt_cols)
            support = self._support_table(rule)
            col_def = ", ".join(f"{_qident(c)} BLOB" for c in tgt_cols)
            conn.execute(
                f"CREATE TABLE {support} ({col_def},"
                " alive_count INTEGER NOT NULL,"
                f" PRIMARY KEY ({key_sql})) WITHOUT ROWID")
            # No secondary index on alive_count: maintaining one across every
            # propagation update dominates the cost of a thin elimination
            # chain.  The support table holds one row per distinct target key,
            # so scanning it for the few zeroed keys stays cheap, while the
            # large row tables are always reached by point SEARCH through
            # their source-side partial indexes.
        conn.commit()

    def _build_source_indexes(self) -> None:
        """Build the per-rule partial source indexes after the bulk load.

        One index per outgoing rule over its source-side key columns,
        restricted to alive rows, so referrer removal after a key loses
        support is an index SEARCH.  Building them here (rather than before
        staging) keeps bulk inserts free of index maintenance.
        """
        conn = self._conn
        for schema in self.catalog.all():
            rel = schema.relation
            table = self._row_table(rel)
            for rule in self._rules_by_source[rel]:
                cols = ", ".join(
                    _qident(c) for c in self._key_columns(rule, "source"))
                conn.execute(
                    f"CREATE INDEX idx_{self._rule_tokens[rule.rule_id]} "
                    f"ON {table} ({cols}) WHERE status = 0")

    def _populate_support(self) -> None:
        """Build initial target-key support counts from the staged rows.

        Runs *before* quarantine marking, so every input row contributes
        exactly once here and every later removal (quarantine included)
        decrements exactly once through the propagation path.  NULL-key groups
        are excluded entirely (a NULL component can never match, so such target
        rows support nobody); every non-NULL distinct key gets one row whose
        count is the number of target rows carrying it.
        """
        conn = self._conn
        for rule in self.rules:
            tgt_table = self._row_table(rule.target_relation)
            tgt_cols = self._key_columns(rule, "target")
            key_sql = ", ".join(_qident(c) for c in tgt_cols)
            key_select = ", ".join(_qident(c) for c in tgt_cols)
            not_null = " AND ".join(
                f"{_qident(c)} IS NOT NULL" for c in tgt_cols)
            support = self._support_table(rule)
            conn.execute(
                f"INSERT INTO {support} ({key_sql}, alive_count) "
                f"SELECT {key_select}, COUNT(*) FROM {tgt_table} "
                f"WHERE status = 0 AND {not_null} GROUP BY {key_sql}")

    def _stage_relations(self, inject_fault: Optional[str]) -> None:
        conn = self._conn
        staged_first = False
        for schema in self.catalog.all():
            rel = schema.relation
            table = self._row_table(rel)
            col_sql = ", ".join(_qident(c.name) for c in schema.columns)
            params = ", ".join("?" for _ in schema.columns)
            stmt = (f"INSERT INTO {table}(row_id, {col_sql}) "
                    f"VALUES (?, {params})")
            scan = Scan(schema, self.input_paths[rel], rel)
            batch: List[Tuple[Any, ...]] = []
            total = 0
            for row in scan:
                rid = row.rid_of(rel)
                batch.append((rid, *row.values))
                total += 1
                if len(batch) >= self.LOAD_BATCH:
                    conn.executemany(stmt, batch)
                    batch.clear()
            if batch:
                conn.executemany(stmt, batch)
            self._total_counts[rel] = total
            self._trace("relation_staged", rel_token=self._rel_token(rel),
                        key={"rows": total})
            if not staged_first:
                staged_first = True
                if inject_fault == "after_first_staged_table":
                    raise DeliveryError(
                        "injected fault after first staged relation")

    def _mark_quarantine(self) -> None:
        conn = self._conn
        for rel, rid in self.quarantine:
            table = self._row_table(rel)
            cur = conn.execute(
                f"UPDATE {table} SET status = 1 WHERE row_id = ? AND "
                "status = 0", (rid,))
            if cur.rowcount == 0:
                if not self._row_exists(conn, table, rid):
                    raise QuarantineError(
                        f"quarantine entry names unknown row {rel}#{rid}")
                # Already marked via a duplicate entry — duplicates are
                # rejected up front, so this is defensive.
                continue
            conn.execute(
                "INSERT OR IGNORE INTO work_queue(rel_token, row_id) "
                "VALUES (?, ?)", (self._rel_token(rel), rid))
            self._trace("quarantined", rule_id=None,
                        rel_token=self._rel_token(rel), key={"row_id": rid})

    @staticmethod
    def _row_exists(conn: sqlite3.Connection, table: str, rid: int) -> bool:
        return conn.execute(
            f"SELECT 1 FROM {table} WHERE row_id = ? LIMIT 1",
            (rid,)).fetchone() is not None

    # ---------------------------------------------------- initial gaps
    def _find_initial_gaps(self) -> None:
        """Mark rows whose required support is absent in the staged snapshot.

        Two independent gap kinds, each driven by an indexed set operation:

        * a source key containing a NULL component can never match anything;
        * a non-NULL source key with no support-table row has zero candidates.

        Rows already dead (quarantined) are never reprocessed.
        """
        conn = self._conn
        for rule in self.rules:
            src_table = self._row_table(rule.source_relation)
            support = self._support_table(rule)
            src_cols = self._key_columns(rule, "source")
            tgt_cols = self._key_columns(rule, "target")
            src_token = self._rel_token(rule.source_relation)

            null_pred = " OR ".join(
                f"s.{_qident(c)} IS NULL" for c in src_cols)
            conn.execute(
                "INSERT OR IGNORE INTO work_queue(rel_token, row_id) "
                f"SELECT ?, s.row_id FROM {src_table} s "
                f"WHERE s.status = 0 AND ({null_pred})",
                (src_token,))
            cur = conn.execute(
                f"UPDATE {src_table} AS s SET status = 1 "
                f"WHERE s.status = 0 AND ({null_pred})")
            if cur.rowcount:
                self._trace("null_key_gap", rule_id=rule.rule_id,
                            rel_token=src_token,
                            key={"rows": cur.rowcount})

            # Non-NULL source keys without any support row.  The NOT EXISTS
            # uses the support table's primary-key index; no full cross join.
            join_pred = " AND ".join(
                f"s.{_qident(sc)} = t.{_qident(tc)}"
                for sc, tc in zip(src_cols, tgt_cols))
            not_null_pred = " AND ".join(
                f"s.{_qident(c)} IS NOT NULL" for c in src_cols)
            conn.execute(
                "INSERT OR IGNORE INTO work_queue(rel_token, row_id) "
                f"SELECT ?, s.row_id FROM {src_table} s "
                f"WHERE s.status = 0 AND {not_null_pred} AND NOT EXISTS ("
                f"SELECT 1 FROM {support} t WHERE {join_pred})",
                (src_token,))
            cur = conn.execute(
                f"UPDATE {src_table} AS s SET status = 1 "
                f"WHERE s.status = 0 AND {not_null_pred} AND NOT EXISTS ("
                f"SELECT 1 FROM {support} t WHERE {join_pred})")
            if cur.rowcount:
                self._trace("unsupported_key_gap", rule_id=rule.rule_id,
                            rel_token=src_token,
                            key={"rows": cur.rowcount})

    # ----------------------------------------------------- propagation
    def _propagate(self) -> None:
        """Drain the work queue along lost keys only.

        Each dequeued dead row decrements support counts for the target keys
        it carries (for rules where it is a target).  When a count reaches
        zero, every still-alive referrer on that key is killed in one indexed
        statement and queued.  Rows/keys not touched by a loss are never read:
        there is no per-round rescan of any relation.
        """
        conn = self._conn
        rel_for_token = {token: rel for rel, token in self._rel_tokens.items()}
        while True:
            # Atomically pull and remove one bounded batch in a single
            # statement (DELETE ... RETURNING).  The durable queue guarantees
            # each dead row is handled at most once; INSERT OR IGNORE at every
            # enqueue site keeps it deduplicated.
            batch = conn.execute(
                "DELETE FROM work_queue WHERE (rel_token, row_id) IN ("
                "SELECT rel_token, row_id FROM work_queue LIMIT ?) "
                "RETURNING rel_token, row_id",
                (self.WORK_BATCH,)).fetchall()
            if not batch:
                return
            # Aggregate this batch's dequeued rows per relation.
            per_rel: Dict[str, List[int]] = {}
            for rel_token, rid in batch:
                per_rel.setdefault(rel_token, []).append(rid)
            for rel_token, rids in per_rel.items():
                rel = rel_for_token[rel_token]
                self._propagate_from(rel, rids)

    def _propagation_sql(self, rule: ReferenceRule,
                         n_dead: int = 1) -> List[Tuple[str, str, int]]:
        """Build the localized propagation SQL for one rule.

        Returns ``(name, sql, n_params)`` triples shared by execution and
        :meth:`explain_local_lookups`.  ``n_dead`` sets the number of dead
        target row_id parameters in the decrement statement; the enqueue
        statement takes one source relation token.
        """
        tgt_table = self._row_table(rule.target_relation)
        src_table = self._row_table(rule.source_relation)
        support = self._support_table(rule)
        tgt_quoted = [_qident(c) for c in self._key_columns(rule, "target")]
        src_quoted = [_qident(c) for c in self._key_columns(rule, "source")]
        placeholders = ",".join("?" for _ in range(max(1, n_dead)))
        not_null = " AND ".join(f"t.{q} IS NOT NULL" for q in tgt_quoted)
        group_cols = ", ".join(f"t.{q}" for q in tgt_quoted)
        # Correlated count so the multiplicity of same-key dead rows is
        # preserved (the scalar subquery counts every dead row).  NULL-key
        # rows never held support (NULL never matches).  Both filters are
        # driven by the dead row_id parameter list; the support table is
        # reached through its PRIMARY KEY.
        corr_match = " AND ".join(
            f"t.{q} = {support}.{q}" for q in tgt_quoted)
        dead_filter = f"t.row_id IN ({placeholders}) AND {not_null}"
        decrement = (
            f"UPDATE {support} SET alive_count = alive_count - ("
            f"SELECT COUNT(*) FROM {tgt_table} t "
            f"WHERE {dead_filter} AND {corr_match}) "
            f"WHERE ({', '.join(tgt_quoted)}) IN ("
            f"SELECT {group_cols} FROM {tgt_table} t "
            f"WHERE {dead_filter})")
        n_decrement_params = 2 * max(1, n_dead)
        # Referrers are reached through the source-side partial index as a
        # point SEARCH per zeroed key; the zeroed keys come from the compact
        # support table (one row per distinct key).
        zero_keys = (
            f"SELECT {', '.join(tgt_quoted)} FROM {support} "
            "WHERE alive_count = 0")
        src_key_tuple = ", ".join(src_quoted)
        enqueue = (
            "INSERT OR IGNORE INTO work_queue(rel_token, row_id) "
            f"SELECT ?, r.row_id FROM {src_table} r "
            f"WHERE r.status = 0 AND ({src_key_tuple}) IN ({zero_keys})")
        kill = (
            f"UPDATE {src_table} SET status = 1 WHERE status = 0 "
            f"AND ({src_key_tuple}) IN ({zero_keys})")
        # The dead row_id list is bound twice (correlated count + key filter).
        return [("support_decrement", decrement, n_decrement_params),
                ("referrer_enqueue", enqueue, 1),
                ("referrers_killed", kill, 0)]

    def _propagate_from(self, rel: str, rids: Sequence[int]) -> None:
        conn = self._conn
        for rule in self._rules_by_target[rel]:
            sql_by_name = {name: sql for name, sql, _ in
                           self._propagation_sql(rule, n_dead=len(rids))}

            # 1. Local decrement by the exact multiplicity of dead rows per
            #    key.  The statement binds the dead row_id list twice (once
            #    for the correlated count, once for the support-key filter);
            #    NULL-key rows never held support (NULL never matches).
            conn.execute(sql_by_name["support_decrement"],
                         tuple(rids) + tuple(rids))
            self._trace("support_decrement", rule_id=rule.rule_id,
                        rel_token=self._rel_token(rel),
                        key={"dead_rows": len(rids)})

            # 2. Keys whose last candidate just vanished: enqueue and kill
            #    their referrers via point SEARCHes on the source index.
            conn.execute(
                sql_by_name["referrer_enqueue"],
                (self._rel_token(rule.source_relation),))
            cur = conn.execute(sql_by_name["referrers_killed"])
            if cur.rowcount:
                self._trace("referrers_killed", rule_id=rule.rule_id,
                            rel_token=self._rel_token(
                                rule.source_relation),
                            key={"rows": cur.rowcount})

            # Zeroed support rows are left in place rather than deleted each
            # hop: alive_count only ever decreases, so a zero stays zero and a
            # key can never regain support.  The support table remains bounded
            # by the number of distinct target keys (one row per key), never
            # by referrer/candidate pairs, and avoiding a DELETE per hop keeps
            # a thin elimination chain at three localized statements per hop.

    # -------------------------------------------------------- delivery
    def _write_delivery(self, inject_fault: Optional[str]) -> None:
        conn = self._conn
        first_written = False
        for schema in self.catalog.all():
            rel = schema.relation
            table = self._row_table(rel)
            out_path = os.path.join(self._work_out_dir, rel + ".jsonl")
            col_names = [c.name for c in schema.columns]
            kept = 0
            # Stream retained rows in original row_id/storage order; the
            # SQLite window over the primary key is traversed, not buffered.
            cur = conn.execute(
                f"SELECT row_id, {', '.join(_qident(c) for c in col_names)} "
                f"FROM {table} WHERE status = 0 ORDER BY row_id")
            with open(out_path, "w", encoding="utf-8") as fh:
                while True:
                    rows = cur.fetchmany(self.LOAD_BATCH)
                    if not rows:
                        break
                    for record in rows:
                        obj = {"row_id": record[0]}
                        for name, value in zip(col_names, record[1:]):
                            obj[name] = value
                        fh.write(json.dumps(
                            obj, ensure_ascii=False,
                            separators=(",", ":")) + "\n")
                        kept += 1
            self._kept_counts[rel] = kept
            self._trace("delivery_written", rel_token=self._rel_token(rel),
                        key={"rows": kept})
            if not first_written:
                first_written = True
                if inject_fault == "after_first_output":
                    raise DeliveryError(
                        "injected fault after first delivery table")

    # -------------------------------------------------------- evidence
    def _trace(self, event: str, rule_id: Optional[str] = None,
               rel_token: Optional[str] = None,
               key: Optional[Mapping[str, Any]] = None) -> None:
        # Aggregated in a Python dict and flushed in one executemany at phase
        # boundaries: propagation may emit thousands of localized events and
        # must not pay a SQLite round-trip (or one table row) per event.
        signature = (event, rule_id, rel_token,
                     json.dumps(key, sort_keys=True) if key is not None
                     else None)
        self._trace_buffer[signature] = \
            self._trace_buffer.get(signature, 0) + 1

    def _flush_trace(self) -> None:
        if self._trace_buffer:
            self._conn.executemany(
                "INSERT INTO access_trace(event, rule_id, rel_token,"
                " key_json, count) VALUES (?, ?, ?, ?, ?)",
                ((*signature, n) for signature, n in
                 self._trace_buffer.items()))
            self._trace_buffer.clear()

    def access_trace(self) -> List[Dict[str, Any]]:
        """Durable, count-aggregated locality evidence from the last run.

        Only available while the staging database exists (i.e. inside tests
        driving the phases directly); the normal :meth:`run` removes staging
        after the atomic publication.
        """
        if self._conn is None:
            raise CleanupError(
                "access_trace is only available while staging is open")
        rows = self._conn.execute(
            "SELECT event, rule_id, rel_token, key_json, count "
            "FROM access_trace ORDER BY trace_id").fetchall()
        return [{"event": e, "rule_id": r, "rel_token": t,
                 "key_json": k, "count": n}
                for e, r, t, k, n in rows]

    def explain_local_lookups(self) -> List[Dict[str, Any]]:
        """EXPLAIN QUERY PLAN for the localized propagation statements.

        Shows index ``SEARCH`` plans (no full table scans) for support
        decrements and referrer removal.  The SQL is produced by the same
        builder used during propagation, so the plans describe real work.
        """
        if self._conn is None:
            raise CleanupError(
                "explain_local_lookups is only available while staging is open")
        plans: List[Dict[str, Any]] = []
        for rule in self.rules:
            statements = self._propagation_sql(rule)
            for name, sql, n_params in statements:
                lines = [r[3] for r in self._conn.execute(
                    "EXPLAIN QUERY PLAN " + sql,
                    tuple(0 for _ in range(n_params))).fetchall()]
                plans.append({"rule_id": rule.rule_id, "access": name,
                              "sql": sql, "plan": lines})
        return plans

    def staging_cardinality(self) -> Dict[str, int]:
        """Row counts of every staging table (evidence against pair blow-up).

        The sum of all support-table sizes is bounded by the number of
        distinct target keys; no table can hold referrer x candidate pairs.
        """
        if self._conn is None:
            raise CleanupError(
                "staging_cardinality is only available while staging is open")
        out: Dict[str, int] = {}
        for schema in self.catalog.all():
            table = self._row_table(schema.relation)
            out[table] = self._conn.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for rule in self.rules:
            support = self._support_table(rule)
            out[support] = self._conn.execute(
                f"SELECT COUNT(*) FROM {support}").fetchone()[0]
        out["work_queue"] = self._conn.execute(
            "SELECT COUNT(*) FROM work_queue").fetchone()[0]
        return out

    # --------------------------------------------------------- result
    def result_summary(self) -> Dict[str, Any]:
        return {
            "delivery_dir": self.delivery_dir,
            "output_paths": self.output_paths(),
            "rules": [r.to_dict() for r in self.rules],
            "quarantined": [[rel, rid] for rel, rid in self.quarantine],
            "total_rows": dict(self._total_counts),
            "kept_rows": dict(self._kept_counts),
        }

    def output_paths(self) -> Dict[str, str]:
        """API: output JSONL path for each relation (empty tables included)."""
        return {s.relation: os.path.join(self.delivery_dir,
                                         s.relation + ".jsonl")
                for s in self.catalog.all()}


# ------------------------------------------------------- one-call entry


def clean_snapshot(catalog: SchemaCatalog,
                   input_paths: Mapping[str, str],
                   rules: Sequence[ReferenceRule],
                   quarantine: Sequence[Tuple[str, int]],
                   delivery_dir: str) -> Dict[str, Any]:
    """Build the maximal reference-closed snapshot in ``delivery_dir``.

    Returns a summary whose ``output_paths`` mapping hands each relation's
    delivered JSONL file (always present, even when empty) to downstream
    executors.  Raises a :class:`JoinExecError` subclass on any contract
    violation; on failure the delivery directory does not exist and the input
    files are untouched.
    """
    cleanup = SnapshotCleanup(catalog, input_paths, rules, quarantine,
                              delivery_dir)
    return cleanup.run()
