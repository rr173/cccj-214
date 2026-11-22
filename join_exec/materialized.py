"""Durable, incrementally maintained summary view over a local SQLite state DB.

Supported (and only supported) query shape::

    AGGREGATE  keys=[<cat-alias>.<group-column>]  sum=<fact-alias>.<int64-column>
      SEMI JOIN  on [<fact-alias>.<ek> = <elig-alias>.<ekey>]
        LEFT JOIN  on [<fact-alias>.<fk> = <cat-alias>.<ck>]
          SCAN fact (F)
          SCAN category (B)
        SCAN eligibility (C)

Semantics, identical to the one-shot engine in :mod:`join_exec`:

* the LEFT join contributes one output row per matching category row
  (duplicate category rows, even with equal group values, each contribute),
  or exactly one NULL-padded row when nothing matches; a NULL fact join key
  never matches and therefore also yields the single padded row;
* the SEMI join only decides whether the fact row survives (at most once,
  regardless of how many eligibility rows match); NULL join keys never match;
* groups are the category group column (NULL is a legal group value);
* ``COUNT(*)`` counts the joined rows and ``SUM`` ignores NULL amounts, so a
  group whose amounts are all NULL publishes sum = NULL; a group with no
  surviving joined rows disappears entirely;
* every sum is exact integer arithmetic and a published value outside int64
  rejects the whole batch.

Refresh locality
----------------
A batch correction may touch all three relations and may move join keys.  A
refresh reads only:

1. the directly corrected rows,
2. fact rows whose fact-side LEFT/SEMI key equals a key that a corrected
   category / eligibility row moved away from or into,
3. the category and eligibility rows matched by those facts' current keys
   (indexed point lookups).

No unrelated fact row is ever scanned, no full join is recomputed and the
summary is updated only for groups whose contributions changed.  Every local
access is recorded in the durable ``access_trace`` table and the lookup
statements are exposed through :meth:`MaterializedSummary.explain_local_access`
(``EXPLAIN QUERY PLAN``) so tests can verify locality instead of trusting a
row-count report.

Atomicity
---------
A batch runs in one SQLite transaction.  Bad rows, deleting a missing
``row_id``, an int64 overflow or an injected write/publish fault all trigger
``ROLLBACK``: source rows, maintenance state, the summary and the version
either all stay on the old version or all advance together.  Reopening the
database resumes from the last committed version.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import BadRowError, JoinExecError, SchemaError
from .scan import Scan
from .schema_catalog import SchemaCatalog
from .types import INT64_MAX, INT64_MIN, Schema, check_int64

# ---------------------------------------------------------------- errors


class MaterializedError(JoinExecError):
    """Base class for materialized summary errors."""


class CorrectionError(MaterializedError):
    """A correction batch is malformed or violates the declared contract."""


class OverflowRejected(MaterializedError):
    """A published sum/count would leave the int64 range; batch rejected."""


# ---------------------------------------------------------------- spec


@dataclass(frozen=True)
class SummarySpec:
    """The fully resolved, immutable description of the maintained view."""

    fact_rel: str
    fact_alias: str
    cat_rel: str
    cat_alias: str
    elig_rel: str
    elig_alias: str
    fact_fk: str          # fact column joined to the category table
    cat_key: str          # category join-key column
    fact_ek: str          # fact column joined to the eligibility table
    elig_key: str         # eligibility join-key column
    group_col: str        # category column supplying the group value
    sum_col: str          # fact int64 column summed by the aggregate

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "SummarySpec":
        return cls(**json.loads(text))


def _as_scan(node: dict) -> dict:
    if node.get("node") != "scan":
        raise SchemaError(
            "materialized summary only accepts the fixed shape "
            "aggregate(semi(left(scan fact, scan cat), scan elig))")
    return node


def spec_from_tree(tree: dict, catalog: SchemaCatalog) -> SummarySpec:
    """Resolve and validate the single supported query-tree shape."""
    if tree.get("node") != "aggregate":
        raise SchemaError("materialized summary root must be an aggregate node")
    keys = tree.get("keys", [])
    sum_ref = tree.get("sum")
    if len(keys) != 1 or not sum_ref:
        raise SchemaError(
            "materialized summary requires exactly one group key and one sum")

    semi = tree.get("input")
    if semi.get("node") != "join" or semi.get("type") != "semi":
        raise SchemaError("aggregate input must be a SEMI join")
    left_join = semi.get("left")
    if left_join.get("node") != "join" or left_join.get("type") != "left":
        raise SchemaError("SEMI left input must be a LEFT join")

    fact_node = _as_scan(left_join.get("left"))
    cat_node = _as_scan(left_join.get("right"))
    elig_node = _as_scan(semi.get("right"))

    if len(left_join.get("on", [])) != 1 or len(semi.get("on", [])) != 1:
        raise SchemaError("both joins must be single-column equi-joins")
    (lf_ref, rc_ref), = left_join["on"]
    (lf2_ref, re_ref), = semi["on"]

    fact_alias = fact_node.get("alias", fact_node["relation"])
    cat_alias = cat_node.get("alias", cat_node["relation"])
    elig_alias = elig_node.get("alias", elig_node["relation"])

    def split(ref: str) -> Tuple[str, str]:
        try:
            alias, col = ref.split(".", 1)
        except ValueError as exc:  # pragma: no cover - defensive
            raise SchemaError(f"bad field reference {ref!r}") from exc
        return alias, col

    lf_alias, fact_fk = split(lf_ref)
    rc_alias, cat_key = split(rc_ref)
    lf2_alias, fact_ek = split(lf2_ref)
    re_alias, elig_key = split(re_ref)
    g_alias, group_col = split(keys[0])
    s_alias, sum_col = split(sum_ref)

    if lf_alias != fact_alias or lf2_alias != fact_alias \
            or s_alias != fact_alias:
        raise SchemaError("fact-side join/sum references must name the fact scan")
    if rc_alias != cat_alias or g_alias != cat_alias:
        raise SchemaError("category references must name the category scan")
    if re_alias != elig_alias:
        raise SchemaError("eligibility reference must name the eligibility scan")

    fact_schema = catalog.get(fact_node["relation"])
    cat_schema = catalog.get(cat_node["relation"])
    elig_schema = catalog.get(elig_node["relation"])
    for col in (fact_fk, fact_ek, sum_col):
        fact_schema.column(col)
    cat_schema.column(cat_key)
    cat_schema.column(group_col)
    elig_schema.column(elig_key)
    if fact_schema.column(sum_col).dtype != "int64":
        raise SchemaError("sum column must be int64")

    return SummarySpec(
        fact_rel=fact_node["relation"], fact_alias=fact_alias,
        cat_rel=cat_node["relation"], cat_alias=cat_alias,
        elig_rel=elig_node["relation"], elig_alias=elig_alias,
        fact_fk=fact_fk, cat_key=cat_key,
        fact_ek=fact_ek, elig_key=elig_key,
        group_col=group_col, sum_col=sum_col)


# --------------------------------------------------------------- decoding


def decode_object(schema: Schema, obj: Any) -> Tuple[int, Tuple[Any, ...]]:
    """Validate one JSONL row object against ``schema`` (same rules as Scan)."""
    if not isinstance(obj, dict) or "row_id" not in obj:
        raise CorrectionError("correction row must be an object with row_id")
    rid = check_int64(obj["row_id"])
    values: List[Any] = []
    for col in schema.columns:
        if col.name not in obj:
            raise CorrectionError(
                f"relation {schema.relation!r} row {rid}: missing column "
                f"{col.name!r}")
        v = obj[col.name]
        if v is None:
            values.append(None)
        elif col.dtype == "int64":
            try:
                values.append(check_int64(v))
            except BadRowError as exc:
                raise CorrectionError(str(exc)) from exc
        else:
            if not isinstance(v, str):
                raise CorrectionError(
                    f"relation {schema.relation!r} row {rid}: column "
                    f"{col.name!r} expects text")
            values.append(v)
    return rid, tuple(values)


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _key_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ------------------------------------------------------------- the view


class MaterializedSummary:
    """A persistent summary view refreshed by row-level correction batches.

    Lifecycle: :meth:`initialize` once on the full inputs, then :meth:`open`
    across process restarts.  One entry point accepts corrections
    (:meth:`apply_corrections`) and one entry point reads the current full
    summary and its version (:meth:`read_summary` / :meth:`version`).
    """

    FACT_TABLE = "fact_rows"
    CAT_TABLE = "cat_rows"
    ELIG_TABLE = "elig_rows"
    SUMMARY_TABLE = "summary_rows"
    TRACE_TABLE = "access_trace"
    META_TABLE = "meta"

    # --------------------------------------------------------- lifecycle
    def __init__(self, db_path: str, conn: sqlite3.Connection,
                 catalog: SchemaCatalog, spec: SummarySpec,
                 batch_count: int) -> None:
        self.db_path = db_path
        self._conn = conn
        self.catalog = catalog
        self.spec = spec
        self._batch_count = batch_count
        self._col_index: Dict[str, Dict[str, int]] = {}
        for rel in (spec.fact_rel, spec.cat_rel, spec.elig_rel):
            schema = catalog.get(rel)
            self._col_index[rel] = {
                c.name: i for i, c in enumerate(schema.columns)}
        # Per-refresh caches of indexed point lookups (reset every batch).
        self._cat_cache: Dict[Any, Optional[List[Any]]] = {}
        self._elig_cache: Dict[Any, bool] = {}

    @classmethod
    def initialize(cls, db_path: str, tree: dict, catalog: SchemaCatalog,
                   input_paths: Optional[Mapping[str, str]] = None
                   ) -> "MaterializedSummary":
        """Create the state database from the full initial JSONL inputs."""
        spec = spec_from_tree(tree, catalog)
        if os.path.exists(db_path):
            raise MaterializedError(f"state database already exists: {db_path}")
        directory = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(directory, exist_ok=True)

        scan_files: Dict[str, str] = {}
        for node in tree_walk_scans(tree):
            scan_files[node["relation"]] = node["file"]
        if input_paths:
            scan_files.update(dict(input_paths))
        for rel in (spec.fact_rel, spec.cat_rel, spec.elig_rel):
            if rel not in scan_files:
                raise SchemaError(f"no input file for relation {rel!r}")

        conn = cls._connect(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            view = cls(db_path, conn, catalog, spec, batch_count=0)
            view._create_schema()
            view._load_initial(scan_files)
            conn.execute(
                f"INSERT INTO {cls.META_TABLE}(key, value) VALUES (?, ?)",
                ("version", "1"))
            conn.execute(
                f"UPDATE {cls.TRACE_TABLE} SET version_after = 1 "
                "WHERE version_after IS NULL")
            conn.commit()
            return view
        except BaseException:
            conn.rollback()
            conn.close()
            if os.path.exists(db_path):
                os.unlink(db_path)
            raise

    @classmethod
    def open(cls, db_path: str) -> "MaterializedSummary":
        """Reopen a previously committed state database."""
        if not os.path.exists(db_path):
            raise MaterializedError(f"state database not found: {db_path}")
        conn = cls._connect(db_path)
        try:
            meta = dict(conn.execute(
                f"SELECT key, value FROM {cls.META_TABLE}").fetchall())
            spec = SummarySpec.from_json(meta["spec"])
            catalog = SchemaCatalog(
                _schema_from_dict(d) for d in json.loads(meta["schemas"]))
            batch_count = int(meta.get("batch_count", "0"))
        finally:
            conn.close()
        conn = cls._connect(db_path)
        return cls(db_path, conn, catalog, spec, batch_count)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MaterializedSummary":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def _connect(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, isolation_level=None)
        # Readers (separate connections) must never see an open transaction.
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    # ------------------------------------------------------------- DDL
    def _table_name(self, rel: str) -> str:
        return {
            self.spec.fact_rel: self.FACT_TABLE,
            self.spec.cat_rel: self.CAT_TABLE,
            self.spec.elig_rel: self.ELIG_TABLE,
        }[rel]

    def _create_schema(self) -> None:
        spec = self.spec
        ddl: List[str] = [
            f"CREATE TABLE {self.META_TABLE} ("
            " key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            f"CREATE TABLE {self.TRACE_TABLE} ("
            " trace_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " batch_seq INTEGER NOT NULL,"
            " version_after INTEGER,"
            " kind TEXT NOT NULL,"
            " access TEXT NOT NULL,"
            " table_name TEXT NOT NULL,"
            " row_id INTEGER,"
            " key_value TEXT)",
        ]
        for rel, table in (
                (spec.fact_rel, self.FACT_TABLE),
                (spec.cat_rel, self.CAT_TABLE),
                (spec.elig_rel, self.ELIG_TABLE)):
            schema = self.catalog.get(rel)
            cols = ", ".join(
                f"{_qident(c.name)} BLOB" for c in schema.columns)
            ddl.append(
                f"CREATE TABLE {table} ("
                f"row_id INTEGER PRIMARY KEY, {cols})")
        ddl.append(
            f"CREATE INDEX idx_fact_fk ON {self.FACT_TABLE}"
            f"({_qident(spec.fact_fk)})")
        ddl.append(
            f"CREATE INDEX idx_fact_ek ON {self.FACT_TABLE}"
            f"({_qident(spec.fact_ek)})")
        ddl.append(
            f"CREATE INDEX idx_cat_key ON {self.CAT_TABLE}"
            f"({_qident(spec.cat_key)})")
        ddl.append(
            f"CREATE INDEX idx_elig_key ON {self.ELIG_TABLE}"
            f"({_qident(spec.elig_key)})")
        # g_isnull first so the NULL group never collides with a real value.
        ddl.append(
            f"CREATE TABLE {self.SUMMARY_TABLE} ("
            " g_isnull INTEGER NOT NULL, g BLOB NOT NULL,"
            " cnt INTEGER NOT NULL, nonnull_cnt INTEGER NOT NULL,"
            " total INTEGER NOT NULL,"
            " PRIMARY KEY (g_isnull, g)) WITHOUT ROWID")
        for stmt in ddl:
            self._conn.execute(stmt)

        self._conn.execute(
            f"INSERT INTO {self.META_TABLE}(key, value) VALUES (?, ?)",
            ("spec", spec.to_json()))
        self._conn.execute(
            f"INSERT INTO {self.META_TABLE}(key, value) VALUES (?, ?)",
            ("schemas", json.dumps(
                [{"relation": s.relation,
                  "columns": [{"name": c.name, "type": c.dtype}
                              for c in s.columns]}
                 for s in self.catalog.all()],
                sort_keys=True)))
        self._conn.execute(
            f"INSERT INTO {self.META_TABLE}(key, value) VALUES (?, ?)",
            ("batch_count", "0"))

    # ------------------------------------------------------ initial load
    def _load_initial(self, scan_files: Mapping[str, str]) -> None:
        spec = self.spec
        inserted = {self.FACT_TABLE: 0, self.CAT_TABLE: 0,
                    self.ELIG_TABLE: 0}
        fact_rows: List[Tuple[int, Tuple[Any, ...]]] = []
        for rel, table in (
                (spec.fact_rel, self.FACT_TABLE),
                (spec.cat_rel, self.CAT_TABLE),
                (spec.elig_rel, self.ELIG_TABLE)):
            schema = self.catalog.get(rel)
            alias = {spec.fact_rel: spec.fact_alias,
                     spec.cat_rel: spec.cat_alias,
                     spec.elig_rel: spec.elig_alias}[rel]
            scan = Scan(schema, scan_files[rel], alias)
            col_sql = ", ".join(_qident(c.name) for c in schema.columns)
            params = ", ".join("?" for _ in schema.columns)
            stmt = (f"INSERT INTO {table}(row_id, {col_sql}) "
                    f"VALUES (?, {params})")
            batch: List[Tuple[Any, ...]] = []
            for row in scan:
                rid = row.rid_of(alias)
                batch.append((rid, *row.values))
                if rel == spec.fact_rel:
                    fact_rows.append((rid, row.values))
                if len(batch) >= 512:
                    self._conn.executemany(stmt, batch)
                    inserted[table] += len(batch)
                    batch.clear()
            if batch:
                self._conn.executemany(stmt, batch)
                inserted[table] += len(batch)
            self._trace(0, None, "initial", "initial_load", table,
                        count=inserted[table])

        # Build the summary with the same per-fact contribution logic used by
        # incremental refreshes, so there is a single semantics implementation.
        groups: Dict[Tuple[int, Any], List[int]] = {}
        for rid, values in fact_rows:
            for g_enc, amount in self._contributions(
                    self._conn, rid, values, batch_seq=0, trace=False):
                acc = groups.setdefault(g_enc, [0, 0, 0])
                acc[0] += 1
                if amount is not None:
                    acc[1] += 1
                    acc[2] = self._checked_add(acc[2], amount)
        for (g_isnull, g), (cnt, nn, total) in groups.items():
            self._insert_summary(g_isnull, g, cnt, nn, total)

    # -------------------------------------------------- batch validation
    def _validate_batch(self, corrections: Sequence[Mapping[str, Any]]
                        ) -> List[Dict[str, Any]]:
        if not corrections:
            raise CorrectionError("an empty batch publishes no version")
        spec = self.spec
        known = {spec.fact_rel, spec.cat_rel, spec.elig_rel}
        seen: set = set()
        normalized: List[Dict[str, Any]] = []
        for c in corrections:
            if not isinstance(c, Mapping):
                raise CorrectionError("each correction must be an object")
            rel = c.get("relation")
            if rel not in known:
                raise CorrectionError(
                    f"unknown relation {rel!r}; schemas are fixed at init")
            if "row_id" not in c:
                raise CorrectionError("correction requires row_id")
            rid = check_int64(c["row_id"])
            if (rel, rid) in seen:
                raise CorrectionError(
                    f"duplicate correction for {rel} row_id {rid} in batch")
            seen.add((rel, rid))
            op = c.get("op")
            if op not in ("upsert", "delete"):
                raise CorrectionError(
                    f"correction op must be 'upsert' or 'delete', got {op!r}")
            entry: Dict[str, Any] = {"relation": rel, "row_id": rid, "op": op}
            if op == "upsert":
                schema = self.catalog.get(rel)
                obj = c.get("row")
                obj_rid, values = decode_object(schema, obj)
                if obj_rid != rid:
                    raise CorrectionError(
                        f"row.row_id {obj_rid} differs from correction "
                        f"row_id {rid}")
                entry["values"] = values
            normalized.append(entry)
        return normalized

    # -------------------------------------------------------- refresh API
    def apply_corrections(self, corrections: Sequence[Mapping[str, Any]],
                          inject_fault: Optional[str] = None) -> Dict[str, Any]:
        """Apply one atomic batch; returns evidence about local accesses.

        ``inject_fault`` is test-only: ``"after_first_write"`` fails once the
        first correction row is written, ``"before_commit"`` fails after all
        maintenance but before the version is published.  Both roll back.
        """
        if inject_fault not in (None, "after_first_write", "before_commit"):
            raise ValueError(f"unknown fault {inject_fault!r}")
        normalized = self._validate_batch(corrections)
        conn = self._conn
        spec = self.spec
        # Point-lookup caches are per-batch: every batch starts cold so the
        # access trace evidences each localized lookup of this refresh.
        self._cat_cache.clear()
        self._elig_cache.clear()
        try:
            seq = self._batch_count + 1
            conn.execute("BEGIN IMMEDIATE")

            # Existence checks for deletes happen before any write.
            for entry in normalized:
                if entry["op"] == "delete" and not self._row_exists(
                        self._table_name(entry["relation"]),
                        entry["row_id"]):
                    raise CorrectionError(
                        f"cannot delete missing row "
                        f"{entry['relation']}#{entry['row_id']}")
                if entry["op"] == "delete":
                    self._trace(seq, None, "refresh",
                                "delete_existence_check",
                                self._table_name(entry["relation"]),
                                row_id=entry["row_id"])

            affected = self._collect_affected_facts(conn, normalized)
            old_facts = self._load_facts(conn, affected, batch_seq=seq)
            old_contrib: List[Tuple[Tuple[int, Any], Any]] = []
            for rid, values in old_facts:
                old_contrib.extend(
                    self._contributions(conn, rid, values,
                                        batch_seq=seq, trace=True))

            first_write_done = False
            for entry in normalized:
                self._write_correction(conn, entry)
                self._trace(seq, None, "refresh",
                            "correction_write",
                            self._table_name(entry["relation"]),
                            row_id=entry["row_id"])
                if not first_write_done:
                    first_write_done = True
                    if inject_fault == "after_first_write":
                        raise RuntimeError(
                            "injected fault after first correction write")

            # The point-lookup caches reflect pre-correction rows; drop them
            # so new-state contributions read the just-written match state.
            self._cat_cache.clear()
            self._elig_cache.clear()
            new_facts = self._load_facts(conn, affected, batch_seq=seq)
            new_contrib: List[Tuple[Tuple[int, Any], Any]] = []
            for rid, values in new_facts:
                new_contrib.extend(
                    self._contributions(conn, rid, values,
                                        batch_seq=seq, trace=True))

            groups_updated = self._apply_deltas(
                conn, old_contrib, new_contrib)

            if inject_fault == "before_commit":
                raise RuntimeError("injected fault before commit")

            new_batch = seq
            new_version = self._version_on(conn) + 1
            self._set_meta("batch_count", str(new_batch))
            self._set_meta("version", str(new_version))
            # Re-tag this batch's trace rows with the published version.
            conn.execute(
                f"UPDATE {self.TRACE_TABLE} SET version_after = ? "
                "WHERE batch_seq = ? AND version_after IS NULL",
                (new_version, new_batch))
            conn.commit()
        except BaseException:
            conn.rollback()
            self._cat_cache.clear()
            self._elig_cache.clear()
            raise
        self._batch_count += 1
        self._cat_cache.clear()
        self._elig_cache.clear()
        return {
            "version": self.version(),
            "affected_fact_row_ids": sorted(affected),
            "groups_updated": groups_updated,
            "access": self.trace_summary(batch_seq=self._batch_count),
        }

    # ------------------------------------------------- read side (one API)
    def version(self) -> int:
        return self._version_on(self._conn)

    @staticmethod
    def _version_on(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            f"SELECT value FROM {MaterializedSummary.META_TABLE} "
            "WHERE key = 'version'").fetchone()
        return int(row[0])

    def read_summary(self) -> List[Dict[str, Any]]:
        """Return the whole current summary as committed rows, deterministically."""
        with self._read_conn() as conn:
            rows = conn.execute(
                f"SELECT g_isnull, g, cnt, nonnull_cnt, total "
                f"FROM {self.SUMMARY_TABLE}").fetchall()
        out: List[Dict[str, Any]] = []
        for g_isnull, g, cnt, nonnull_cnt, total in rows:
            group = None if g_isnull else g
            out.append({
                "group": group,
                "count": cnt,
                "sum": total if nonnull_cnt > 0 else None,
            })
        out.sort(key=lambda d: _group_sort_key(d["group"]))
        return out

    def _read_conn(self):
        # A fresh connection only ever sees committed data, so a reader can
        # never observe a half-written batch even while a refresh is open.
        return _closing(sqlite3.connect(self.db_path, isolation_level=None))

    # --------------------------------------------------------- evidence
    def explain_local_access(self) -> List[Dict[str, Any]]:
        """EXPLAIN QUERY PLAN for every localized lookup used by a refresh."""
        spec = self.spec
        statements = [
            ("fact_by_row_id",
             f"SELECT * FROM {self.FACT_TABLE} WHERE row_id = ?"),
            ("corrected_row_by_id",
             f"SELECT * FROM {self.CAT_TABLE} WHERE row_id = ?"),
            ("facts_by_category_key",
             f"SELECT row_id FROM {self.FACT_TABLE} "
             f"WHERE {_qident(spec.fact_fk)} = ?"),
            ("facts_by_eligibility_key",
             f"SELECT row_id FROM {self.FACT_TABLE} "
             f"WHERE {_qident(spec.fact_ek)} = ?"),
            ("category_rows_by_key",
             f"SELECT {_qident(spec.group_col)} FROM {self.CAT_TABLE} "
             f"WHERE {_qident(spec.cat_key)} = ?"),
            ("eligibility_exists_by_key",
             f"SELECT 1 FROM {self.ELIG_TABLE} "
             f"WHERE {_qident(spec.elig_key)} = ? LIMIT 1"),
        ]
        plan: List[Dict[str, Any]] = []
        with self._read_conn() as conn:
            for access, sql in statements:
                # A dummy binding is enough: the plan depends on indexes,
                # not the bound value's type.
                lines = [r[3] for r in conn.execute(
                    "EXPLAIN QUERY PLAN " + sql, (0,)).fetchall()]
                plan.append({"access": access, "sql": sql, "plan": lines})
        return plan

    def access_trace(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = (f"SELECT batch_seq, version_after, kind, access, table_name,"
               f" row_id, key_value FROM {self.TRACE_TABLE}"
               " ORDER BY trace_id DESC")
        params: Tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        cols = ("batch_seq", "version_after", "kind", "access",
                "table_name", "row_id", "key_value")
        with self._read_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def last_access_trace(self) -> List[Dict[str, Any]]:
        with self._read_conn() as conn:
            latest = conn.execute(
                f"SELECT MAX(batch_seq) FROM {self.TRACE_TABLE}").fetchone()[0]
        if latest is None:
            return []
        return [t for t in self.access_trace() if t["batch_seq"] == latest][::-1]

    def trace_summary(self, batch_seq: Optional[int] = None) -> Dict[str, int]:
        where = ""
        params: Tuple[Any, ...] = ()
        if batch_seq is not None:
            where = " WHERE batch_seq = ?"
            params = (batch_seq,)
        with self._read_conn() as conn:
            rows = conn.execute(
                f"SELECT access, table_name, COUNT(*) FROM {self.TRACE_TABLE}"
                f"{where} GROUP BY access, table_name "
                "ORDER BY access, table_name",
                params).fetchall()
        return {f"{access}/{table}": count for access, table, count in rows}

    def reset_access_trace(self) -> None:
        self._conn.execute(f"DELETE FROM {self.TRACE_TABLE}")

    # ---------------------------------------------------------- internals
    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            f"UPDATE {self.META_TABLE} SET value = ? WHERE key = ?",
            (value, key))

    def _row_exists(self, table: str, rid: int) -> bool:
        return self._conn.execute(
            f"SELECT 1 FROM {table} WHERE row_id = ?", (rid,)).fetchone() \
            is not None

    def _trace(self, batch_seq: int, version_after: Optional[int],
               kind: str, access: str, table: str,
               row_id: Optional[int] = None, key: Any = None,
               count: Optional[int] = None) -> None:
        if count is not None:
            # Bulk evidence: one row carrying the number of loaded rows.
            self._conn.execute(
                f"INSERT INTO {self.TRACE_TABLE}"
                "(batch_seq, version_after, kind, access, table_name,"
                " row_id, key_value) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (batch_seq, version_after, kind, access, table, None,
                 _key_json({"count": count})))
            return
        self._conn.execute(
            f"INSERT INTO {self.TRACE_TABLE}"
            "(batch_seq, version_after, kind, access, table_name,"
            " row_id, key_value) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (batch_seq, version_after, kind, access, table, row_id,
             _key_json(key) if key is not None else None))

    def _collect_affected_facts(self, conn: sqlite3.Connection,
                                normalized: Sequence[Mapping[str, Any]]
                                ) -> set:
        spec = self.spec
        affected: set = set()
        cat_table = self.CAT_TABLE
        elig_table = self.ELIG_TABLE
        moved_cat_keys: set = set()
        moved_elig_keys: set = set()

        for entry in normalized:
            rel = entry["relation"]
            rid = entry["row_id"]
            if rel == spec.fact_rel:
                affected.add(rid)  # directly corrected fact
                continue
            if entry["op"] == "delete":
                old = self._get_row(conn, self._table_name(rel), rid,
                                    trace_seq=self._batch_count + 1)
                if old is None:  # pragma: no cover - validated earlier
                    raise CorrectionError(
                        f"cannot delete missing row {rel}#{rid}")
                new_values: Optional[Tuple[Any, ...]] = None
            else:
                old = self._get_row(conn, self._table_name(rel), rid,
                                    trace_seq=self._batch_count + 1)
                new_values = entry["values"]

            if rel == spec.cat_rel:
                key_col = spec.cat_key
                fact_col = spec.fact_fk
                target_keys = moved_cat_keys
                table = cat_table
            else:
                key_col = spec.elig_key
                fact_col = spec.fact_ek
                target_keys = moved_elig_keys
                table = elig_table
            idx = self._col_index[rel][key_col]
            old_key = old[idx] if old is not None else None
            new_key = new_values[idx] if new_values is not None else None
            for key in (old_key, new_key):
                if key is None:
                    continue  # NULL keys never match: no fact depends on them
                target_keys.add(key)

        for key in moved_cat_keys:
            for (frid,) in conn.execute(
                    f"SELECT row_id FROM {self.FACT_TABLE} "
                    f"WHERE {_qident(spec.fact_fk)} = ?", (key,)):
                affected.add(frid)
            self._trace(self._batch_count + 1, None, "refresh",
                        "facts_by_category_key", self.FACT_TABLE, key=key)
        for key in moved_elig_keys:
            for (frid,) in conn.execute(
                    f"SELECT row_id FROM {self.FACT_TABLE} "
                    f"WHERE {_qident(spec.fact_ek)} = ?", (key,)):
                affected.add(frid)
            self._trace(self._batch_count + 1, None, "refresh",
                        "facts_by_eligibility_key", self.FACT_TABLE, key=key)
        return affected

    def _get_row(self, conn: sqlite3.Connection, table: str,
                 rid: int, trace_seq: Optional[int] = None
                 ) -> Optional[Tuple[Any, ...]]:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE row_id = ?", (rid,)).fetchone()
        if trace_seq is not None:
            self._trace(trace_seq, None, "refresh",
                        "corrected_row_by_id", table, row_id=rid)
        if row is None:
            return None
        return tuple(row[1:])  # drop row_id

    def _load_facts(self, conn: sqlite3.Connection, rids: set,
                    batch_seq: int) -> List[Tuple[int, Tuple[Any, ...]]]:
        if not rids:
            return []
        out: List[Tuple[int, Tuple[Any, ...]]] = []
        for rid in sorted(rids):
            row = conn.execute(
                f"SELECT * FROM {self.FACT_TABLE} WHERE row_id = ?",
                (rid,)).fetchone()
            self._trace(batch_seq, None, "refresh",
                        "fact_by_row_id", self.FACT_TABLE, row_id=rid)
            if row is not None:
                out.append((rid, tuple(row[1:])))
        return out

    def _write_correction(self, conn: sqlite3.Connection,
                          entry: Mapping[str, Any]) -> None:
        rel = entry["relation"]
        table = self._table_name(rel)
        rid = entry["row_id"]
        if entry["op"] == "delete":
            conn.execute(f"DELETE FROM {table} WHERE row_id = ?", (rid,))
            return
        schema = self.catalog.get(rel)
        cols = ", ".join(_qident(c.name) for c in schema.columns)
        params = ", ".join("?" for _ in schema.columns)
        conn.execute(
            f"INSERT OR REPLACE INTO {table}(row_id, {cols}) "
            f"VALUES (?, {params})",
            (rid, *entry["values"]))

    def _contributions(self, conn: sqlite3.Connection, rid: int,
                       values: Sequence[Any], batch_seq: int,
                       trace: bool) -> List[Tuple[Tuple[int, Any], Any]]:
        """Joined rows one fact contributes after SEMI then LEFT, with groups.

        Returns ``((g_isnull, g_encoded), amount_or_none)`` per joined row.
        """
        spec = self.spec
        f = self._col_index[spec.fact_rel]
        ek = values[f[spec.fact_ek]]
        # SEMI: a NULL join key never matches, so the fact is dropped wholly
        # (no multiplication and no LEFT pad either).
        if ek is None or not self._eligibility_matches(
                conn, ek, batch_seq, trace):
            return []

        fk = values[f[spec.fact_fk]]
        groups: List[Any]
        if fk is None:
            groups = [None]  # NULL join key never matches -> single pad
        else:
            matched = self._category_groups(conn, fk, batch_seq, trace)
            groups = matched if matched else [None]

        amount = values[f[spec.sum_col]]
        result: List[Tuple[Tuple[int, Any], Any]] = []
        for g in groups:
            if g is None:
                result.append(((1, 0), amount))
            else:
                result.append(((0, g), amount))
        return result

    def _eligibility_matches(self, conn: sqlite3.Connection, key: Any,
                             batch_seq: int, trace: bool) -> bool:
        if key in self._elig_cache:
            return self._elig_cache[key]
        row = conn.execute(
            f"SELECT 1 FROM {self.ELIG_TABLE} "
            f"WHERE {_qident(self.spec.elig_key)} = ? LIMIT 1",
            (key,)).fetchone()
        matches = row is not None
        self._elig_cache[key] = matches
        if trace:
            self._trace(batch_seq, None, "refresh",
                        "eligibility_exists_by_key", self.ELIG_TABLE,
                        key=key)
        return matches

    def _category_groups(self, conn: sqlite3.Connection, key: Any,
                         batch_seq: int, trace: bool) -> List[Any]:
        if key in self._cat_cache:
            return self._cat_cache[key]  # type: ignore[return-value]
        rows = conn.execute(
            f"SELECT {_qident(self.spec.group_col)} FROM {self.CAT_TABLE} "
            f"WHERE {_qident(self.spec.cat_key)} = ?",
            (key,)).fetchall()
        groups = [r[0] for r in rows]
        self._cat_cache[key] = groups
        if trace:
            self._trace(batch_seq, None, "refresh",
                        "category_rows_by_key", self.CAT_TABLE, key=key)
        return groups

    def _apply_deltas(self, conn: sqlite3.Connection,
                      old_contrib: Sequence[Tuple[Tuple[int, Any], Any]],
                      new_contrib: Sequence[Tuple[Tuple[int, Any], Any]]
                      ) -> List[Any]:
        deltas: Dict[Tuple[int, Any], List[int]] = {}

        def add(contrib: Sequence[Tuple[Tuple[int, Any], Any]], sign: int
                ) -> None:
            for g_enc, amount in contrib:
                d = deltas.setdefault(g_enc, [0, 0, 0])
                d[0] += sign
                if amount is not None:
                    d[1] += sign
                    d[2] += sign * amount

        add(old_contrib, -1)
        add(new_contrib, 1)

        updated: List[Any] = []
        for (g_isnull, g), (dcnt, dnn, dtot) in deltas.items():
            row = conn.execute(
                f"SELECT cnt, nonnull_cnt, total FROM {self.SUMMARY_TABLE} "
                "WHERE g_isnull = ? AND g = ?", (g_isnull, g)).fetchone()
            if row is None:
                cnt, nn, total = dcnt, dnn, dtot
                if cnt == 0:
                    continue  # nothing to publish
            else:
                cnt, nn, total = row[0] + dcnt, row[1] + dnn, row[2] + dtot
            if cnt < 0 or nn < 0 or nn > cnt:  # pragma: no cover - invariant
                raise MaterializedError("internal group invariant violated")
            if not INT64_MIN <= cnt <= INT64_MAX:
                raise OverflowRejected("count outside int64")
            if not INT64_MIN <= total <= INT64_MAX:
                raise OverflowRejected(
                    f"published sum for group {_decode_group(g_isnull, g)!r} "
                    "overflows int64; batch rejected")
            if cnt == 0:
                conn.execute(
                    f"DELETE FROM {self.SUMMARY_TABLE} "
                    "WHERE g_isnull = ? AND g = ?", (g_isnull, g))
            elif row is None:
                self._insert_summary(g_isnull, g, cnt, nn, total)
            else:
                conn.execute(
                    f"UPDATE {self.SUMMARY_TABLE} SET cnt = ?, nonnull_cnt = ?,"
                    " total = ? WHERE g_isnull = ? AND g = ?",
                    (cnt, nn, total, g_isnull, g))
            updated.append(_decode_group(g_isnull, g))
        updated.sort(key=_group_sort_key)
        return updated

    def _insert_summary(self, g_isnull: int, g: Any, cnt: int,
                        nonnull_cnt: int, total: int) -> None:
        self._conn.execute(
            f"INSERT INTO {self.SUMMARY_TABLE}"
            "(g_isnull, g, cnt, nonnull_cnt, total) VALUES (?, ?, ?, ?, ?)",
            (g_isnull, g, cnt, nonnull_cnt, total))

    @staticmethod
    def _checked_add(a: int, b: int) -> int:
        r = a + b
        if r < INT64_MIN or r > INT64_MAX:
            raise OverflowRejected(
                f"int64 sum overflow while building summary: {a} + {b}")
        return r


# ---------------------------------------------------------------- helpers


def tree_walk_scans(node: dict) -> List[dict]:
    if node.get("node") == "scan":
        return [node]
    if node.get("node") == "join":
        return tree_walk_scans(node["left"]) + tree_walk_scans(node["right"])
    return tree_walk_scans(node["input"])


def _decode_group(g_isnull: int, g: Any) -> Any:
    return None if g_isnull else g


def _group_sort_key(value: Any):
    if value is None:
        return (0, 0)
    if isinstance(value, int):
        return (1, value)
    return (2, value)


def _schema_from_dict(d: dict):
    from .types import schema_from_json
    return schema_from_json(d["relation"], d["columns"])


class _closing:
    """Minimal context manager that always closes its SQLite connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, *exc: Any) -> None:
        self._conn.close()
