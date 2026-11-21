"""Runtime-adaptive relational join executor.

Public entry points:

* :class:`~join_exec.engine.QueryEngine` — builds and runs query trees;
* :class:`~join_exec.join.HybridJoin` — the observable join operator that
  owns its memory/disk mode and switch state;
* :class:`~join_exec.reference.ReferenceInterpreter` — independent nested-loop
  test oracle;
* :class:`~join_exec.budget.Budget`, :class:`~join_exec.control.ControlPoint`,
  :class:`~join_exec.spill.SpillManager` — runtime controls and accounting.
"""

from .budget import Budget
from .control import ControlPoint
from .engine import QueryEngine, count_leaves, normalize_inner_tree
from .errors import (
    BadRowError,
    JoinExecError,
    MemoryLimitExceeded,
    ProofError,
    RunCancelled,
    SchemaError,
    SpillError,
)
from .join import HybridJoin
from .proof_store import ProofReader, ProofStore
from .provenance import ProvenanceSession
from .reference import ReferenceInterpreter
from .schema_catalog import SchemaCatalog
from .spill import SpillManager
from .types import Header, Row, Schema, schema_from_json

__all__ = [
    "Budget",
    "ControlPoint",
    "QueryEngine",
    "count_leaves",
    "normalize_inner_tree",
    "BadRowError",
    "JoinExecError",
    "MemoryLimitExceeded",
    "ProofError",
    "RunCancelled",
    "SchemaError",
    "SpillError",
    "HybridJoin",
    "ProofReader",
    "ProofStore",
    "ProvenanceSession",
    "ReferenceInterpreter",
    "SchemaCatalog",
    "SpillManager",
    "Header",
    "Row",
    "Schema",
    "schema_from_json",
]
