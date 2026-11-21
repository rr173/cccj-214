"""Exception types for the runtime-adaptive join executor.

All error conditions that must terminate a run have distinct types so callers
can distinguish bad input, budget problems, cancellation and spill IO failure.
"""


class JoinExecError(Exception):
    """Base class for all executor errors."""


class BadRowError(JoinExecError):
    """A JSONL line cannot be decoded or violates the declared schema."""


class SchemaError(JoinExecError):
    """A query references an unknown field or builds an invalid node."""


class MemoryLimitExceeded(JoinExecError):
    """Raised by the shared budget when a reservation cannot be satisfied.

    Join operators treat this as a *signal* to attempt a memory->disk switch;
    other operators propagate it as a fatal error.
    """


class SpillError(JoinExecError):
    """Writing or reading a spill partition failed."""


class RunCancelled(JoinExecError):
    """The run was cancelled via a control point or consumer close."""
