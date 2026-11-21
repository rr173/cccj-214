"""Test/control points shared by the running query.

Two switch request channels are provided and both are consumed by the *same*
``HybridJoin._switch`` implementation:

* ``request_switch_after_outputs(n)`` / batch variant target a specific join
  operator (or all operators when ``join_id`` is None);
* automatic pressure triggers (budget overflow, hot-key threshold) call
  exactly the same request method internally.

Cancellation is delivered through the same object and checked at every row
boundary so a run ends promptly with files and sub-iterators closed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ControlPoint:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _switch_after_outputs: dict = field(default_factory=dict)   # join_id -> remaining
    _switch_after_batches: dict = field(default_factory=dict)   # join_id -> remaining
    _cancelled: bool = False
    reason: Optional[str] = None

    # -- external requests -------------------------------------------------
    def request_switch_after_outputs(self, n: int, join_id: Optional[str] = None) -> None:
        with self._lock:
            self._switch_after_outputs[join_id] = n

    def request_switch_after_input_batches(self, n: int, join_id: Optional[str] = None) -> None:
        with self._lock:
            self._switch_after_batches[join_id] = n

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            self._cancelled = True
            self.reason = reason

    # -- operator hooks ----------------------------------------------------
    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def note_output(self, join_id: str) -> bool:
        """Return True when this output boundary requests a switch."""
        with self._lock:
            hit = False
            for key in (join_id, None):
                if key in self._switch_after_outputs:
                    self._switch_after_outputs[key] -= 1
                    if self._switch_after_outputs[key] <= 0:
                        del self._switch_after_outputs[key]
                        hit = True
            return hit

    def note_input_batch(self, join_id: str) -> bool:
        with self._lock:
            hit = False
            for key in (join_id, None):
                if key in self._switch_after_batches:
                    self._switch_after_batches[key] -= 1
                    if self._switch_after_batches[key] <= 0:
                        del self._switch_after_batches[key]
                        hit = True
            return hit

    def pending(self) -> dict:
        with self._lock:
            return {
                "switch_after_outputs": dict(self._switch_after_outputs),
                "switch_after_batches": dict(self._switch_after_batches),
                "cancelled": self._cancelled,
            }


class NullControl(ControlPoint):
    """Control point that never fires; used when the caller passes nothing."""
