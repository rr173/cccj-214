"""Shared process-wide logical memory budget.

All live operators reserve/release through one :class:`MemoryBudget`, so a
hot build side, buffered probe rows and aggregate state compete for the same
limit.  Reservations raise :class:`MemoryLimitExceeded` when the budget is
exhausted; the join operator catches that as its automatic switch trigger.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .errors import MemoryLimitExceeded


@dataclass
class Budget:
    limit_bytes: int
    used_bytes: int = 0
    peak_bytes: int = 0
    #: Per-owner reservations for explain/debug output.
    owners: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, n: int, owner: str = "?") -> None:
        if n <= 0:
            return
        with self._lock:
            if self.used_bytes + n > self.limit_bytes:
                raise MemoryLimitExceeded(
                    f"budget {self.limit_bytes} bytes exceeded: "
                    f"used={self.used_bytes} request={n} owner={owner}"
                )
            self.used_bytes += n
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)
            self.owners[owner] = self.owners.get(owner, 0) + n

    def release(self, n: int, owner: str = "?") -> None:
        if n <= 0:
            return
        with self._lock:
            self.used_bytes -= n
            if self.used_bytes < 0:  # pragma: no cover - defensive
                self.used_bytes = 0
            self.owners[owner] = max(0, self.owners.get(owner, 0) - n)

    def available(self) -> int:
        return max(0, self.limit_bytes - self.used_bytes)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "limit_bytes": self.limit_bytes,
                "used_bytes": self.used_bytes,
                "peak_bytes": self.peak_bytes,
                "owners": dict(self.owners),
            }
