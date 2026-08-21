from __future__ import annotations

import time
from dataclasses import dataclass

from app.errors import DeadlineExceeded


@dataclass(frozen=True, slots=True)
class Deadline:
    expires_at: float | None

    @classmethod
    def unbounded(cls) -> "Deadline":
        """Create an execution window with no shared wall-clock cutoff.

        Upstream calls still receive their own explicit timeout cap.
        """
        return cls(expires_at=None)

    @classmethod
    def after(cls, seconds: float | None) -> "Deadline":
        if seconds is None:
            return cls.unbounded()
        return cls(expires_at=time.monotonic() + seconds)

    def with_timeout_cap(self, seconds: float | None) -> "Deadline":
        """Derive a stage deadline without extending an existing request deadline."""
        if seconds is None:
            return self
        capped_expiry = time.monotonic() + seconds
        if self.expires_at is None:
            return Deadline(expires_at=capped_expiry)
        return Deadline(expires_at=min(self.expires_at, capped_expiry))

    def remaining(self, cap: float | None = None) -> float:
        if self.expires_at is None:
            if cap is None:
                raise ValueError("an unbounded deadline requires an explicit cap")
            return cap
        value = self.expires_at - time.monotonic()
        if value <= 0:
            raise DeadlineExceeded()
        return min(value, cap) if cap is not None else value

    def can_start(self, minimum_seconds: float) -> bool:
        if self.expires_at is None:
            return True
        return self.expires_at - time.monotonic() >= minimum_seconds
