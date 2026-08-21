from __future__ import annotations

import time
from dataclasses import dataclass

from app.errors import DeadlineExceeded


@dataclass(frozen=True, slots=True)
class Deadline:
    expires_at: float

    @classmethod
    def after(cls, seconds: float) -> "Deadline":
        return cls(expires_at=time.monotonic() + seconds)

    def remaining(self, cap: float | None = None) -> float:
        value = self.expires_at - time.monotonic()
        if value <= 0:
            raise DeadlineExceeded()
        return min(value, cap) if cap is not None else value

    def can_start(self, minimum_seconds: float) -> bool:
        return self.expires_at - time.monotonic() >= minimum_seconds
