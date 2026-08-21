from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AppError(Exception):
    message: str
    status_code: int = 500
    code: str = "internal_error"
    error_type: str = "server_error"
    param: str | None = None

    def __str__(self) -> str:
        return self.message

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class ConfigurationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=503,
            code="service_not_configured",
            error_type="configuration_error",
        )


class UpstreamError(AppError):
    def __init__(self, message: str, *, code: str = "upstream_error") -> None:
        super().__init__(
            message=message,
            status_code=502,
            code=code,
            error_type="upstream_error",
        )


class DeadlineExceeded(AppError):
    def __init__(self, message: str = "The request exceeded its execution deadline.") -> None:
        super().__init__(
            message=message,
            status_code=504,
            code="deadline_exceeded",
            error_type="timeout_error",
        )
