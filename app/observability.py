from __future__ import annotations

import contextvars
import logging


_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "lunit_request_id",
    default="-",
)


def set_request_id(request_id: str) -> None:
    _REQUEST_ID.set(request_id or "-")


def get_request_id() -> str:
    return _REQUEST_ID.get()


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"
        ),
    )
    context_filter = RequestContextFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(context_filter)
