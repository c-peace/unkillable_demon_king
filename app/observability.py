from __future__ import annotations

import json
import logging
from typing import Any


def emit_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one compact structural event without message or evidence content."""
    payload = {"event": event, **fields}
    logger.log(
        level,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )
