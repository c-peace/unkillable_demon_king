from __future__ import annotations

import logging
import sys

from app.config import Settings
from app.observability import configure_logging
from app.server import serve


def main() -> int:
    configure_logging()
    logger = logging.getLogger("lunit_server")
    try:
        settings = Settings.from_env()
    except (TypeError, ValueError) as exc:
        logger.error("invalid_configuration %s", exc)
        return 2
    try:
        serve(settings)
    except Exception:
        # The evaluator only ever sees this process's output. A bare traceback on a
        # failed bind or startup is hard to read in the dashboard log tail, so name
        # the failure explicitly before re-raising the detail.
        logger.exception(
            "startup_failed host=%s port=%s model=%s",
            settings.host,
            settings.port,
            settings.model,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
