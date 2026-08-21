from __future__ import annotations

import logging
import sys

from app.config import Settings
from app.server import serve


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        settings = Settings.from_env()
    except (TypeError, ValueError) as exc:
        logging.getLogger("lunit_server").error("invalid_configuration %s", exc)
        return 2
    serve(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
