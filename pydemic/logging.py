import logging
import os

logging.basicConfig(format="%(levelname)s: %(message)s")
log = logging.getLogger("pydemic")

if os.environ.get("DEBUG", "").lower() in ("1", "true", "on"):
    log.setLevel(logging.DEBUG)
    log.info("Debug enabled")
