import logging
from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_log_dir(log_root: str | Path, run_id: str | None = None) -> Path:
    run_dir = Path(log_root) / (run_id or timestamp())
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_logger(name: str, log_file: str | Path) -> logging.Logger:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_mapping(logger: logging.Logger, title: str, values: dict) -> None:
    logger.info("%s", title)
    for key, value in values.items():
        logger.info("  %s=%s", key, value)
