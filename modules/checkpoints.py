import re
from pathlib import Path
from typing import Any


CKPT_RE = re.compile(r"_epoch(?P<epoch>\d+)_metric(?P<metric>[0-9.]+)\.ckpt$")


def checkpoint_info(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    match = CKPT_RE.search(path.name)
    epoch = int(match.group("epoch")) if match else None
    metric = float(match.group("metric")) if match else None
    task = "segmentation" if "Segmentation" in path.name else "classification" if "Classification" in path.name else "unknown"
    return {
        "path": str(path),
        "name": path.name,
        "task": task,
        "epoch": epoch,
        "metric": metric,
        "bytes": path.stat().st_size if path.exists() else None,
        "modified_time": path.stat().st_mtime if path.exists() else None,
    }


def find_checkpoints(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("**/*.ckpt"))


def best_checkpoint(directory: str | Path) -> Path | None:
    checkpoints = find_checkpoints(directory)
    if not checkpoints:
        return None

    def sort_key(path: Path) -> tuple[float, float]:
        info = checkpoint_info(path)
        metric = info["metric"] if info["metric"] is not None else float("-inf")
        return metric, path.stat().st_mtime

    return max(checkpoints, key=sort_key)


def checkpoint_table(directory: str | Path) -> list[dict[str, Any]]:
    rows = [checkpoint_info(path) for path in find_checkpoints(directory)]
    return sorted(
        rows,
        key=lambda row: (
            row["task"],
            row["metric"] if row["metric"] is not None else float("-inf"),
            row["epoch"] if row["epoch"] is not None else -1,
        ),
        reverse=True,
    )

