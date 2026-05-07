import os
import re
import sys
from pathlib import Path
from typing import Optional

import yaml


DATASET_DIRS = (
    "modelnet40_ply_hdf5_2048",
    "shapenet_part_seg_hdf5_data",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_to_syspath(path: str | Path, prepend: bool = True) -> None:
    value = str(Path(path).resolve())
    if value in sys.path:
        return
    if prepend:
        sys.path.insert(0, value)
    else:
        sys.path.append(value)


def ensure_data_links(pointnet_dir: Path, data_dir: Path) -> None:
    pointnet_data = pointnet_dir / "data"
    pointnet_data.mkdir(parents=True, exist_ok=True)

    for dataset_dir in DATASET_DIRS:
        src = data_dir / dataset_dir
        dst = pointnet_data / dataset_dir
        if not src.exists():
            raise FileNotFoundError(f"Missing dataset directory: {src}")
        if dst.is_symlink() and dst.resolve() == src.resolve():
            continue
        if dst.exists() and not dst.is_symlink():
            continue
        if dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)


def checkpoint_metric(path: Path) -> Optional[float]:
    match = re.search(r"_metric([0-9.]+)\.ckpt$", path.name)
    return float(match.group(1)) if match else None


def best_checkpoint(directory: str | Path) -> Optional[Path]:
    ckpts = list(Path(directory).glob("**/*.ckpt"))
    if not ckpts:
        return None

    def sort_key(path: Path) -> tuple[float, float]:
        metric = checkpoint_metric(path)
        metric_value = metric if metric is not None else float("-inf")
        return metric_value, path.stat().st_mtime

    return max(ckpts, key=sort_key)
