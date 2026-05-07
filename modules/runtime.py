import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path | None, default: str | Path | None = None) -> Path | None:
    raw = value if value is not None else default
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def add_to_syspath(path: str | Path, prepend: bool = True) -> None:
    value = str(Path(path).resolve())
    if value in sys.path:
        return
    if prepend:
        sys.path.insert(0, value)
    else:
        sys.path.append(value)


def device_from_gpu(gpu: int | None) -> torch.device:
    if gpu is None:
        gpu = 0
    if gpu == -1 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu}")


def amp_enabled(requested: bool, device: torch.device) -> bool:
    return bool(requested and device.type == "cuda")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_summary(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
    }
    if torch.cuda.is_available():
        summary["cuda_version"] = torch.version.cuda
        summary["device_count"] = torch.cuda.device_count()
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            summary["device_name"] = props.name
            summary["device_memory_gb"] = round(props.total_memory / (1024**3), 2)
    return summary


def ensure_data_links(pointnet_dir: Path, data_dir: Path) -> None:
    pointnet_data = pointnet_dir / "data"
    pointnet_data.mkdir(parents=True, exist_ok=True)
    for dataset_dir in ("modelnet40_ply_hdf5_2048", "shapenet_part_seg_hdf5_data"):
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

