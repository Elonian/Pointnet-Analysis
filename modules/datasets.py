from pathlib import Path
from typing import Any

import h5py
import numpy as np


MODELNET_DIR = "modelnet40_ply_hdf5_2048"
SHAPENET_DIR = "shapenet_part_seg_hdf5_data"


def read_lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def modelnet_class_names(data_dir: str | Path) -> list[str]:
    return read_lines(Path(data_dir) / MODELNET_DIR / "shape_names.txt")


def shapenet_categories(data_dir: str | Path) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(read_lines(Path(data_dir) / SHAPENET_DIR / "all_object_categories.txt")):
        name, synset = line.split()
        rows.append({"index": index, "name": name, "synset": synset})
    return rows


def shapenet_part_ids() -> dict[int, list[int]]:
    return {
        0: [0, 1, 2, 3],
        1: [4, 5],
        2: [6, 7],
        3: [8, 9, 10, 11],
        4: [12, 13, 14, 15],
        5: [16, 17, 18],
        6: [19, 20, 21],
        7: [22, 23],
        8: [24, 25, 26, 27],
        9: [28, 29],
        10: [30, 31, 32, 33, 34, 35],
        11: [36, 37],
        12: [38, 39, 40],
        13: [41, 42, 43],
        14: [44, 45, 46],
        15: [47, 48, 49],
    }


def h5_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    datasets: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            item = f[key]
            info: dict[str, Any] = {
                "shape": list(item.shape),
                "dtype": str(item.dtype),
            }
            if key in ("label", "pid"):
                values = np.asarray(item[:]).reshape(-1)
                info.update(
                    {
                        "min": int(values.min()),
                        "max": int(values.max()),
                        "unique": int(np.unique(values).size),
                    }
                )
            datasets[key] = info
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "datasets": datasets,
    }


def dataset_manifest(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    manifest: dict[str, Any] = {
        "data_dir": str(data_dir),
        "modelnet40": {
            "classes": modelnet_class_names(data_dir),
            "files": [],
        },
        "shapenet_part": {
            "categories": shapenet_categories(data_dir),
            "files": [],
        },
    }
    for path in sorted((data_dir / MODELNET_DIR).glob("*.h5")):
        manifest["modelnet40"]["files"].append(h5_summary(path))
    for path in sorted((data_dir / SHAPENET_DIR).glob("*.h5")):
        manifest["shapenet_part"]["files"].append(h5_summary(path))
    return manifest

