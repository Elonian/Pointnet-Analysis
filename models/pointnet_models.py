import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POINTNET_DIR = PROJECT_ROOT / "sub-modules" / "pointnet"

if str(POINTNET_DIR) not in sys.path:
    sys.path.insert(0, str(POINTNET_DIR))

from model import (  # noqa: E402
    PointNetCls,
    PointNetFeat,
    PointNetPartSeg,
    STNKd,
    get_orthogonal_loss,
)


__all__ = [
    "PointNetCls",
    "PointNetFeat",
    "PointNetPartSeg",
    "STNKd",
    "get_orthogonal_loss",
]
