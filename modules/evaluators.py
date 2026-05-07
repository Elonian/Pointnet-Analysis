import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.datasets import modelnet_class_names, shapenet_categories, shapenet_part_ids
from modules.runtime import add_to_syspath


def prepare_pointnet_imports(pointnet_dir: str | Path) -> None:
    pointnet_dir = Path(pointnet_dir).resolve()
    add_to_syspath(pointnet_dir, prepend=True)
    for name in list(sys.modules):
        if name == "model" or name == "dataloaders" or name.startswith("dataloaders."):
            sys.modules.pop(name, None)
        if name == "utils" or name.startswith("utils."):
            module = sys.modules.get(name)
            module_file = getattr(module, "__file__", "") or ""
            if not module_file or str(pointnet_dir) not in module_file:
                sys.modules.pop(name, None)


def load_pointnet_api(pointnet_dir: str | Path) -> dict[str, Any]:
    prepare_pointnet_imports(pointnet_dir)
    model_module = importlib.import_module("model")
    modelnet_module = importlib.import_module("dataloaders.modelnet")
    shapenet_module = importlib.import_module("dataloaders.shapenet_partseg")
    return {
        "PointNetCls": model_module.PointNetCls,
        "PointNetPartSeg": model_module.PointNetPartSeg,
        "get_orthogonal_loss": model_module.get_orthogonal_loss,
        "get_modelnet_loaders": modelnet_module.get_data_loaders,
        "get_shapenet_loaders": shapenet_module.get_data_loaders,
    }


def load_state(model: torch.nn.Module, ckpt_path: str | Path, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)


def classification_eval(
    pointnet_dir: str | Path,
    data_dir: str | Path,
    ckpt_path: str | Path,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    api = load_pointnet_api(pointnet_dir)
    model = api["PointNetCls"](num_classes=40, input_transform=True, feature_transform=True).to(device)
    load_state(model, ckpt_path, device)
    model.eval()

    _, dataloaders = api["get_modelnet_loaders"](
        data_dir=str(Path(pointnet_dir) / "data"),
        batch_size=batch_size,
        phases=["test"],
    )
    names = modelnet_class_names(data_dir)
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    losses = []
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_index, (points, labels) in enumerate(tqdm(dataloaders[0], desc="classification eval")):
            if max_batches is not None and batch_index >= max_batches:
                break
            points = points.to(device)
            labels = labels.to(device).long()
            logits, _, feat_trans = model(points)
            loss = F.cross_entropy(logits, labels) + api["get_orthogonal_loss"](feat_trans)
            preds = torch.argmax(logits, dim=1)
            losses.append(float(loss.detach().cpu()))
            total += int(labels.numel())
            correct += int((preds == labels).sum().detach().cpu())
            for target, pred in zip(labels.detach().cpu().numpy(), preds.detach().cpu().numpy()):
                confusion[int(target), int(pred)] += 1

    per_class_acc = []
    for i, name in enumerate(names):
        class_total = int(confusion[i].sum())
        per_class_acc.append(
            {
                "class_id": i,
                "class_name": name,
                "accuracy": float(confusion[i, i] / class_total) if class_total else 0.0,
                "count": class_total,
            }
        )

    return {
        "checkpoint": str(ckpt_path),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(correct / total) if total else 0.0,
        "total": total,
        "correct": correct,
        "class_names": names,
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": per_class_acc,
    }


def _masked_segmentation_prediction(logit: torch.Tensor, class_label: int) -> torch.Tensor:
    pids = shapenet_part_ids()[class_label]
    mask = torch.zeros_like(logit)
    mask[pids, :] = 1
    return torch.argmax(logit.masked_fill(mask == 0, -1e9), dim=0)


def segmentation_eval(
    pointnet_dir: str | Path,
    data_dir: str | Path,
    ckpt_path: str | Path,
    batch_size: int,
    device: torch.device,
    sample_count: int = 0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    api = load_pointnet_api(pointnet_dir)
    model = api["PointNetPartSeg"]().to(device)
    load_state(model, ckpt_path, device)
    model.eval()

    _, dataloaders = api["get_shapenet_loaders"](
        data_dir=str(Path(pointnet_dir) / "data"),
        batch_size=batch_size,
        phases=["test"],
    )
    categories = shapenet_categories(data_dir)
    part_map = shapenet_part_ids()
    class_iou_sum = np.zeros(len(categories), dtype=np.float64)
    class_iou_count = np.zeros(len(categories), dtype=np.int64)
    part_iou_sum = np.zeros(50, dtype=np.float64)
    part_iou_count = np.zeros(50, dtype=np.int64)
    losses = []
    raw_correct = 0
    masked_correct = 0
    total_points = 0
    collected_points = []
    collected_targets = []
    collected_preds = []

    with torch.no_grad():
        for batch_index, (points, pc_labels, class_labels) in enumerate(tqdm(dataloaders[0], desc="segmentation eval")):
            if max_batches is not None and batch_index >= max_batches:
                break
            points = points.to(device)
            pc_labels = pc_labels.to(device).long()
            class_labels = class_labels.to(device).long()
            logits, _, feat_trans = model(points)
            loss = F.cross_entropy(logits, pc_labels) + api["get_orthogonal_loss"](feat_trans)
            raw_preds = torch.argmax(logits, dim=1)
            losses.append(float(loss.detach().cpu()))
            raw_correct += int((raw_preds == pc_labels).sum().detach().cpu())
            total_points += int(pc_labels.numel())

            for i in range(points.shape[0]):
                class_id = int(class_labels[i].detach().cpu())
                target = pc_labels[i]
                masked_pred = _masked_segmentation_prediction(logits[i], class_id)
                masked_correct += int((masked_pred == target).sum().detach().cpu())
                instance_iou = 0.0
                for pid in part_map[class_id]:
                    pred_mask = masked_pred == pid
                    target_mask = target == pid
                    union = (pred_mask | target_mask).sum()
                    inter = (pred_mask & target_mask).sum()
                    iou = 1.0 if int(union) == 0 else float((inter.float() / union.float()).detach().cpu())
                    instance_iou += iou
                    part_iou_sum[pid] += iou
                    part_iou_count[pid] += 1
                instance_iou /= len(part_map[class_id])
                class_iou_sum[class_id] += instance_iou
                class_iou_count[class_id] += 1

                if len(collected_points) < sample_count:
                    collected_points.append(points[i].detach().cpu().numpy())
                    collected_targets.append(target.detach().cpu().numpy())
                    collected_preds.append(masked_pred.detach().cpu().numpy())

    class_iou = []
    for category in categories:
        idx = category["index"]
        count = int(class_iou_count[idx])
        class_iou.append(
            {
                "class_id": idx,
                "class_name": category["name"],
                "miou": float(class_iou_sum[idx] / count) if count else 0.0,
                "count": count,
            }
        )
    part_iou = []
    for pid in range(50):
        count = int(part_iou_count[pid])
        part_iou.append(
            {
                "part_id": pid,
                "miou": float(part_iou_sum[pid] / count) if count else 0.0,
                "count": count,
            }
        )

    return {
        "checkpoint": str(ckpt_path),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "raw_point_accuracy": float(raw_correct / total_points) if total_points else 0.0,
        "masked_point_accuracy": float(masked_correct / total_points) if total_points else 0.0,
        "miou": float(class_iou_sum.sum() / class_iou_count.sum()) if class_iou_count.sum() else 0.0,
        "total_points": total_points,
        "class_iou": class_iou,
        "part_iou": part_iou,
        "samples": {
            "pointclouds": np.asarray(collected_points).tolist(),
            "groundtruths": np.asarray(collected_targets).tolist(),
            "predictions": np.asarray(collected_preds).tolist(),
        },
    }
