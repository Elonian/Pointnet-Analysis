from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_curves(parsed_logs: list[dict[str, Any]], output_dir: str | Path) -> list[str]:
    output_dir = Path(output_dir)
    outputs: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {"classification": [], "segmentation": []}
    for log in parsed_logs:
        for record in log["records"]:
            grouped.setdefault(record["task"], []).append(record)

    for task, records in grouped.items():
        if not records:
            continue
        records = sorted(records, key=lambda row: row["epoch"])
        epochs = [row["epoch"] for row in records]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
        axes[0].plot(epochs, [row["train_loss"] for row in records], label="train")
        axes[0].plot(epochs, [row["val_loss"] for row in records], label="val")
        axes[0].set_title(f"{task} loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        axes[1].plot(epochs, [row["train_acc"] for row in records], label="train acc")
        axes[1].plot(epochs, [row["val_acc"] for row in records], label="val acc")
        if task == "segmentation" and "val_miou" in records[0]:
            axes[1].plot(epochs, [row["val_miou"] for row in records], label="val mIoU")
        axes[1].set_title(f"{task} metrics")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("percent")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()

        fig.tight_layout()
        path = _prepare_output(output_dir / f"{task}_training_curves.png")
        fig.savefig(path)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def plot_checkpoint_summary(rows: list[dict[str, Any]], output_path: str | Path) -> str | None:
    rows = [row for row in rows if row.get("metric") is not None]
    if not rows:
        return None
    labels = [f"{row['task']}\ne{row['epoch']}" for row in rows]
    values = [row["metric"] for row in rows]
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.5), 4), dpi=150)
    ax.bar(labels, values, color="#3568a8")
    ax.set_ylabel("checkpoint metric")
    ax.set_title("saved checkpoint metrics")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = _prepare_output(output_path)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: str | Path, title: str) -> str:
    fig_size = max(8, min(16, len(labels) * 0.35))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=150)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("target")
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = _prepare_output(output_path)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_bar(values: list[float], labels: list[str], output_path: str | Path, title: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 4.5), dpi=150)
    ax.bar(labels, values, color="#2f8f6b")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1 if max(values or [1]) <= 1 else None)
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = _prepare_output(output_path)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_segmentation_samples(
    pointclouds: np.ndarray,
    groundtruths: np.ndarray,
    predictions: np.ndarray,
    color_map: np.ndarray,
    output_path: str | Path,
) -> str:
    count = min(len(pointclouds), len(groundtruths), len(predictions))
    if count == 0:
        raise ValueError("No segmentation samples to plot")
    fig = plt.figure(figsize=(8, count * 3.2), dpi=150)
    for i in range(count):
        for col, labels, title in (
            (1, groundtruths[i], "ground truth"),
            (2, predictions[i], "prediction"),
        ):
            ax = fig.add_subplot(count, 2, 2 * i + col, projection="3d")
            pc = pointclouds[i]
            ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=color_map[labels], s=3)
            ax.set_title(title if i == 0 else "")
            ax.set_xlim(-0.75, 0.75)
            ax.set_ylim(-0.75, 0.75)
            ax.set_zlim(-0.75, 0.75)
            ax.axis("off")
    fig.tight_layout()
    path = _prepare_output(output_path)
    fig.savefig(path)
    plt.close(fig)
    return str(path)

