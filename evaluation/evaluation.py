import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.checkpoints import best_checkpoint, checkpoint_table  # noqa: E402
from modules.datasets import dataset_manifest  # noqa: E402
from modules.evaluators import classification_eval, segmentation_eval  # noqa: E402
from modules.plotting import plot_bar, plot_confusion_matrix  # noqa: E402
from modules.reporting import write_report_bundle  # noqa: E402
from modules.runtime import (  # noqa: E402
    cuda_summary,
    device_from_gpu,
    ensure_data_links,
    load_yaml,
    resolve_path,
    seed_everything,
    write_json,
)
from utils.logging_utils import configure_logger, make_log_dir  # noqa: E402


def load_config(path: str | Path | None) -> dict[str, Any]:
    default_path = PROJECT_ROOT / "configs" / "evaluation.yml"
    if path is None and default_path.exists():
        return load_yaml(default_path)
    if path is None:
        return {}
    return load_yaml(path)


def output_directory(config: dict[str, Any], override: str | None) -> Path:
    project = config.get("project", {})
    root = PROJECT_ROOT
    base = resolve_path(root, override, project.get("output_dir", "analysis_outputs/evaluation"))
    assert base is not None
    return base


def resolve_checkpoint(task: str, config: dict[str, Any], override: str | None, pointnet_dir: Path) -> Path:
    if override:
        return Path(override)
    task_cfg = config.get("tasks", {}).get(task, {})
    configured = task_cfg.get("checkpoint")
    if configured and configured != "auto":
        return Path(configured)
    directory = pointnet_dir / "checkpoints" / task
    ckpt = best_checkpoint(directory)
    if ckpt is None:
        raise FileNotFoundError(f"No {task} checkpoint found in {directory}")
    return ckpt


def selected_tasks(arg_task: str, config: dict[str, Any]) -> list[str]:
    names = ["classification", "segmentation"] if arg_task == "both" else [arg_task]
    task_cfg = config.get("tasks", {})
    return [name for name in names if task_cfg.get(name, {}).get("enabled", True)]


def summarize_for_markdown(result: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in result.items():
        if key in {"confusion_matrix", "samples", "class_names", "per_class_accuracy", "class_iou", "part_iou"}:
            continue
        if isinstance(value, float):
            summary[key] = round(value, 4)
        else:
            summary[key] = value
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PointNet checkpoints")
    parser.add_argument("--config", help="Evaluation YAML config")
    parser.add_argument("--task", choices=("classification", "segmentation", "both"), default="both")
    parser.add_argument("--classification_ckpt")
    parser.add_argument("--segmentation_ckpt")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--max_batches", type=int, help="Evaluate only the first N batches")
    parser.add_argument("--no-plots", action="store_true", help="Disable evaluation plots")
    args = parser.parse_args()

    config = load_config(args.config)
    project_cfg = config.get("project", {})
    runtime_cfg = config.get("runtime", {})
    output_cfg = config.get("outputs", {})

    pointnet_dir = resolve_path(PROJECT_ROOT, project_cfg.get("pointnet_dir", "sub-modules/pointnet"))
    data_dir = resolve_path(PROJECT_ROOT, project_cfg.get("data_dir", "data"))
    log_root = resolve_path(PROJECT_ROOT, project_cfg.get("log_dir", "logs"))
    output_dir = output_directory(config, args.output_dir)
    assert pointnet_dir is not None and data_dir is not None and log_root is not None

    run_dir = make_log_dir(log_root, run_id=output_dir.name if output_dir.name else None)
    logger = configure_logger("pointnet_evaluation", run_dir / "evaluation.log")
    logger.info("pointnet_dir=%s", pointnet_dir)
    logger.info("data_dir=%s", data_dir)
    logger.info("output_dir=%s", output_dir)

    ensure_data_links(pointnet_dir, data_dir)
    seed_everything(args.seed if args.seed is not None else runtime_cfg.get("seed", 1))
    device = device_from_gpu(args.gpu if args.gpu is not None else runtime_cfg.get("gpu", 0))
    batch_size = args.batch_size if args.batch_size is not None else runtime_cfg.get("batch_size", 128)
    max_batches = args.max_batches if args.max_batches is not None else runtime_cfg.get("max_batches")
    logger.info("device=%s batch_size=%s", device, batch_size)
    if max_batches is not None:
        logger.info("max_batches=%s", max_batches)
    logger.info("cuda=%s", cuda_summary(device))

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    results: dict[str, Any] = {
        "runtime": cuda_summary(device),
        "checkpoints": {},
        "tasks": {},
    }

    if output_cfg.get("dataset_manifest", True):
        logger.info("writing dataset manifest")
        write_json(output_dir / "dataset_manifest.json", dataset_manifest(data_dir))

    checkpoint_rows = []
    for task in ("classification", "segmentation"):
        checkpoint_rows.extend(checkpoint_table(pointnet_dir / "checkpoints" / task))
    results["checkpoints"]["available"] = checkpoint_rows

    for task in selected_tasks(args.task, config):
        logger.info("evaluating %s", task)
        if task == "classification":
            ckpt = resolve_checkpoint(task, config, args.classification_ckpt, pointnet_dir)
            result = classification_eval(pointnet_dir, data_dir, ckpt, batch_size, device, max_batches=max_batches)
            results["tasks"][task] = result
            if not args.no_plots and output_cfg.get("plots", True):
                matrix = np.asarray(result["confusion_matrix"], dtype=np.int64)
                plot_confusion_matrix(
                    matrix,
                    result["class_names"],
                    plots_dir / "classification_confusion_matrix.png",
                    "ModelNet40 confusion matrix",
                )
        elif task == "segmentation":
            ckpt = resolve_checkpoint(task, config, args.segmentation_ckpt, pointnet_dir)
            sample_count = int(output_cfg.get("sample_count", 0))
            result = segmentation_eval(
                pointnet_dir,
                data_dir,
                ckpt,
                batch_size,
                device,
                sample_count=sample_count,
                max_batches=max_batches,
            )
            results["tasks"][task] = result
            if not args.no_plots and output_cfg.get("plots", True):
                class_iou = result["class_iou"]
                plot_bar(
                    [row["miou"] for row in class_iou],
                    [row["class_name"] for row in class_iou],
                    plots_dir / "segmentation_class_miou.png",
                    "ShapeNet Part mIoU by object class",
                    "mIoU",
                )

    markdown_sections = {
        "Runtime": results["runtime"],
        "Checkpoints": [row["name"] for row in checkpoint_rows],
        "Results": {task: summarize_for_markdown(result) for task, result in results["tasks"].items()},
    }
    write_report_bundle(output_dir, "evaluation_report", results, markdown_sections)
    logger.info("wrote evaluation report to %s", output_dir)


if __name__ == "__main__":
    main()
