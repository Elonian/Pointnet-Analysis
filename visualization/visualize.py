import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.checkpoints import best_checkpoint, checkpoint_table  # noqa: E402
from modules.evaluators import classification_eval, segmentation_eval  # noqa: E402
from modules.log_parsing import collect_logs  # noqa: E402
from modules.plotting import (  # noqa: E402
    plot_bar,
    plot_checkpoint_summary,
    plot_confusion_matrix,
    plot_segmentation_samples,
    plot_training_curves,
)
from modules.reporting import write_report_bundle  # noqa: E402
from modules.runtime import (  # noqa: E402
    cuda_summary,
    device_from_gpu,
    ensure_data_links,
    load_yaml,
    resolve_path,
    seed_everything,
)
from utils.logging_utils import configure_logger, make_log_dir  # noqa: E402


def load_config(path: str | Path | None) -> dict[str, Any]:
    default_path = PROJECT_ROOT / "configs" / "visualization.yml"
    if path is None and default_path.exists():
        return load_yaml(default_path)
    if path is None:
        return {}
    return load_yaml(path)


def resolve_checkpoint(task: str, pointnet_dir: Path, configured: str | None) -> Path | None:
    if configured and configured != "auto":
        return Path(configured)
    return best_checkpoint(pointnet_dir / "checkpoints" / task)


def shape_part_color_map(data_dir: Path) -> np.ndarray:
    path = data_dir / "shapenet_part_seg_hdf5_data" / "part_color_mapping.json"
    with path.open("r", encoding="utf-8") as f:
        return np.asarray(json.load(f), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PointNet analysis visualizations")
    parser.add_argument("--config", help="Visualization YAML config")
    parser.add_argument("--task", choices=("classification", "segmentation", "both"), default="both")
    parser.add_argument("--output_dir")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_batches", type=int, help="Evaluate only the first N batches")
    parser.add_argument("--skip-eval", action="store_true", help="Only plot existing logs/checkpoints")
    args = parser.parse_args()

    config = load_config(args.config)
    project_cfg = config.get("project", {})
    runtime_cfg = config.get("runtime", {})
    viz_cfg = config.get("visualizations", {})

    pointnet_dir = resolve_path(PROJECT_ROOT, project_cfg.get("pointnet_dir", "sub-modules/pointnet"))
    data_dir = resolve_path(PROJECT_ROOT, project_cfg.get("data_dir", "data"))
    log_dir = resolve_path(PROJECT_ROOT, project_cfg.get("log_dir", "logs"))
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir, project_cfg.get("output_dir", "analysis_outputs/visualization"))
    assert pointnet_dir is not None and data_dir is not None and log_dir is not None and output_dir is not None

    run_dir = make_log_dir(log_dir, run_id=output_dir.name if output_dir.name else None)
    logger = configure_logger("pointnet_visualization", run_dir / "visualization.log")
    logger.info("pointnet_dir=%s", pointnet_dir)
    logger.info("data_dir=%s", data_dir)
    logger.info("output_dir=%s", output_dir)
    ensure_data_links(pointnet_dir, data_dir)

    seed_everything(args.seed if args.seed is not None else runtime_cfg.get("seed", 1))
    device = device_from_gpu(args.gpu if args.gpu is not None else runtime_cfg.get("gpu", 0))
    batch_size = args.batch_size if args.batch_size is not None else runtime_cfg.get("batch_size", 128)
    max_batches = args.max_batches if args.max_batches is not None else runtime_cfg.get("max_batches")
    logger.info("runtime=%s", cuda_summary(device))
    if max_batches is not None:
        logger.info("max_batches=%s", max_batches)

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    artifacts: list[str] = []
    payload: dict[str, Any] = {
        "runtime": cuda_summary(device),
        "artifacts": artifacts,
        "logs": [],
        "evaluations": {},
    }

    if viz_cfg.get("training_curves", True):
        roots = [resolve_path(PROJECT_ROOT, root) for root in project_cfg.get("log_roots", ["training_runs", "logs"])]
        parsed_logs = collect_logs([root for root in roots if root is not None])
        payload["logs"] = parsed_logs
        logger.info("parsed %s logs for curves", len(parsed_logs))
        artifacts.extend(plot_training_curves(parsed_logs, plots_dir))

    if viz_cfg.get("checkpoint_summary", True):
        rows = []
        for task in ("classification", "segmentation"):
            rows.extend(checkpoint_table(pointnet_dir / "checkpoints" / task))
        payload["checkpoints"] = rows
        path = plot_checkpoint_summary(rows, plots_dir / "checkpoint_summary.png")
        if path:
            artifacts.append(path)

    run_eval = bool(viz_cfg.get("evaluation_plots", True)) and not args.skip_eval
    selected = ["classification", "segmentation"] if args.task == "both" else [args.task]
    if run_eval:
        if "classification" in selected and viz_cfg.get("classification_confusion", True):
            ckpt = resolve_checkpoint(
                "classification",
                pointnet_dir,
                config.get("tasks", {}).get("classification", {}).get("checkpoint"),
            )
            if ckpt is None:
                logger.warning("no classification checkpoint found")
            else:
                logger.info("evaluating classification checkpoint=%s", ckpt)
                result = classification_eval(pointnet_dir, data_dir, ckpt, batch_size, device, max_batches=max_batches)
                payload["evaluations"]["classification"] = result
                artifacts.append(
                    plot_confusion_matrix(
                        np.asarray(result["confusion_matrix"], dtype=np.int64),
                        result["class_names"],
                        plots_dir / "classification_confusion_matrix.png",
                        "ModelNet40 confusion matrix",
                    )
                )

        if "segmentation" in selected:
            ckpt = resolve_checkpoint(
                "segmentation",
                pointnet_dir,
                config.get("tasks", {}).get("segmentation", {}).get("checkpoint"),
            )
            if ckpt is None:
                logger.warning("no segmentation checkpoint found")
            else:
                sample_count = int(viz_cfg.get("segmentation_sample_count", 6))
                logger.info("evaluating segmentation checkpoint=%s", ckpt)
                result = segmentation_eval(
                    pointnet_dir,
                    data_dir,
                    ckpt,
                    batch_size,
                    device,
                    sample_count=sample_count,
                    max_batches=max_batches,
                )
                payload["evaluations"]["segmentation"] = result
                if viz_cfg.get("segmentation_iou", True):
                    artifacts.append(
                        plot_bar(
                            [row["miou"] for row in result["class_iou"]],
                            [row["class_name"] for row in result["class_iou"]],
                            plots_dir / "segmentation_class_miou.png",
                            "ShapeNet Part mIoU by object class",
                            "mIoU",
                        )
                    )
                if viz_cfg.get("segmentation_samples", True) and result["samples"]["pointclouds"]:
                    artifacts.append(
                        plot_segmentation_samples(
                            np.asarray(result["samples"]["pointclouds"], dtype=np.float32),
                            np.asarray(result["samples"]["groundtruths"], dtype=np.int64),
                            np.asarray(result["samples"]["predictions"], dtype=np.int64),
                            shape_part_color_map(data_dir),
                            plots_dir / "segmentation_prediction_samples.png",
                        )
                    )

    markdown_sections = {
        "Runtime": payload["runtime"],
        "Artifacts": artifacts,
        "Evaluations": list(payload["evaluations"].keys()),
    }
    write_report_bundle(output_dir, "visualization_report", payload, markdown_sections)
    logger.info("wrote visualization report to %s", output_dir)


if __name__ == "__main__":
    main()
