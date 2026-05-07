import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common import ensure_data_links, load_yaml, resolve_path  # noqa: E402
from utils.logging_utils import configure_logger, make_log_dir  # noqa: E402


def bool_override(current: bool, enabled: bool, disabled: bool) -> bool:
    if enabled:
        return True
    if disabled:
        return False
    return current


def task_command(
    python_bin: str,
    task_cfg: dict,
    gpu: int,
    amp: bool,
    save_checkpoints: bool,
) -> list[str]:
    cmd = [
        python_bin,
        "-u",
        task_cfg["script"],
        "--epochs",
        str(task_cfg["epochs"]),
        "--batch_size",
        str(task_cfg["batch_size"]),
        "--lr",
        str(task_cfg["lr"]),
        "--seed",
        str(task_cfg["seed"]),
        "--gpu",
        str(gpu),
    ]
    if amp:
        cmd.append("--amp")
    if not save_checkpoints:
        cmd.append("--no_save")
    return cmd


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> None:
    print(f"\n$ {' '.join(cmd)}")
    print(f"log: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def selected_tasks(config: dict, task_name: str) -> list[tuple[str, dict]]:
    tasks = config["tasks"]
    if task_name == "both":
        names = ("classification", "segmentation")
    else:
        names = (task_name,)
    return [(name, tasks[name]) for name in names if tasks[name].get("enabled", True)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PointNet assignment training")
    parser.add_argument("--config", default="configs/training.yml")
    parser.add_argument("--task", choices=("classification", "segmentation", "both"), default="both")
    parser.add_argument("--epochs", type=int, help="Override epochs for selected task(s)")
    parser.add_argument("--batch_size", type=int, help="Override batch size for selected task(s)")
    parser.add_argument("--lr", type=float, help="Override learning rate for selected task(s)")
    parser.add_argument("--gpu", type=int, help="Override GPU index; use -1 for CPU")
    parser.add_argument("--amp", action="store_true", help="Force mixed precision on")
    parser.add_argument("--no-amp", action="store_true", help="Force mixed precision off")
    parser.add_argument("--no-save", action="store_true", help="Disable checkpoint saving")
    parser.add_argument("--install-requirements", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = ROOT
    config = load_yaml(resolve_path(root, args.config))

    project_cfg = config["project"]
    runtime_cfg = config["runtime"]
    pointnet_dir = resolve_path(root, project_cfg["pointnet_dir"])
    data_dir = resolve_path(root, project_cfg["data_dir"])
    log_dir = resolve_path(root, project_cfg["log_dir"])
    python_bin = runtime_cfg.get("python", sys.executable)
    run_dir = make_log_dir(log_dir)
    logger = configure_logger("pointnet_training", run_dir / "runner.log")

    if project_cfg.get("link_data", True):
        ensure_data_links(pointnet_dir, data_dir)

    if args.install_requirements or runtime_cfg.get("install_requirements", False):
        run_command(
            [python_bin, "-m", "pip", "install", "-r", str(root / "requirements.txt")],
            cwd=root,
            log_path=run_dir / "install_requirements.log",
        )

    gpu = args.gpu if args.gpu is not None else runtime_cfg.get("gpu", 0)
    amp = bool_override(runtime_cfg.get("amp", False), args.amp, args.no_amp)
    save_checkpoints = runtime_cfg.get("save_checkpoints", True) and not args.no_save
    logger.info("pointnet_dir=%s", pointnet_dir)
    logger.info("data_dir=%s", data_dir)
    logger.info("log_dir=%s", run_dir)
    logger.info("task=%s gpu=%s amp=%s save_checkpoints=%s", args.task, gpu, amp, save_checkpoints)

    for name, task_cfg in selected_tasks(config, args.task):
        task_cfg = dict(task_cfg)
        if args.epochs is not None:
            task_cfg["epochs"] = args.epochs
        if args.batch_size is not None:
            task_cfg["batch_size"] = args.batch_size
        if args.lr is not None:
            task_cfg["lr"] = args.lr

        cmd = task_command(python_bin, task_cfg, gpu, amp, save_checkpoints)
        log_path = run_dir / f"{name}.log"
        if args.dry_run:
            print(f"{name}: cd {pointnet_dir} && {' '.join(cmd)}")
            print(f"{name} log: {log_path}")
            continue
        logger.info("running %s", name)
        run_command(cmd, cwd=pointnet_dir, log_path=log_path)
        logger.info("finished %s", name)


if __name__ == "__main__":
    main()
