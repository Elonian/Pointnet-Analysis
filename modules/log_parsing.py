import re
from pathlib import Path
from typing import Any


CLASSIFICATION_RE = re.compile(
    r"train loss: (?P<train_loss>[0-9.]+) train acc: (?P<train_acc>[0-9.]+)% "
    r"\| val loss: (?P<val_loss>[0-9.]+) val acc: (?P<val_acc>[0-9.]+)%"
)
SEGMENTATION_RE = re.compile(
    r"train loss: (?P<train_loss>[0-9.]+) \| train acc: (?P<train_acc>[0-9.]+)% "
    r"\| val loss: (?P<val_loss>[0-9.]+) \| val acc: (?P<val_acc>[0-9.]+)% "
    r"\| val mIoU: (?P<val_miou>[0-9.]+)%"
)
TEST_CLASSIFICATION_RE = re.compile(r"test acc: (?P<test_acc>[0-9.]+)%$")
TEST_SEGMENTATION_RE = re.compile(r"test acc: (?P<test_acc>[0-9.]+)% \| test mIoU: (?P<test_miou>[0-9.]+)%")


def _float_fields(match: re.Match[str]) -> dict[str, float]:
    return {key: float(value) for key, value in match.groupdict().items()}


def parse_training_log(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    records = []
    tests = []
    for line in path.read_text(errors="replace").replace("\r", "\n").splitlines():
        cls = CLASSIFICATION_RE.search(line)
        if cls:
            record = _float_fields(cls)
            record.update({"task": "classification", "epoch": len([r for r in records if r["task"] == "classification"]) + 1})
            records.append(record)
            continue
        seg = SEGMENTATION_RE.search(line)
        if seg:
            record = _float_fields(seg)
            record.update({"task": "segmentation", "epoch": len([r for r in records if r["task"] == "segmentation"]) + 1})
            records.append(record)
            continue
        test_cls = TEST_CLASSIFICATION_RE.search(line)
        if test_cls:
            test = _float_fields(test_cls)
            test["task"] = "classification"
            tests.append(test)
            continue
        test_seg = TEST_SEGMENTATION_RE.search(line)
        if test_seg:
            test = _float_fields(test_seg)
            test["task"] = "segmentation"
            tests.append(test)

    return {
        "path": str(path),
        "records": records,
        "tests": tests,
    }


def collect_logs(log_roots: list[str | Path]) -> list[dict[str, Any]]:
    parsed = []
    seen: set[Path] = set()
    for root in log_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.glob("**/*.log")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            item = parse_training_log(path)
            if item["records"] or item["tests"]:
                parsed.append(item)
    return parsed

