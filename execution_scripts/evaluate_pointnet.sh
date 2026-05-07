#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT_DIR/configs/evaluation.yml}"

cd "$ROOT_DIR"
python3 "$ROOT_DIR/evaluation/evaluation.py" --config "$CONFIG" "$@"

