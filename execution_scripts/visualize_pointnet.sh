#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT_DIR/configs/visualization.yml}"

cd "$ROOT_DIR"
python3 "$ROOT_DIR/visualization/visualize.py" --config "$CONFIG" "$@"

