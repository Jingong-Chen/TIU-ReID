#!/usr/bin/env bash
# Usage: source scripts/env.sh
set -euo pipefail

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REID_DATA_DIR="$PROJECT_ROOT/data"
export REID_OUTPUT_DIR="$PROJECT_ROOT/output"

mkdir -p "$REID_DATA_DIR" "$REID_OUTPUT_DIR"

echo "[OK] PROJECT_ROOT=$PROJECT_ROOT"
echo "[OK] REID_DATA_DIR=$REID_DATA_DIR"
echo "[OK] REID_OUTPUT_DIR=$REID_OUTPUT_DIR"


