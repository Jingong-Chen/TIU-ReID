#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "reid_unlearning" ]]; then
  echo "[ERR] Please conda activate reid_unlearning"
  exit 2
fi

FORGET_ID=2
SEED=0
TEACHER_DIR="${REID_OUTPUT_DIR}/transreid/market_teacher_r50"
TEACHER_CFG="$(cat "$TEACHER_DIR/teacher_cfg_path.txt")"
TEACHER_WEIGHTS="$(cat "$TEACHER_DIR/teacher_weights_path.txt")"
PROBE_DIR="${REID_OUTPUT_DIR}/probes/market1501/seed0_hard200"

# ========== Step 1.1: BASE_TEACHER ==========
echo ""
echo "=========================================="
echo "Step 1.1: BASE_TEACHER baseline"
echo "=========================================="
BASE_DIR="${REID_OUTPUT_DIR}/removable_baselines/base_teacher_market1501_id2_seed0"
python "$REPO/scripts/eval_baseline_probes.py" \
  --cfg "$TEACHER_CFG" \
  --weights "$TEACHER_WEIGHTS" \
  --probe_dir "$PROBE_DIR" \
  --out_dir "$BASE_DIR" \
  --mode without \
  --batch 128 \
  --num_workers 4

# ========== Step 1.2: WITH_ONLY training ==========
echo ""
echo "=========================================="
echo "Step 1.2: WITH_ONLY training"
echo "=========================================="
WITH_ONLY_DIR="${REID_OUTPUT_DIR}/removable_baselines/with_only_market1501_id2_seed0"
python "$REPO/scripts/train_with_only.py" \
  --cfg "$TEACHER_CFG" \
  --weights "$TEACHER_WEIGHTS" \
  --out_dir "$WITH_ONLY_DIR" \
  --dataset market1501 \
  --data_root "$REID_DATA_DIR" \
  --forget_id "$FORGET_ID" \
  --seed "$SEED" \
  --epochs 8 \
  --batch 16 \
  --num_instances 4 \
  --num_workers 2 \
  --lr_module 3e-4 \
  --lr_head 3e-4 \
  --lambda_id 1.0 \
  --lambda_tri 1.0 \
  --module_hidden 512 \
  --module_dropout 0.1 \
  --target_base_scale 1.0 \
  --neck_feat before

# ========== Step 1.2: WITH_ONLY evaluation ==========
echo ""
echo "=========================================="
echo "Step 1.2: WITH_ONLY evaluation (WITH_MODULE)"
echo "=========================================="
python "$REPO/scripts/eval_baseline_probes.py" \
  --cfg "$TEACHER_CFG" \
  --weights "$WITH_ONLY_DIR/base_ckpt.pth" \
  --probe_dir "$PROBE_DIR" \
  --out_dir "$WITH_ONLY_DIR" \
  --target_module "$WITH_ONLY_DIR/target_module.pth" \
  --mode with \
  --target_pid "$FORGET_ID" \
  --target_base_scale 1.0 \
  --batch 128 \
  --num_workers 4

# Rename outputs
mv "$WITH_ONLY_DIR/metrics_base_retain.json" "$WITH_ONLY_DIR/metrics_with_only_with_retain.json"
mv "$WITH_ONLY_DIR/metrics_base_forget.json" "$WITH_ONLY_DIR/metrics_with_only_with_forget.json"
mv "$WITH_ONLY_DIR/summary_base.json" "$WITH_ONLY_DIR/summary_with_only_with.json"

echo ""
echo "=========================================="
echo "Step 1.2: WITH_ONLY evaluation (WITHOUT_MODULE)"
echo "=========================================="
python "$REPO/scripts/eval_baseline_probes.py" \
  --cfg "$TEACHER_CFG" \
  --weights "$WITH_ONLY_DIR/base_ckpt.pth" \
  --probe_dir "$PROBE_DIR" \
  --out_dir "$WITH_ONLY_DIR" \
  --mode without \
  --batch 128 \
  --num_workers 4

# Rename outputs
mv "$WITH_ONLY_DIR/metrics_base_retain.json" "$WITH_ONLY_DIR/metrics_with_only_without_retain.json"
mv "$WITH_ONLY_DIR/metrics_base_forget.json" "$WITH_ONLY_DIR/metrics_with_only_without_forget.json"
mv "$WITH_ONLY_DIR/summary_base.json" "$WITH_ONLY_DIR/summary_with_only_without.json"

# Combine summaries
python -c "
import json
from pathlib import Path

with_only_dir = Path('$WITH_ONLY_DIR')
with_summary = json.loads((with_only_dir / 'summary_with_only_with.json').read_text())
without_summary = json.loads((with_only_dir / 'summary_with_only_without.json').read_text())

combined = {
    'with': with_summary,
    'without': without_summary,
}
(with_only_dir / 'summary_with_only.json').write_text(json.dumps(combined, indent=2) + '\n')
print('[OK] Combined summary saved')
"

echo ""
echo "=========================================="
echo "All baselines completed!"
echo "=========================================="
