#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Args: TAG LAMBDA_BASEONLY WARMUP_FRAC
TAG="${1:-removable_market1501_id2_seed0_t8v2_retfix_test}"
LAMBDA_BASEONLY="${2:-2.5}"
WARMUP_FRAC="${3:-0.5}"

FORGET_ID=2
SEED=0
TEACHER_DIR="${REID_OUTPUT_DIR}/transreid/market_teacher_r50"
TEACHER_CFG="$(cat "$TEACHER_DIR/teacher_cfg_path.txt")"
TEACHER_WEIGHTS="$(cat "$TEACHER_DIR/teacher_weights_path.txt")"
TEACHER_FEAT="${REID_OUTPUT_DIR}/mvp/auto_market1501_id2_seed0_t3/features/train_teacher.npz"
PROBE_DIR="${REID_OUTPUT_DIR}/probes/market1501/seed0_hard200"

# Params for gentler forgetting
EPOCHS=15
MAX_STEPS=200
BATCH=16
NUM_WORKERS=2

# Additional args from command line (optional)
LAMBDA_ADV="${4:-0.5}"
LAMBDA_FEAT="${5:-1.0}"

OUT_DIR="${REID_OUTPUT_DIR}/removable/${TAG}"

echo "=========================================="
echo "Running: $TAG"
echo "  lambda_baseonly=$LAMBDA_BASEONLY"
echo "  warmup_frac=$WARMUP_FRAC"
echo "  lambda_adv=$LAMBDA_ADV"
echo "  lambda_feat=$LAMBDA_FEAT"
echo "=========================================="

python "$REPO/scripts/train_removable_unlearn_v3.py" \
  --cfg "$TEACHER_CFG" \
  --weights "$TEACHER_WEIGHTS" \
  --out_dir "$OUT_DIR" \
  --dataset market1501 \
  --data_root "$REID_DATA_DIR" \
  --forget_id "$FORGET_ID" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --batch "$BATCH" \
  --num_instances 4 \
  --num_workers "$NUM_WORKERS" \
  --lr_base 1e-4 \
  --lr_module 5e-4 \
  --lr_head 5e-4 \
  --lr_disc 5e-4 \
  --lambda_id 1.0 \
  --lambda_tri 1.0 \
  --lambda_adv "$LAMBDA_ADV" \
  --lambda_baseonly_forget "$LAMBDA_BASEONLY" \
  --baseonly_forget_mode entropy \
  --lambda_feat_confuse "$LAMBDA_FEAT" \
  --lambda_feat_spread "$LAMBDA_FEAT" \
  --baseonly_warmup_frac "$WARMUP_FRAC" \
  --anti_warmup_frac "$WARMUP_FRAC" \
  --lambda_nontarget_consist 1.0 \
  --lambda_delta 0.01 \
  --lambda_delta_non 0.1 \
  --grl_lambda 1.0 \
  --module_hidden 512 \
  --module_dropout 0.1 \
  --detach_base_for_target 1 \
  --detach_base_in_delta 1 \
  --target_base_scale 1.0 \
  --neck_feat before \
  --freeze_base 0 \
  --probe_dir "$PROBE_DIR"

echo "[OK] Completed: $TAG"
