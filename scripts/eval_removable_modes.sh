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

DATASET="${DATASET:-market1501}"
SEED="${SEED:-0}"
FORGET_ID="${FORGET_ID:-2}"
FORGET_DISTRACTOR_IDS="${FORGET_DISTRACTOR_IDS:-200}"
FORGET_DISTRACTOR_MODE="${FORGET_DISTRACTOR_MODE:-hard}"
TARGET_BASE_SCALE="${TARGET_BASE_SCALE:-0.0}"

TAG="${1:-removable_${DATASET}_id${FORGET_ID}_seed${SEED}_t1}"
OUT_DIR="${REID_OUTPUT_DIR}/removable/${TAG}"

BASE_CKPT="$OUT_DIR/base_ckpt.pth"
TARGET_MODULE="$OUT_DIR/target_module.pth"

TEACHER_DIR="${REID_OUTPUT_DIR}/transreid/market_teacher_r50"
TEACHER_CFG="$(cat "$TEACHER_DIR/teacher_cfg_path.txt")"

SPLIT_DIR="${REID_OUTPUT_DIR}/splits/${DATASET}/seed${SEED}"
PROBE_DIR="${REID_OUTPUT_DIR}/probes/${DATASET}/seed${SEED}_${FORGET_DISTRACTOR_MODE}${FORGET_DISTRACTOR_IDS}"

FEAT_TRAIN_NPZ="${FEAT_TRAIN_NPZ:-$REID_OUTPUT_DIR/mvp/auto_market1501_id2_seed0_t3/features/train_teacher.npz}"
if [[ ! -f "$FEAT_TRAIN_NPZ" ]]; then
  echo "[ERR] Missing feat_train_npz: $FEAT_TRAIN_NPZ"
  exit 2
fi

python "$REPO/scripts/eval_removable_modes.py" \
  --cfg "$TEACHER_CFG" \
  --base_ckpt "$BASE_CKPT" \
  --target_module "$TARGET_MODULE" \
  --forget_id "$FORGET_ID" \
  --probe_dir "$PROBE_DIR" \
  --split_dir "$SPLIT_DIR" \
  --feat_train_npz "$FEAT_TRAIN_NPZ" \
  --out_dir "$OUT_DIR" \
  --dataset "$DATASET" \
  --batch 128 \
  --num_workers 4 \
  --distractor_mode "$FORGET_DISTRACTOR_MODE" \
  --forget_distractor_ids "$FORGET_DISTRACTOR_IDS" \
  --neck_feat before \
  --target_base_scale "$TARGET_BASE_SCALE"

echo "[OK] Removable eval done -> $OUT_DIR"
