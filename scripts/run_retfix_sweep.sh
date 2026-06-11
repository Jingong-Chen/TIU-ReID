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

# ========== Fixed Configuration ==========
FORGET_ID=2
SEED=0
TEACHER_DIR="${REID_OUTPUT_DIR}/transreid/market_teacher_r50"
TEACHER_CFG="$(cat "$TEACHER_DIR/teacher_cfg_path.txt")"
TEACHER_WEIGHTS="$(cat "$TEACHER_DIR/teacher_weights_path.txt")"
TEACHER_FEAT="${REID_OUTPUT_DIR}/mvp/auto_market1501_id2_seed0_t3/features/train_teacher.npz"
PROBE_DIR="${REID_OUTPUT_DIR}/probes/market1501/seed0_hard200"
SPLIT_DIR="${REID_OUTPUT_DIR}/splits/market1501/seed0"

# Training params
EPOCHS=12
MAX_STEPS=300
BATCH=16
NUM_WORKERS=4

# Fixed params from t8v2_diag_v2
LAMBDA_ID=1.0
LAMBDA_TRI=1.0
LAMBDA_DELTA=0.001
LAMBDA_DELTA_NON=0.1
TARGET_BASE_SCALE=1.0
MODULE_HIDDEN=1024
MODULE_DROPOUT=0.1
LR_BASE=1e-4
LR_MODULE=5e-4
LR_HEAD=5e-4
LR_DISC=5e-4

# Fixed consistency
LAMBDA_NONTARGET_CONSIST=1.0

# Base values from t8v2_diag_v2 (to be reduced)
BASE_LAMBDA_BASEONLY=5.0  # was 5.0
BASE_LAMBDA_ADV=2.0       # was 2.0
BASE_LAMBDA_FEAT_CONFUSE=5.0
BASE_LAMBDA_FEAT_SPREAD=5.0

run_experiment() {
    local TAG=$1
    local LAMBDA_BASEONLY=$2
    local WARMUP_FRAC=$3
    
    echo ""
    echo "=========================================="
    echo "Running experiment: $TAG"
    echo "  lambda_baseonly_forget=$LAMBDA_BASEONLY"
    echo "  warmup_frac=$WARMUP_FRAC"
    echo "=========================================="
    
    OUT_DIR="${REID_OUTPUT_DIR}/removable/${TAG}"
    
    python "$REPO/scripts/train_removable_unlearn_v3.py" \
      --cfg "$TEACHER_CFG" \
      --weights "$TEACHER_WEIGHTS" \
      --out_dir "$OUT_DIR" \
      --dataset market1501 \
      --data_root "$REID_DATA_DIR" \
      --teacher_feat_npz "$TEACHER_FEAT" \
      --forget_id "$FORGET_ID" \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      --max_steps "$MAX_STEPS" \
      --batch "$BATCH" \
      --num_instances 4 \
      --num_workers "$NUM_WORKERS" \
      --lr_base "$LR_BASE" \
      --lr_module "$LR_MODULE" \
      --lr_head "$LR_HEAD" \
      --lr_disc "$LR_DISC" \
      --lambda_id "$LAMBDA_ID" \
      --lambda_tri "$LAMBDA_TRI" \
      --lambda_adv "$BASE_LAMBDA_ADV" \
      --lambda_baseonly_forget "$LAMBDA_BASEONLY" \
      --baseonly_forget_mode entropy \
      --lambda_feat_confuse "$BASE_LAMBDA_FEAT_CONFUSE" \
      --lambda_feat_spread "$BASE_LAMBDA_FEAT_SPREAD" \
      --baseonly_warmup_frac "$WARMUP_FRAC" \
      --anti_warmup_frac "$WARMUP_FRAC" \
      --lambda_nontarget_consist "$LAMBDA_NONTARGET_CONSIST" \
      --lambda_delta "$LAMBDA_DELTA" \
      --lambda_delta_non "$LAMBDA_DELTA_NON" \
      --grl_lambda 1.0 \
      --module_hidden "$MODULE_HIDDEN" \
      --module_dropout "$MODULE_DROPOUT" \
      --detach_base_for_target 1 \
      --detach_base_in_delta 1 \
      --target_base_scale "$TARGET_BASE_SCALE" \
      --neck_feat before \
      --freeze_base 0 \
      --probe_dir "$PROBE_DIR" \
      --split_dir "$SPLIT_DIR"
    
    echo "[OK] Completed: $TAG"
}

# ========== Run 4 experiments ==========
# A: (baseonly=1/2, warmup=0.5)
run_experiment "removable_market1501_id2_seed0_t8v2_retfix_A" "2.5" "0.5"

# B: (baseonly=1/2, warmup=0.7)
run_experiment "removable_market1501_id2_seed0_t8v2_retfix_B" "2.5" "0.7"

# C: (baseonly=1/4, warmup=0.5)
run_experiment "removable_market1501_id2_seed0_t8v2_retfix_C" "1.25" "0.5"

# D: (baseonly=1/4, warmup=0.7)
run_experiment "removable_market1501_id2_seed0_t8v2_retfix_D" "1.25" "0.7"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
