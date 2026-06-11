#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
export PYTHONUNBUFFERED=1

if [[ "${CONDA_DEFAULT_ENV:-}" != "reid_unlearning" ]]; then
  echo "[ERR] Please conda activate reid_unlearning"
  exit 2
fi

TR="$REPO/third_party/TransReID"
cd "$TR"

DATASET="${1:-market1501}"
TAG="${2:-market_teacher_r50}"
OUT="$REID_OUTPUT_DIR/transreid/$TAG"
mkdir -p "$OUT"

# Build a read-only dataset mirror under output/ (no data modification)
TMP_DATA="$REID_OUTPUT_DIR/transreid_data"
mkdir -p "$TMP_DATA"
if [[ "${DATASET,,}" == "market1501" ]]; then
  ln -sfn "$REID_DATA_DIR/market1501/Market-1501-v15.09.15" "$TMP_DATA/market1501"
elif [[ "${DATASET,,}" == "dukemtmc-reid" ]]; then
  ln -sfn "$REID_DATA_DIR/dukemtmc-reid/DukeMTMC-reID" "$TMP_DATA/dukemtmc"
else
  echo "[ERR] Unsupported dataset: $DATASET"
  exit 2
fi

# Choose config per README (TransReID stride for Market/Duke)
if [[ "${DATASET,,}" == "market1501" ]]; then
  CFG="$TR/configs/Market/vit_transreid_stride.yml"
elif [[ "${DATASET,,}" == "dukemtmc-reid" || "${DATASET,,}" == "dukemtmc" ]]; then
  CFG="$TR/configs/DukeMTMC/vit_transreid_stride.yml"
else
  echo "[ERR] Unsupported dataset for config: $DATASET"
  exit 2
fi

CFG_ABS="$CFG"
echo "[OK] Using config: $CFG_ABS"
echo "[OK] Output dir: $OUT"
echo "[OK] DATA_ROOT: $TMP_DATA"

# Check whether CUDA arch is supported by current PyTorch build
USE_CPU=0
if python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit(1)
archs = set(torch.cuda.get_arch_list())
cap = torch.cuda.get_device_capability()
tag = f"sm_{cap[0]}{cap[1]}"
sys.exit(0 if tag in archs else 1)
PY
then
  USE_CPU=0
else
  USE_CPU=1
fi

EXTRA_OPTS=()
MAX_EPOCHS=120
IMS_BATCH=64
WORKERS=8
if [[ "$USE_CPU" -eq 1 ]]; then
  echo "[WARN] CUDA kernels not available; fallback to CPU."
  export CUDA_VISIBLE_DEVICES=""
  EXTRA_OPTS+=(MODEL.DEVICE cpu)
  EXTRA_OPTS+=(SOLVER.WARMUP_EPOCHS 0)
  MAX_EPOCHS=1
  IMS_BATCH=8
  WORKERS=2
else
  EXTRA_OPTS+=(MODEL.DEVICE_ID "'0'")
fi

# Ensure ImageNet pretrain exists (vit_base_patch16_224)
PRETRAIN_PATH="$REID_OUTPUT_DIR/pretrained/jx_vit_base_p16_224-80ecf9dd.pth"
PRETRAIN_URL="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
if [[ ! -f "$PRETRAIN_PATH" ]]; then
  mkdir -p "$(dirname "$PRETRAIN_PATH")"
  echo "[OK] Downloading ImageNet pretrain (vit_base_patch16_224)..."
  python - <<PY
import os, urllib.request
url = "${PRETRAIN_URL}"
path = "${PRETRAIN_PATH}"
os.makedirs(os.path.dirname(path), exist_ok=True)
urllib.request.urlretrieve(url, path)
print("[OK] downloaded ->", path)
PY
fi
if [[ ! -f "$PRETRAIN_PATH" ]]; then
  echo "[ERR] Pretrain weights not found: $PRETRAIN_PATH"
  exit 2
fi
echo "[OK] PRETRAIN_PATH: $PRETRAIN_PATH"

python train.py --config_file "$CFG_ABS" \
  OUTPUT_DIR "$OUT" \
  DATASETS.NAMES "('${DATASET,,}')" \
  DATASETS.ROOT_DIR "$TMP_DATA" \
  MODEL.PRETRAIN_CHOICE "imagenet" \
  MODEL.PRETRAIN_PATH "$PRETRAIN_PATH" \
  SOLVER.MAX_EPOCHS "$MAX_EPOCHS" \
  SOLVER.IMS_PER_BATCH "$IMS_BATCH" \
  SOLVER.CHECKPOINT_PERIOD 1 \
  SOLVER.EVAL_PERIOD 9999 \
  SOLVER.LOG_PERIOD 50 \
  DATALOADER.NUM_WORKERS "$WORKERS" \
  "${EXTRA_OPTS[@]}"

# pick last checkpoint
WEIGHTS="$(ls -1t "$OUT"/*.pth 2>/dev/null | head -n 1 || true)"
if [[ -z "$WEIGHTS" ]]; then
  echo "[ERR] No weight file found under $OUT."
  exit 2
fi

echo "$CFG_ABS" > "$OUT/teacher_cfg_path.txt"
echo "$WEIGHTS" > "$OUT/teacher_weights_path.txt"

echo "[OK] Teacher training done."
echo "[OK] teacher_cfg_path.txt -> $CFG_ABS"
echo "[OK] teacher_weights_path.txt -> $WEIGHTS"

# Eval-only sanity (non-blocking)
EVAL_OUT="$REID_OUTPUT_DIR/transreid/${TAG}_eval"
mkdir -p "$EVAL_OUT"
set +e
python test.py --config_file "$CFG_ABS" \
  OUTPUT_DIR "$EVAL_OUT" \
  DATASETS.NAMES "('${DATASET,,}')" \
  DATASETS.ROOT_DIR "$TMP_DATA" \
  TEST.WEIGHT "$WEIGHTS" \
  MODEL.PRETRAIN_CHOICE "imagenet" \
  MODEL.PRETRAIN_PATH "$PRETRAIN_PATH" \
  "${EXTRA_OPTS[@]}"
EVAL_RC=$?
set -e
if [[ "$EVAL_RC" -eq 0 ]]; then
  echo "success" > "$EVAL_OUT/eval_status.txt"
  echo "[OK] Eval-only succeeded -> $EVAL_OUT"
else
  echo "failed" > "$EVAL_OUT/eval_status.txt"
  echo "[WARN] Eval-only failed -> $EVAL_OUT"
fi


