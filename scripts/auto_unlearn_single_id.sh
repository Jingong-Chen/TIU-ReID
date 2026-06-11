#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
export PYTHONUNBUFFERED=1
exec 2>&1

if [[ "${CONDA_DEFAULT_ENV:-}" != "reid_unlearning" ]]; then
  echo "[ERR] Please conda activate reid_unlearning"
  exit 2
fi

DATASET="${1:-market1501}"
FORGET_ID="${2:-2}"
SEED="${3:-0}"
MAX_TRIALS="${MAX_TRIALS:-3}"
FORGET_DISTRACTOR_IDS="${FORGET_DISTRACTOR_IDS:-200}"
TARGET_RETAIN="${TARGET_RETAIN:-0.70}"
TARGET_FORGET="${TARGET_FORGET:-0.10}"

UNLEARN_FORGET_MODE="${UNLEARN_FORGET_MODE:-negrad_cos}"
UNLEARN_LAMBDA_DISTILL="${UNLEARN_LAMBDA_DISTILL:-10.0}"
UNLEARN_LAMBDA_FORGET="${UNLEARN_LAMBDA_FORGET:-0.1}"
UNLEARN_LAMBDA_COLLAPSE="${UNLEARN_LAMBDA_COLLAPSE:-0.0}"
UNLEARN_DISC_STEPS="${UNLEARN_DISC_STEPS:-1}"

export DATASET FORGET_ID SEED MAX_TRIALS
export FORGET_DISTRACTOR_IDS TARGET_RETAIN TARGET_FORGET
export UNLEARN_FORGET_MODE UNLEARN_LAMBDA_DISTILL UNLEARN_LAMBDA_FORGET UNLEARN_LAMBDA_COLLAPSE UNLEARN_DISC_STEPS

echo "[AUTO] dataset=$DATASET forget_id=$FORGET_ID seed=$SEED max_trials=$MAX_TRIALS"
echo "[AUTO] targets retain_mAP>=$TARGET_RETAIN forget_mAP<=$TARGET_FORGET"

for trial in $(seq 1 "$MAX_TRIALS"); do
  TAG="auto_${DATASET}_id${FORGET_ID}_seed${SEED}_t${trial}"
  OUT="$REID_OUTPUT_DIR/mvp/$TAG"
  mkdir -p "$OUT"

  echo "[AUTO] trial=$trial tag=$TAG"
  echo "[AUTO] params mode=$UNLEARN_FORGET_MODE distill=$UNLEARN_LAMBDA_DISTILL forget=$UNLEARN_LAMBDA_FORGET collapse=$UNLEARN_LAMBDA_COLLAPSE distractors=$FORGET_DISTRACTOR_IDS"

  FORGET_ID="$FORGET_ID" FORGET_DISTRACTOR_IDS="$FORGET_DISTRACTOR_IDS" \
  UNLEARN_FORGET_MODE="$UNLEARN_FORGET_MODE" \
  UNLEARN_LAMBDA_DISTILL="$UNLEARN_LAMBDA_DISTILL" \
  UNLEARN_LAMBDA_FORGET="$UNLEARN_LAMBDA_FORGET" \
  UNLEARN_LAMBDA_COLLAPSE="$UNLEARN_LAMBDA_COLLAPSE" \
  UNLEARN_DISC_STEPS="$UNLEARN_DISC_STEPS" \
    bash "$REPO/scripts/run_mvp_pipeline.sh" "$DATASET" "$SEED" 0.2 "$TAG" \
    | tee "$OUT/pipeline.log"

  python "$REPO/scripts/summarize_mvp.py" \
    --mvp_dir "$OUT" \
    --teacher_dir "$REID_OUTPUT_DIR/transreid/market_teacher_r50"

  # Evaluate and propose next params
  NEXT_JSON="$OUT/auto_next.json"
  export OUT_DIR="$OUT"
  export NEXT_JSON
  python - <<'PY'
import json, os, sys
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
summary = json.loads((out_dir / "summary.json").read_text())
retain = float(summary["unlearn_retain"]["mAP"])
forget = float(summary["unlearn_forget"]["mAP"])

target_retain = float(os.environ["TARGET_RETAIN"])
target_forget = float(os.environ["TARGET_FORGET"])

mode = os.environ["UNLEARN_FORGET_MODE"]
distill = float(os.environ["UNLEARN_LAMBDA_DISTILL"])
forget_w = float(os.environ["UNLEARN_LAMBDA_FORGET"])
collapse = float(os.environ["UNLEARN_LAMBDA_COLLAPSE"])
distr = int(os.environ["FORGET_DISTRACTOR_IDS"])

ok = retain >= target_retain and forget <= target_forget
print(f"[AUTO] retain_mAP={retain:.4f} forget_mAP={forget:.4f} ok={ok}")

if ok:
    (out_dir / "auto_status.txt").write_text("success\n")
    sys.exit(0)

if retain < target_retain:
    distill = min(distill * 2.0, 50.0)
    forget_w = max(forget_w * 0.5, 0.01)

if forget > target_forget:
    forget_w = min(forget_w * 2.0, 5.0)
    distr = min(distr + 100, 1000)

next_params = {
    "UNLEARN_FORGET_MODE": mode,
    "UNLEARN_LAMBDA_DISTILL": distill,
    "UNLEARN_LAMBDA_FORGET": forget_w,
    "UNLEARN_LAMBDA_COLLAPSE": collapse,
    "FORGET_DISTRACTOR_IDS": distr,
}

(out_dir / "auto_status.txt").write_text("retry\n")
(out_dir / "auto_next.json").write_text(json.dumps(next_params, indent=2) + "\n")
print("[AUTO] next params:", next_params)
PY

  if [[ -f "$OUT/auto_status.txt" ]] && grep -q "success" "$OUT/auto_status.txt"; then
    echo "[AUTO] success achieved at trial $trial"
    break
  fi

  if [[ -f "$NEXT_JSON" ]]; then
    eval "$(python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["NEXT_JSON"])
data = json.loads(p.read_text())
for k, v in data.items():
    print(f'export {k}="{v}"')
PY
)"
  else
    echo "[AUTO] missing next params; stopping"
    break
  fi
done

