#!/usr/bin/env bash
# 对 multi-seed 的 5 个 variant 各 run 跑 eval_test_split，写入 scorecard test_without_mAP。
# 从项目根或 reid_adv_unlearning 运行；可加 --device cpu 若 GPU OOM。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"
cd "$ROOT"

CFG="${ROOT}/third_party/TransReID/configs/Market/vit_transreid_stride.yml"
OUT="${ROOT}/output/removable"
FORGET_ID=2
BATCH=32
DEVICE=""  # 留空用 GPU；OOM 时设为 cpu

# Seed 0（已有目录名）
DIRS=(
  "ablation_Conf_only_market1501_id${FORGET_ID}_seed0"
  "ablation_LD_low_auc"
  "ablation_L_empty_market1501_id${FORGET_ID}_seed0"
  "ablation_Spr_only_market1501_id${FORGET_ID}_seed0"
  "ablation_Full_optimal"
)
# Seed 1, 2
for s in 1 2; do
  DIRS+=("ablation_Conf_only_market1501_id${FORGET_ID}_seed${s}")
  DIRS+=("ablation_LD_low_auc_seed${s}")
  DIRS+=("ablation_L_empty_market1501_id${FORGET_ID}_seed${s}")
  DIRS+=("ablation_Spr_only_market1501_id${FORGET_ID}_seed${s}")
  DIRS+=("ablation_Full_optimal_seed${s}")
done

EXTRA=""
[[ -n "$DEVICE" ]] && EXTRA="--device $DEVICE"

for d in "${DIRS[@]}"; do
  full="${OUT}/${d}"
  if [[ ! -d "$full" ]]; then
    echo "[SKIP] $d (dir not found)"
    continue
  fi
  if [[ -f "$full/scorecard.json" ]] && python3 -c "import json; print(json.load(open('$full/scorecard.json')).get('test_without_mAP') is not None)" 2>/dev/null | grep -q True; then
    echo "[OK] $d (Test already in scorecard)"
    continue
  fi
  echo "[EVAL] $d"
  python "$ROOT/scripts/eval_test_split.py" --mvp_dir "$full" --cfg "$CFG" --forget_id $FORGET_ID --batch $BATCH $EXTRA 2>&1 | tail -5
done
echo "[DONE] eval_test_split 完成。运行 aggregate_ablation_table.py 生成 LaTeX。"
