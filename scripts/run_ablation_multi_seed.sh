#!/usr/bin/env bash
# 按论文 Table 2 的 5 个 variant 配置，跑 seed 1 和 seed 2。
# Seed 0 已有：L_D=ablation_LD_low_auc，Full=ablation_Full_optimal；Conf/Entropy/Spread 的 seed 0 需单独跑时设 RUN_ABLATION_SEED0=1。
# Market1501, forget_id=2. 从项目根或 reid_adv_unlearning 运行均可。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT/scripts/env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

CFG="${ROOT}/third_party/TransReID/configs/Market/vit_transreid_stride.yml"
WTS="${ROOT}/output/transreid/market_teacher_r50/transformer_120.pth"
# 若 teacher_weights_path.txt 里是旧机器路径且文件不存在，则用当前项目路径
if [[ -f "${ROOT}/output/transreid/market_teacher_r50/teacher_weights_path.txt" ]]; then
  WTS_READ="$(cat "${ROOT}/output/transreid/market_teacher_r50/teacher_weights_path.txt")"
  [[ -f "$WTS_READ" ]] && WTS="$WTS_READ"
fi
PROBE="${ROOT}/output/probes/market1501/seed0_hard200_tid2"
OUT="${ROOT}/output/removable"
FORGET_ID=2
EPOCHS=10
BATCH=16
NW=4
DATA_ROOT="${REID_DATA_DIR:-$ROOT/data}"

# 只跑 seed 1,2；若要补 Conf/Entropy/Spread 的 seed 0，运行前：RUN_ABLATION_SEED0=1 bash run_ablation_multi_seed.sh
RUN_SEED0="${RUN_ABLATION_SEED0:-0}"
SEEDS="1 2"
[[ "$RUN_SEED0" == "1" ]] && SEEDS="0 1 2"

run_one() {
  local seed=$1
  local out_name=$2
  shift 2
  python "$ROOT/scripts/train_removable_unlearn_v3.py" \
    --cfg "$CFG" --weights "$WTS" --out_dir "${OUT}/${out_name}" \
    --dataset market1501 --data_root "$DATA_ROOT" --forget_id $FORGET_ID \
    --seed $seed --epochs $EPOCHS --batch $BATCH --num_workers $NW --probe_dir "$PROBE" \
    "$@"
}

for SEED in $SEEDS; do
  # 1) Confusion L_conf only
  TAG_CONF="ablation_Conf_only_market1501_id${FORGET_ID}_seed${SEED}"
  echo "[RUN] $TAG_CONF (seed=$SEED)"
  run_one $SEED "$TAG_CONF" \
    --lambda_adv 0 --lambda_baseonly_forget 0 --lambda_feat_confuse 0.5 --lambda_feat_spread 0 \
    --baseonly_warmup_frac 0 --anti_warmup_frac 0 --grl_lambda 1.0
done

for SEED in $SEEDS; do
  # 2) Adversarial L_D only（seed 0 已有 ablation_LD_low_auc）
  [[ "$SEED" == "0" ]] && echo "[SKIP] L_D seed 0 已有 (ablation_LD_low_auc)" && continue
  TAG_LD="ablation_LD_low_auc_seed${SEED}"
  echo "[RUN] $TAG_LD (seed=$SEED)"
  run_one $SEED "$TAG_LD" \
    --lambda_adv 0.6 --lambda_baseonly_forget 0 --lambda_feat_confuse 0 --lambda_feat_spread 0 \
    --anti_warmup_frac 0 --grl_lambda 1.0 --disc_hidden 64
done

for SEED in $SEEDS; do
  # 3) Entropy Max L_∅ only
  TAG_ENT="ablation_L_empty_market1501_id${FORGET_ID}_seed${SEED}"
  echo "[RUN] $TAG_ENT (seed=$SEED)"
  run_one $SEED "$TAG_ENT" \
    --lambda_adv 0 --lambda_baseonly_forget 0.5 --baseonly_forget_mode entropy \
    --lambda_feat_confuse 0 --lambda_feat_spread 0 \
    --baseonly_warmup_frac 0 --anti_warmup_frac 0 --grl_lambda 1.0
done

for SEED in $SEEDS; do
  # 4) Spread Reg L_spr only
  TAG_SPR="ablation_Spr_only_market1501_id${FORGET_ID}_seed${SEED}"
  echo "[RUN] $TAG_SPR (seed=$SEED)"
  run_one $SEED "$TAG_SPR" \
    --lambda_adv 0 --lambda_baseonly_forget 0 --lambda_feat_confuse 0 --lambda_feat_spread 0.5 \
    --baseonly_warmup_frac 0 --anti_warmup_frac 0 --grl_lambda 1.0
done

for SEED in $SEEDS; do
  # 5) Full objective（seed 0 已有 ablation_Full_optimal）
  [[ "$SEED" == "0" ]] && echo "[SKIP] Full seed 0 已有 (ablation_Full_optimal)" && continue
  TAG_FULL="ablation_Full_optimal_seed${SEED}"
  echo "[RUN] $TAG_FULL (seed=$SEED)"
  run_one $SEED "$TAG_FULL" \
    --lambda_adv 0.5 --lambda_baseonly_forget 0.1 --lambda_feat_confuse 1.0 --lambda_feat_spread 1.0 \
    --anti_warmup_frac 0 --grl_lambda 1.2 --disc_hidden 64
done

echo "[DONE] 多 seed 训练完成。对每个新 run 跑 eval_test_split 后执行 aggregate_ablation_table.py 生成 mean±std LaTeX。"
echo "  bash $ROOT/scripts/eval_ablation_multi_seed.sh"
echo "  python $ROOT/scripts/aggregate_ablation_table.py --removable_dir $OUT --forget_id $FORGET_ID --out_tex $OUT/ablation_table_mean_std.tex"
