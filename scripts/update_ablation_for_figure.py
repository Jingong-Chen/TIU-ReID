#!/usr/bin/env python3
"""
用当前 ablation 的 scorecard（3 seeds 取均值）更新 all_runs_master.csv 中
Conf-only / Disc-only / L-empty-only / Spr-only / Full 的 without 行，
使 fig_ablation_bars.pdf 与 Table 2 的 mean 一致。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output"
REMOVABLE_DIR = OUT_DIR / "removable"
MASTER_CSV = OUT_DIR / "compare" / "all_runs_master.csv"

# 与 Table 2 / aggregate_ablation_table.py 一致：method -> 3 个 run 目录名
METHOD_TO_TAGS = {
    "Conf-only": [
        "ablation_Conf_only_market1501_id2_seed0",
        "ablation_Conf_only_market1501_id2_seed1",
        "ablation_Conf_only_market1501_id2_seed2",
    ],
    "Disc-only": ["ablation_LD_low_auc", "ablation_LD_low_auc_seed1", "ablation_LD_low_auc_seed2"],
    "L-empty-only": [
        "ablation_L_empty_market1501_id2_seed0",
        "ablation_L_empty_market1501_id2_seed1",
        "ablation_L_empty_market1501_id2_seed2",
    ],
    "Spr-only": [
        "ablation_Spr_only_market1501_id2_seed0",
        "ablation_Spr_only_market1501_id2_seed1",
        "ablation_Spr_only_market1501_id2_seed2",
    ],
    "Full": ["ablation_Full_optimal", "ablation_Full_optimal_seed1", "ablation_Full_optimal_seed2"],
}


def load_run_metrics(tag: str) -> dict | None:
    base = REMOVABLE_DIR / tag
    sc_path = base / "scorecard.json"
    if not sc_path.exists():
        return None
    with open(sc_path) as f:
        sc = json.load(f)
    ret = sc.get("without_retain_mAP") or sc.get("with_retain_mAP")
    fgt = sc.get("without_forget_mAP")
    if ret is None or fgt is None:
        return None
    baseline_fgt = 1.0
    drop_r = (baseline_fgt - fgt) / baseline_fgt if baseline_fgt > 0 else 0.0
    return {"retain_mAP": ret, "forget_mAP": fgt, "ForgetDropRatio": drop_r}


def main():
    # 1) 每个 method 从 3 个 seed 取均值，与 Table 2 一致
    new_rows = []
    for method, tags in METHOD_TO_TAGS.items():
        values = []
        for tag in tags:
            m = load_run_metrics(tag)
            if m is not None:
                values.append(m)
        if not values:
            print(f"[WARN] 未找到 {method} 任一 run，跳过")
            continue
        ret_mean = sum(v["retain_mAP"] for v in values) / len(values)
        fgt_mean = sum(v["forget_mAP"] for v in values) / len(values)
        drop_mean = sum(v["ForgetDropRatio"] for v in values) / len(values)
        row = {
            "tag": f"ablation_mean_seeds012_{method.replace('-', '_')}",
            "method": method,
            "mode": "without",
            "retain_mAP": round(ret_mean, 4),
            "retain_CMC1": "",
            "forget_mAP": round(fgt_mean, 4),
            "forget_CMC1": "",
            "test_mAP": "",
            "test_CMC1": "",
            "ForgetDropAbs": round(1.0 - fgt_mean, 4),
            "ForgetDropRatio": round(drop_mean, 4),
            "delta_ratio_mean": "",
            "base_norm_mean": "",
            "AUC_disc": "",
        }
        new_rows.append(row)

    if not new_rows:
        print("[ERR] 没有可用的 ablation 数据")
        return

    # 2) 读现有 CSV，去掉旧的 ablation without 行
    fieldnames = [
        "tag", "method", "mode", "retain_mAP", "retain_CMC1", "forget_mAP", "forget_CMC1",
        "test_mAP", "test_CMC1", "ForgetDropAbs", "ForgetDropRatio",
        "delta_ratio_mean", "base_norm_mean", "AUC_disc",
    ]
    ablation_methods = set(METHOD_TO_TAGS.keys())
    kept = []
    if MASTER_CSV.exists():
        with open(MASTER_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("method") in ablation_methods and (r.get("mode") or "").lower() == "without":
                    continue
                kept.append(r)

    # 3) 追加新行并写回
    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MASTER_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
        w.writerows(new_rows)

    print(f"[OK] 已更新 {MASTER_CSV}：写入 {len(new_rows)} 行 ablation (without)。")
    print("  重绘图: python scripts/plot_tiu_ablation_bars.py")
    print("  输出: output/figures/tiu/fig_ablation_bars.pdf")


if __name__ == "__main__":
    main()
