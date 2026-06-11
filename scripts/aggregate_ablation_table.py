#!/usr/bin/env python3
"""
从 5 个 variant × 3 seeds 的 scorecard.json 与 removable_diagnose_report.json 汇总
Ret/Fgt/Test/DropR/ρ/AUC，计算 mean±std，输出论文用 LaTeX 表格。
约定：seed 0 的 L_D = ablation_LD_low_auc，Full = ablation_Full_optimal；
其余为 ablation_Conf_only_market1501_id2_seed{s}, ablation_L_empty_market1501_id2_seed{s},
ablation_Spr_only_market1501_id2_seed{s}；seed 1,2 的 L_D/Full = ablation_LD_low_auc_seed{s}, ablation_Full_optimal_seed{s}。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


VARIANTS = [
    ("Confusion $\\mathcal{L}_{\\mathrm{conf}}$", "Conf", [
        "ablation_Conf_only_market1501_id2_seed0",
        "ablation_Conf_only_market1501_id2_seed1",
        "ablation_Conf_only_market1501_id2_seed2",
    ]),
    ("Adversarial Disc. $\\mathcal{L}_{D}$", "LD", [
        "ablation_LD_low_auc",
        "ablation_LD_low_auc_seed1",
        "ablation_LD_low_auc_seed2",
    ]),
    ("Entropy Max. $\\mathcal{L}_{\\varnothing}$", "Entropy", [
        "ablation_L_empty_market1501_id2_seed0",
        "ablation_L_empty_market1501_id2_seed1",
        "ablation_L_empty_market1501_id2_seed2",
    ]),
    ("Spread Reg. $\\mathcal{L}_{\\mathrm{spr}}$", "Spread", [
        "ablation_Spr_only_market1501_id2_seed0",
        "ablation_Spr_only_market1501_id2_seed1",
        "ablation_Spr_only_market1501_id2_seed2",
    ]),
    ("Full objective", "Full", [
        "ablation_Full_optimal",
        "ablation_Full_optimal_seed1",
        "ablation_Full_optimal_seed2",
    ]),
]

METRICS = ["Ret", "Fgt", "Test", "DropR", "rho", "AUC"]


def load_run(removable_dir: Path, run_name: str, forget_id: int) -> dict[str, float] | None:
    base = removable_dir / run_name
    sc_path = base / "scorecard.json"
    diag_path = base / "removable_diagnose_report.json"
    if not sc_path.exists():
        return None
    with open(sc_path) as f:
        sc = json.load(f)
    ret = sc.get("with_retain_mAP")
    fgt = sc.get("without_forget_mAP")
    test = sc.get("test_without_mAP")
    auc = sc.get("disc_auc")
    if ret is None or fgt is None:
        return None
    drop_r = 1.0 - fgt if fgt is not None else None
    rho = None
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)
        delta = diag.get("delta_ratio_final") or {}
        rho = delta.get("mean")
    return {
        "Ret": ret,
        "Fgt": fgt,
        "Test": test if test is not None else 0.0,
        "DropR": drop_r if drop_r is not None else (1.0 - fgt),
        "rho": rho if rho is not None else 0.0,
        "AUC": auc if auc is not None else 0.0,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return round(m, 3), 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return round(m, 3), round(math.sqrt(var), 3)


def fmt_cell(mean: float, std: float) -> str:
    return f"{mean:.3f}$\\pm${std:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Aggregate ablation runs to mean±std LaTeX table")
    ap.add_argument("--removable_dir", type=Path, default=None, help="output/removable dir")
    ap.add_argument("--forget_id", type=int, default=2)
    ap.add_argument("--out_tex", type=Path, default=None, help="Output .tex path")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    removable_dir = args.removable_dir or (root / "output" / "removable")
    out_tex = args.out_tex or (removable_dir / "ablation_table_mean_std.tex")

    rows = []
    for label, _short, run_names in VARIANTS:
        metrics_list: dict[str, list[float]] = {m: [] for m in METRICS}
        for name in run_names:
            row = load_run(removable_dir, name, args.forget_id)
            if row is None:
                continue
            for k in METRICS:
                key = "rho" if k == "rho" else k
                if key in row:
                    metrics_list[k].append(row[key])
        if not metrics_list["Ret"]:
            rows.append((label, None))
            continue
        means_stds = {m: mean_std(metrics_list[m]) for m in METRICS}
        rows.append((label, means_stds))

    # Build LaTeX
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Single-target loss-term ablation on Market1501 with target identity $2$ (seeds $0,1,2$).",
        "Ret, Fgt, and Test report mAP on the retain probe, forget probe, and standard test split; lower Fgt and higher DropR indicate stronger forgetting.",
        "$\\rho$ is the target residual ratio and AUC is the discriminator ROC-AUC. Mean$\\pm$std over 3 runs.}",
        "\\small",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        "Variant & Ret & Fgt & Test & DropR & $\\rho$ & AUC \\\\",
        "\\midrule",
    ]
    for label, means_stds in rows:
        if means_stds is None:
            lines.append(f"{label} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        cells = [fmt_cell(means_stds[m][0], means_stds[m][1]) for m in METRICS]
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:ablation}",
        "\\end{table}",
    ])

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] LaTeX written to {out_tex}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
