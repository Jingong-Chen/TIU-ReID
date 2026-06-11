#!/usr/bin/env python3
"""
从 method_retfixF_per_target.csv 重新生成汇总表（summary_by_target, summary_overall, .md/.tex）。
在 补充test评估.py 全部跑完后运行一次即可。
"""
import csv
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "output" / "compare" / "multitarget_v2"

def _parse_float(s):
    if s is None or s == "" or str(s).strip() == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def main():
    csv_path = OUT_ROOT / "method_retfixF_per_target.csv"
    if not csv_path.exists():
        print(f"❌ 找不到 {csv_path}")
        return
    
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    
    method_rows = []
    for r in rows:
        method_rows.append({
            "seed": int(r.get("seed", 0)),
            "tid": int(r.get("tid", 0)),
            "ret_mAP": _parse_float(r.get("ret_mAP")),
            "fgt_mAP": _parse_float(r.get("fgt_mAP")),
            "DropR": _parse_float(r.get("DropR")),
            "test_mAP": _parse_float(r.get("test_mAP")),
            "test_R1": _parse_float(r.get("test_R1")),
        })
    
    by_tid = {}
    for row in method_rows:
        tid = row["tid"]
        if tid not in by_tid:
            by_tid[tid] = []
        by_tid[tid].append(row)
    
    def mean_std(vals):
        v = [float(x) for x in vals if x is not None]
        if not v:
            return None, None
        return np.mean(v), np.std(v)
    
    summary_by_target = []
    for tid in sorted(by_tid.keys()):
        rs = by_tid[tid]
        ret_m, ret_s = mean_std([r["ret_mAP"] for r in rs])
        fgt_m, fgt_s = mean_std([r["fgt_mAP"] for r in rs])
        drop_m, drop_s = mean_std([r["DropR"] for r in rs])
        test_m, test_s = mean_std([r["test_mAP"] for r in rs])
        summary_by_target.append({
            "tid": tid,
            "ret_mAP_mean": ret_m, "ret_mAP_std": ret_s,
            "fgt_mAP_mean": fgt_m, "fgt_mAP_std": fgt_s,
            "DropR_mean": drop_m, "DropR_std": drop_s,
            "test_mAP_mean": test_m, "test_mAP_std": test_s,
        })
    
    with (OUT_ROOT / "multitarget_summary_by_target.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tid", "ret_mAP_mean", "ret_mAP_std", "fgt_mAP_mean", "fgt_mAP_std", "DropR_mean", "DropR_std", "test_mAP_mean", "test_mAP_std"])
        w.writeheader()
        w.writerows(summary_by_target)
    
    all_ret = [r["ret_mAP"] for r in method_rows if r["ret_mAP"] is not None]
    all_fgt = [r["fgt_mAP"] for r in method_rows if r["fgt_mAP"] is not None]
    all_drop = [r["DropR"] for r in method_rows if r["DropR"] is not None]
    overall = {
        "ret_mAP_mean": np.mean(all_ret) if all_ret else None, "ret_mAP_std": np.std(all_ret) if all_ret else None,
        "fgt_mAP_mean": np.mean(all_fgt) if all_fgt else None, "fgt_mAP_std": np.std(all_fgt) if all_fgt else None,
        "DropR_mean": np.mean(all_drop) if all_drop else None, "DropR_std": np.std(all_drop) if all_drop else None,
    }
    with (OUT_ROOT / "multitarget_summary_overall.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ret_mAP_mean", "ret_mAP_std", "fgt_mAP_mean", "fgt_mAP_std", "DropR_mean", "DropR_std"])
        w.writeheader()
        w.writerow(overall)
    
    def fmt_std(m, s):
        if m is None:
            return "0.000±0.000"
        return f"{m:.3f}±{(s or 0):.3f}"
    
    md_lines = ["| Target ID | Ret mAP (mean±std) | Fgt mAP (mean±std) | DropR (mean±std) | Test mAP (mean±std) |", "|" + "---|" * 5]
    for row in summary_by_target:
        md_lines.append("| {} | {} | {} | {} | {} |".format(
            row["tid"], fmt_std(row["ret_mAP_mean"], row["ret_mAP_std"]),
            fmt_std(row["fgt_mAP_mean"], row["fgt_mAP_std"]),
            fmt_std(row["DropR_mean"], row["DropR_std"]),
            fmt_std(row["test_mAP_mean"], row["test_mAP_std"]),
        ))
    (OUT_ROOT / "multitarget_table.md").write_text("\n".join(md_lines) + "\n")
    
    tex_lines = ["\\begin{tabular}{c|cccc}\nTarget ID & Ret mAP & Fgt mAP & DropR & Test mAP \\\\\n\\hline"]
    for row in summary_by_target:
        tex_lines.append("{} & ${}$ & ${}$ & ${}$ & ${}$ \\\\".format(
            row["tid"], fmt_std(row["ret_mAP_mean"], row["ret_mAP_std"]),
            fmt_std(row["fgt_mAP_mean"], row["fgt_mAP_std"]),
            fmt_std(row["DropR_mean"], row["DropR_std"]),
            fmt_std(row["test_mAP_mean"], row["test_mAP_std"]),
        ))
    tex_lines.append("\\end{tabular}")
    (OUT_ROOT / "multitarget_table.tex").write_text("\n".join(tex_lines) + "\n")
    
    n_with_test = sum(1 for r in method_rows if r["test_mAP"] is not None)
    print(f"✅ 已重新生成汇总表（共 {len(method_rows)} 行，其中 {n_with_test} 行有 test_mAP）")
    print(f"   - multitarget_summary_by_target.csv")
    print(f"   - multitarget_summary_overall.csv")
    print(f"   - multitarget_table.md / multitarget_table.tex")

if __name__ == "__main__":
    main()
