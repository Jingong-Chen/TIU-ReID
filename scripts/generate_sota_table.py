#!/usr/bin/env python
"""Generate Table 1: Comparison on Market1501 under TIU protocol."""
from __future__ import annotations

import json
from pathlib import Path


def load_metrics(run_dir: Path, mode: str) -> dict:
    """Load metrics for a run."""
    metrics = {}
    
    # Try sanity_report_removable.json
    sanity_path = run_dir / "sanity_report_removable.json"
    if sanity_path.exists():
        sanity = json.loads(sanity_path.read_text())
        if mode == "with":
            metrics["retain_mAP"] = sanity.get("with_retain_mAP", 0)
            metrics["retain_CMC1"] = None  # Not in sanity report
            metrics["forget_mAP"] = sanity.get("with_forget_mAP", 0)
            metrics["forget_CMC1"] = None
            metrics["test_mAP"] = sanity.get("test_with_mAP")
            metrics["test_CMC1"] = sanity.get("test_with_CMC1")
        else:
            metrics["retain_mAP"] = sanity.get("without_retain_mAP", 0)
            metrics["retain_CMC1"] = None
            metrics["forget_mAP"] = sanity.get("without_forget_mAP", 0)
            metrics["forget_CMC1"] = None
            metrics["test_mAP"] = sanity.get("test_without_mAP")
            metrics["test_CMC1"] = sanity.get("test_without_CMC1")
    
    # Try summary_base.json (for BASE_TEACHER)
    summary_path = run_dir / "summary_base.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        metrics["retain_mAP"] = summary.get("retain_mAP", 0)
        metrics["retain_CMC1"] = summary.get("retain_CMC1", 0)
        metrics["forget_mAP"] = summary.get("forget_mAP", 0)
        metrics["forget_CMC1"] = summary.get("forget_CMC1", 0)
    
    # Try summary_with_only.json (for WITH_ONLY)
    with_only_path = run_dir / "summary_with_only.json"
    if with_only_path.exists():
        with_only = json.loads(with_only_path.read_text())
        if mode == "with":
            sub = with_only.get("with", {})
        else:
            sub = with_only.get("without", {})
        metrics["retain_mAP"] = sub.get("retain_mAP", 0)
        metrics["retain_CMC1"] = sub.get("retain_CMC1", 0)
        metrics["forget_mAP"] = sub.get("forget_mAP", 0)
        metrics["forget_CMC1"] = sub.get("forget_CMC1", 0)
    
    return metrics


def format_float(v, default="--"):
    """Format float for LaTeX table."""
    if v is None or v == "":
        return default
    return f"{float(v):.3f}"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "output"
    
    # Load baseline forget mAP
    base_dir = output_dir / "removable_baselines" / "base_teacher_market1501_id2_seed0"
    base_metrics = load_metrics(base_dir, "without")
    baseline_forget = base_metrics.get("forget_mAP", 1.0)
    
    # Load all runs
    runs = []
    
    # BASE_TEACHER
    runs.append({
        "method": "BASE\\_TEACHER",
        "mode": "WITHOUT",
        "metrics": load_metrics(base_dir, "without"),
    })
    
    # WITH_ONLY
    with_only_dir = output_dir / "removable_baselines" / "with_only_market1501_id2_seed0"
    runs.append({
        "method": "WITH\\_ONLY",
        "mode": "WITH",
        "metrics": load_metrics(with_only_dir, "with"),
    })
    runs.append({
        "method": "WITH\\_ONLY",
        "mode": "WITHOUT",
        "metrics": load_metrics(with_only_dir, "without"),
    })
    
    # TIU-ReID (retfix_F)
    retfix_f_dir = output_dir / "removable" / "removable_market1501_id2_seed0_retfix_F"
    runs.append({
        "method": "TIU-ReID",
        "mode": "WITH",
        "metrics": load_metrics(retfix_f_dir, "with"),
    })
    runs.append({
        "method": "TIU-ReID",
        "mode": "WITHOUT",
        "metrics": load_metrics(retfix_f_dir, "without"),
    })
    
    # Calculate ForgetDropRatio
    for run in runs:
        m = run["metrics"]
        forget_mAP = m.get("forget_mAP", 0)
        if forget_mAP:
            m["ForgetDropAbs"] = baseline_forget - forget_mAP
            m["ForgetDropRatio"] = (baseline_forget - forget_mAP) / baseline_forget if baseline_forget > 0 else 0
        else:
            m["ForgetDropAbs"] = None
            m["ForgetDropRatio"] = None
    
    # Generate LaTeX table
    tex_dir = repo_root / "tex" / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / "tab_sota_market1501_tiu.tex"
    
    with open(tex_path, "w") as f:
        f.write("\\begin{table*}\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcc|cc|cc|cc|c}\n")
        f.write("\\toprule\n")
        f.write("Method & Mode & Ret mAP & Ret R1 & Fgt mAP & Fgt R1 & Test mAP & Test R1 & DropR \\\\\n")
        f.write("\\midrule\n")
        
        for run in runs:
            m = run["metrics"]
            method = run["method"]
            mode = run["mode"]
            f.write(f"{method} & {mode} & "
                   f"{format_float(m.get('retain_mAP'))} & {format_float(m.get('retain_CMC1'))} & "
                   f"{format_float(m.get('forget_mAP'))} & {format_float(m.get('forget_CMC1'))} & "
                   f"{format_float(m.get('test_mAP'))} & {format_float(m.get('test_CMC1'))} & "
                   f"{format_float(m.get('ForgetDropRatio'))} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Comparison on Market1501 under TIU protocol.}\n")
        f.write("\\label{tab:sota_market1501_tiu}\n")
        f.write("\\end{table*}\n")
    
    print(f"[OK] Generated {tex_path}")
    
    # Also generate markdown preview
    md_path = output_dir / "compare" / "tab_sota_market1501_tiu.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write("# Table 1: Comparison on Market1501 under TIU protocol\n\n")
        f.write("| Method | Mode | Ret mAP | Ret R1 | Fgt mAP | Fgt R1 | Test mAP | Test R1 | DropR |\n")
        f.write("|--------|------|---------|--------|---------|--------|----------|----------|-------|\n")
        for run in runs:
            m = run["metrics"]
            method_name = run['method'].replace('\\_', '_')
            f.write(f"| {method_name} | {run['mode']} | "
                   f"{format_float(m.get('retain_mAP'))} | {format_float(m.get('retain_CMC1'))} | "
                   f"{format_float(m.get('forget_mAP'))} | {format_float(m.get('forget_CMC1'))} | "
                   f"{format_float(m.get('test_mAP'))} | {format_float(m.get('test_CMC1'))} | "
                   f"{format_float(m.get('ForgetDropRatio'))} |\n")
    
    print(f"[OK] Generated {md_path}")


if __name__ == "__main__":
    main()
