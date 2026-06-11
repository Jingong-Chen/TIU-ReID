#!/usr/bin/env python
"""Generate Table 3: Multi-target Generalization."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def load_metrics(run_dir: Path, mode: str) -> dict:
    """Load metrics for a run."""
    metrics = {}
    sanity_path = run_dir / "sanity_report_removable.json"
    if sanity_path.exists():
        sanity = json.loads(sanity_path.read_text())
        if mode == "with":
            metrics["retain_mAP"] = sanity.get("with_retain_mAP", 0)
            metrics["forget_mAP"] = sanity.get("with_forget_mAP", 0)
            metrics["test_mAP"] = sanity.get("test_with_mAP")
        else:
            metrics["retain_mAP"] = sanity.get("without_retain_mAP", 0)
            metrics["forget_mAP"] = sanity.get("without_forget_mAP", 0)
            metrics["test_mAP"] = sanity.get("test_without_mAP")
    return metrics


def format_float(v, default="--"):
    """Format float for LaTeX table."""
    if v is None or v == "":
        return default
    return f"{float(v):.3f}"


def format_mean_std(mean, std, default="--"):
    """Format mean±std for LaTeX table."""
    if mean is None or mean == "":
        return default
    if std is None or std == "" or std == 0:
        return f"{float(mean):.3f}"
    return f"${float(mean):.3f} \\pm {float(std):.3f}$"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "output"
    
    # Load baseline forget mAP
    base_dir = output_dir / "removable_baselines" / "base_teacher_market1501_id2_seed0"
    base_metrics = load_metrics(base_dir, "without")
    baseline_forget = base_metrics.get("forget_mAP", 1.0)
    
    # Find all multi-target runs (2,10,20,52,100: 25,50 not in Market1501 train)
    target_ids = [2, 10, 20, 52, 100]
    runs = []
    
    for tid in target_ids:
        tag = f"multi_target_id{tid}_market1501_seed0"
        run_dir = output_dir / "removable" / tag
        if run_dir.exists():
            m = load_metrics(run_dir, "without")
            forget_mAP = m.get("forget_mAP", 0)
            if forget_mAP:
                m["ForgetDropAbs"] = baseline_forget - forget_mAP
                m["ForgetDropRatio"] = (baseline_forget - forget_mAP) / baseline_forget if baseline_forget > 0 else 0
            runs.append({"target_id": tid, "metrics": m})
    
    if not runs:
        print("[WARN] No multi-target runs found")
        return
    
    # Aggregate statistics
    drop_ratios = [r["metrics"].get("ForgetDropRatio", 0) for r in runs if r["metrics"].get("ForgetDropRatio")]
    retain_maps = [r["metrics"].get("retain_mAP", 0) for r in runs if r["metrics"].get("retain_mAP")]
    test_maps = [r["metrics"].get("test_mAP", 0) for r in runs if r["metrics"].get("test_mAP") and r["metrics"].get("test_mAP") != ""]
    
    drop_mean = np.mean(drop_ratios) if drop_ratios else None
    drop_std = np.std(drop_ratios) if len(drop_ratios) > 1 else None
    retain_mean = np.mean(retain_maps) if retain_maps else None
    retain_std = np.std(retain_maps) if len(retain_maps) > 1 else None
    test_mean = np.mean(test_maps) if test_maps else None
    test_std = np.std(test_maps) if len(test_maps) > 1 else None
    
    # Generate LaTeX table
    tex_dir = repo_root / "tex" / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / "tab_multi_target.tex"
    
    with open(tex_path, "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Metric & Mean & Std & N \\\\\n")
        f.write("\\midrule\n")
        f.write(f"DropR & {format_mean_std(drop_mean, drop_std)} & {len(drop_ratios)} \\\\\n")
        f.write(f"Retain mAP & {format_mean_std(retain_mean, retain_std)} & {len(retain_maps)} \\\\\n")
        f.write(f"Test mAP & {format_mean_std(test_mean, test_std)} & {len(test_maps)} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Multi-target Generalization (5 target IDs).}\n")
        f.write("\\label{tab:multi_target}\n")
        f.write("\\end{table}\n")
    
    print(f"[OK] Generated {tex_path}")
    if drop_mean is not None:
        _std = drop_std if drop_std is not None else 0.0
        print(f"  DropR: {drop_mean:.3f} ± {_std:.3f}")
    if retain_mean is not None:
        _std = retain_std if retain_std is not None else 0.0
        print(f"  Retain mAP: {retain_mean:.3f} ± {_std:.3f}")
    if test_mean is not None:
        _std = test_std if test_std is not None else 0.0
        print(f"  Test mAP: {test_mean:.3f} ± {_std:.3f}")


if __name__ == "__main__":
    main()
