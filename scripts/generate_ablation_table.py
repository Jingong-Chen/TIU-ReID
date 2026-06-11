#!/usr/bin/env python
"""Generate Table 2: Ablation on TIU Components."""
from __future__ import annotations

import json
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
        else:
            metrics["retain_mAP"] = sanity.get("without_retain_mAP", 0)
            metrics["forget_mAP"] = sanity.get("without_forget_mAP", 0)
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
    
    # Define ablation variants
    variants = [
        ("BASE\\_TEACHER", "base_teacher_market1501_id2_seed0", "removable_baselines", "without"),
        ("WITH\\_ONLY", "with_only_market1501_id2_seed0", "removable_baselines", "without"),
        ("L\\_empty only", "ablation_L_empty_market1501_id2_seed0", "removable", "without"),
        ("Disc only", "ablation_Disc_only_market1501_id2_seed0", "removable", "without"),
        ("Conf only", "ablation_Conf_only_market1501_id2_seed0", "removable", "without"),
        ("Spr only", "ablation_Spr_only_market1501_id2_seed0", "removable", "without"),
        ("L\\_empty + Disc", "ablation_L_empty_Disc_market1501_id2_seed0", "removable", "without"),
        ("L\\_empty + Disc + Conf", "ablation_L_empty_Disc_Conf_market1501_id2_seed0", "removable", "without"),
        ("Full", "ablation_Full_market1501_id2_seed0", "removable", "without"),
        ("+ Consistency", "ablation_Full_Consist_market1501_id2_seed0", "removable", "without"),
    ]
    
    runs = []
    for method, tag, subdir, mode in variants:
        run_dir = output_dir / subdir / tag
        if run_dir.exists():
            m = load_metrics(run_dir, mode)
            forget_mAP = m.get("forget_mAP", 0)
            if forget_mAP:
                m["ForgetDropAbs"] = baseline_forget - forget_mAP
                m["ForgetDropRatio"] = (baseline_forget - forget_mAP) / baseline_forget if baseline_forget > 0 else 0
            else:
                m["ForgetDropAbs"] = None
                m["ForgetDropRatio"] = None
            runs.append({"method": method, "metrics": m})
    
    # Generate LaTeX table
    tex_dir = repo_root / "tex" / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / "tab_ablation_components.tex"
    
    with open(tex_path, "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcc|c}\n")
        f.write("\\toprule\n")
        f.write("Method & Ret mAP & Fgt mAP & DropR \\\\\n")
        f.write("\\midrule\n")
        
        for run in runs:
            m = run["metrics"]
            f.write(f"{run['method']} & "
                   f"{format_float(m.get('retain_mAP'))} & "
                   f"{format_float(m.get('forget_mAP'))} & "
                   f"{format_float(m.get('ForgetDropRatio'))} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Ablation on TIU Components.}\n")
        f.write("\\label{tab:ablation_components}\n")
        f.write("\\end{table}\n")
    
    print(f"[OK] Generated {tex_path}")


if __name__ == "__main__":
    main()
