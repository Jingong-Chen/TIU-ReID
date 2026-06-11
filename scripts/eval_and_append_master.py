#!/usr/bin/env python
"""Evaluate a run and append to master CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.eval_baseline_probes import main as eval_probes
from scripts.eval_test_split import main as eval_test


def load_baseline_forget_mAP(probe_dir: Path) -> float:
    """Load BASE_TEACHER forget mAP from baseline."""
    import os
    base_dir = Path(os.environ.get("REID_OUTPUT_DIR", ".")) / "removable_baselines" / "base_teacher_market1501_id2_seed0"
    summary_path = base_dir / "summary_base.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        return summary.get("forget_mAP", 1.0)
    return 1.0


def append_to_master_csv(tag: str, method: str, mode: str, metrics: dict, master_csv: Path):
    """Append run metrics to master CSV."""
    import os
    
    baseline_forget = load_baseline_forget_mAP(Path(os.environ.get("REID_OUTPUT_DIR", ".")) / "probes" / "market1501" / "seed0_hard200")
    
    row = {
        "tag": tag,
        "method": method,
        "mode": mode,
        "retain_mAP": metrics.get("retain_mAP", ""),
        "retain_CMC1": metrics.get("retain_CMC1", ""),
        "forget_mAP": metrics.get("forget_mAP", ""),
        "forget_CMC1": metrics.get("forget_CMC1", ""),
        "test_mAP": metrics.get("test_mAP", ""),
        "test_CMC1": metrics.get("test_CMC1", ""),
        "ForgetDropAbs": baseline_forget - metrics.get("forget_mAP", 0) if metrics.get("forget_mAP") else "",
        "ForgetDropRatio": (baseline_forget - metrics.get("forget_mAP", 0)) / baseline_forget if baseline_forget > 0 and metrics.get("forget_mAP") else "",
        "delta_ratio_mean": metrics.get("delta_ratio_mean", ""),
        "base_norm_mean": metrics.get("base_norm_mean", ""),
        "AUC_disc": metrics.get("AUC_disc", ""),
    }
    
    # Append to CSV
    file_exists = master_csv.exists()
    with open(master_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--method", required=True, help="Method name (e.g., BASE_TEACHER, TIU-ReID)")
    ap.add_argument("--mode", required=True, choices=["with", "without"])
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--master_csv", default=None)
    ap.add_argument("--eval_test", action="store_true", help="Also evaluate test split")
    args = ap.parse_args()

    import os
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir)
    
    if args.master_csv:
        master_csv = Path(args.master_csv)
    else:
        master_csv = Path(os.environ.get("REID_OUTPUT_DIR", repo_root / "output")) / "compare" / "all_runs_master.csv"
    
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metrics if available
    metrics = {}
    
    # Try to load from sanity_report or probe metrics
    sanity_path = run_dir / "sanity_report_removable.json"
    if sanity_path.exists():
        sanity = json.loads(sanity_path.read_text())
        if args.mode == "with":
            metrics["retain_mAP"] = sanity.get("with_retain_mAP")
            metrics["forget_mAP"] = sanity.get("with_forget_mAP")
        else:
            metrics["retain_mAP"] = sanity.get("without_retain_mAP")
            metrics["forget_mAP"] = sanity.get("without_forget_mAP")
    
    # Load test metrics if available
    if args.eval_test:
        test_path = run_dir / f"metrics_test_{args.mode}.json"
        if test_path.exists():
            test_metrics = json.loads(test_path.read_text())
            metrics["test_mAP"] = test_metrics.get("mAP")
            metrics["test_CMC1"] = test_metrics.get("CMC@1")
    
    # Load diagnostic metrics
    diag_path = run_dir / "removable_diagnose_report.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text())
        metrics["delta_ratio_mean"] = diag.get("delta_ratio_final", {}).get("mean")
        metrics["base_norm_mean"] = diag.get("base_norm_final", {}).get("mean")
        metrics["AUC_disc"] = diag.get("discriminator", {}).get("auc")
    
    tag = run_dir.name
    append_to_master_csv(tag, args.method, args.mode, metrics, master_csv)
    print(f"[OK] Appended {tag} ({args.method}, {args.mode}) to {master_csv}")


if __name__ == "__main__":
    main()
