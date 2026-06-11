#!/usr/bin/env python
"""Summarize retfix experiments."""
import json
from pathlib import Path
import csv

repo_root = Path(__file__).resolve().parents[1]
output_dir = repo_root / "output" / "removable"
compare_dir = repo_root / "output" / "compare"
compare_dir.mkdir(parents=True, exist_ok=True)

# Find all retfix runs
runs = []
for d in output_dir.iterdir():
    if d.is_dir() and "retfix" in d.name:
        scorecard_path = d / "scorecard.json"
        sanity_path = d / "sanity_report_removable.json"
        epoch_eval_path = d / "epoch_eval_short.json"
        
        if not sanity_path.exists():
            continue
        
        sanity = json.loads(sanity_path.read_text())
        scorecard = json.loads(scorecard_path.read_text()) if scorecard_path.exists() else {}
        
        # Find best epoch from epoch_eval_short
        best_epoch = None
        best_retain = 0
        best_forget = 1.0
        if epoch_eval_path.exists():
            epochs = json.loads(epoch_eval_path.read_text())
            for e in epochs:
                w_retain = e.get("with", {}).get("retain", {}).get("mAP", 0)
                wo_forget = e.get("without", {}).get("forget", {}).get("mAP", 1.0)
                # Score: retain high, forget low
                if w_retain >= 0.95 and wo_forget < best_forget:
                    best_epoch = e["epoch"]
                    best_retain = w_retain
                    best_forget = wo_forget
        
        runs.append({
            "tag": d.name,
            "with_retain_mAP": sanity.get("with_retain_mAP", 0),
            "without_retain_mAP": sanity.get("without_retain_mAP", 0),
            "without_forget_mAP": sanity.get("without_forget_mAP", 1.0),
            "disc_auc": sanity.get("disc_auc", 1.0),
            "base_norm_mean": sanity.get("base_norm_mean", 0),
            "PASS": scorecard.get("PASS", False),
            "best_epoch": best_epoch,
            "best_retain": best_retain,
            "best_forget": best_forget,
        })

# Sort: PASS first, then by without_forget_mAP ascending, then retain descending
runs.sort(key=lambda x: (
    not x["PASS"],
    x["without_forget_mAP"],
    -x["without_retain_mAP"],
    abs(x["disc_auc"] - 0.5),
))

# Write CSV
csv_path = compare_dir / "sweep_removable_retfix.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "tag", "with_retain_mAP", "without_retain_mAP", "without_forget_mAP",
        "disc_auc", "base_norm_mean", "PASS", "best_epoch", "best_retain", "best_forget"
    ])
    writer.writeheader()
    for r in runs:
        writer.writerow(r)

# Write Markdown
md_path = compare_dir / "sweep_removable_retfix.md"
with open(md_path, "w") as f:
    f.write("# Removable Module Retfix Sweep Results\n\n")
    f.write("| Tag | WITH retain | WITHOUT retain | WITHOUT forget | disc_auc | base_norm | PASS | best_epoch | best_retain | best_forget |\n")
    f.write("|-----|-------------|----------------|----------------|----------|-----------|------|------------|-------------|-------------|\n")
    for r in runs:
        f.write(f"| {r['tag'][-20:]} | {r['with_retain_mAP']:.4f} | {r['without_retain_mAP']:.4f} | {r['without_forget_mAP']:.4f} | {r['disc_auc']:.4f} | {r['base_norm_mean']:.1f} | {r['PASS']} | {r['best_epoch']} | {r['best_retain']:.4f} | {r['best_forget']:.4f} |\n")

print(f"[OK] Wrote {csv_path}")
print(f"[OK] Wrote {md_path}")

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for r in runs:
    status = "✓ PASS" if r["PASS"] else "✗ FAIL"
    print(f"{r['tag'][-30:]:30s} | retain={r['without_retain_mAP']:.4f} forget={r['without_forget_mAP']:.4f} | {status}")
    if r["best_epoch"]:
        print(f"{'':30s} | best_epoch={r['best_epoch']}: retain={r['best_retain']:.4f} forget={r['best_forget']:.4f}")
