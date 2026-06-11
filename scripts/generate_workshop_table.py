#!/usr/bin/env python
"""Generate workshop comparison table."""
import json
from pathlib import Path
import csv

repo_root = Path(__file__).resolve().parents[1]
output_dir = repo_root / "output"
compare_dir = output_dir / "compare"
compare_dir.mkdir(parents=True, exist_ok=True)

# Load baseline teacher
base_dir = output_dir / "removable_baselines" / "base_teacher_market1501_id2_seed0"
base_summary = json.loads((base_dir / "summary_base.json").read_text())
forget_mAP_base = base_summary["forget_mAP"]

# Load WITH_ONLY
with_only_dir = output_dir / "removable_baselines" / "with_only_market1501_id2_seed0"
with_only_summary = json.loads((with_only_dir / "summary_with_only.json").read_text())

# Load retfix_F
retfix_f_dir = output_dir / "removable" / "removable_market1501_id2_seed0_retfix_F"
retfix_f_sanity = json.loads((retfix_f_dir / "sanity_report_removable.json").read_text())
retfix_f_test_with = json.loads((retfix_f_dir / "metrics_test_with.json").read_text())
retfix_f_test_without = json.loads((retfix_f_dir / "metrics_test_without.json").read_text())

# Find best epoch for retfix_F
epoch_eval_path = retfix_f_dir / "epoch_eval_short.json"
best_epoch = None
best_with_retain = 0
best_without_forget = 1.0
if epoch_eval_path.exists():
    epochs = json.loads(epoch_eval_path.read_text())
    for e in epochs:
        w_retain = e.get("with", {}).get("retain", {}).get("mAP", 0)
        wo_forget = e.get("without", {}).get("forget", {}).get("mAP", 1.0)
        if w_retain >= 0.95 and wo_forget < best_without_forget:
            best_epoch = e["epoch"]
            best_with_retain = w_retain
            best_without_forget = wo_forget

# Build rows
rows = []

# BASE_TEACHER
rows.append({
    "Method": "BASE_TEACHER",
    "Mode": "WITHOUT_MODULE",
    "retain_mAP": base_summary["retain_mAP"],
    "retain_CMC1": base_summary["retain_CMC1"],
    "forget_mAP": base_summary["forget_mAP"],
    "forget_CMC1": base_summary["forget_CMC1"],
    "test_mAP": None,
    "test_CMC1": None,
    "ForgettingDropAbs": 0.0,
    "ForgettingDropRatio": 0.0,
})

# WITH_ONLY WITH_MODULE
rows.append({
    "Method": "WITH_ONLY",
    "Mode": "WITH_MODULE",
    "retain_mAP": with_only_summary["with"]["retain_mAP"],
    "retain_CMC1": with_only_summary["with"]["retain_CMC1"],
    "forget_mAP": with_only_summary["with"]["forget_mAP"],
    "forget_CMC1": with_only_summary["with"]["forget_CMC1"],
    "test_mAP": None,
    "test_CMC1": None,
    "ForgettingDropAbs": forget_mAP_base - with_only_summary["with"]["forget_mAP"],
    "ForgettingDropRatio": (forget_mAP_base - with_only_summary["with"]["forget_mAP"]) / forget_mAP_base if forget_mAP_base > 0 else 0.0,
})

# WITH_ONLY WITHOUT_MODULE
rows.append({
    "Method": "WITH_ONLY",
    "Mode": "WITHOUT_MODULE",
    "retain_mAP": with_only_summary["without"]["retain_mAP"],
    "retain_CMC1": with_only_summary["without"]["retain_CMC1"],
    "forget_mAP": with_only_summary["without"]["forget_mAP"],
    "forget_CMC1": with_only_summary["without"]["forget_CMC1"],
    "test_mAP": None,
    "test_CMC1": None,
    "ForgettingDropAbs": forget_mAP_base - with_only_summary["without"]["forget_mAP"],
    "ForgettingDropRatio": (forget_mAP_base - with_only_summary["without"]["forget_mAP"]) / forget_mAP_base if forget_mAP_base > 0 else 0.0,
})

# RETFIX_F WITH_MODULE
rows.append({
    "Method": "RETFIX_F",
    "Mode": "WITH_MODULE",
    "retain_mAP": retfix_f_sanity.get("with_retain_mAP", 0),
    "retain_CMC1": None,  # Not in sanity report
    "forget_mAP": retfix_f_sanity.get("with_forget_mAP", 0),
    "forget_CMC1": None,
    "test_mAP": retfix_f_test_with["mAP"],
    "test_CMC1": retfix_f_test_with["CMC@1"],
    "ForgettingDropAbs": forget_mAP_base - retfix_f_sanity.get("with_forget_mAP", 0),
    "ForgettingDropRatio": (forget_mAP_base - retfix_f_sanity.get("with_forget_mAP", 0)) / forget_mAP_base if forget_mAP_base > 0 else 0.0,
})

# RETFIX_F WITHOUT_MODULE
rows.append({
    "Method": "RETFIX_F",
    "Mode": "WITHOUT_MODULE",
    "retain_mAP": retfix_f_sanity.get("without_retain_mAP", 0),
    "retain_CMC1": None,
    "forget_mAP": retfix_f_sanity.get("without_forget_mAP", 0),
    "forget_CMC1": None,
    "test_mAP": retfix_f_test_without["mAP"],
    "test_CMC1": retfix_f_test_without["CMC@1"],
    "ForgettingDropAbs": forget_mAP_base - retfix_f_sanity.get("without_forget_mAP", 0),
    "ForgettingDropRatio": (forget_mAP_base - retfix_f_sanity.get("without_forget_mAP", 0)) / forget_mAP_base if forget_mAP_base > 0 else 0.0,
})

# Write CSV
csv_path = compare_dir / "workshop_removable_table.csv"
with open(csv_path, "w", newline="") as f:
    fieldnames = ["Method", "Mode", "retain_mAP", "retain_CMC1", "forget_mAP", "forget_CMC1", "test_mAP", "test_CMC1", "ForgettingDropAbs", "ForgettingDropRatio"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# Write Markdown
md_path = compare_dir / "workshop_removable_table.md"
with open(md_path, "w") as f:
    f.write("# Removable Module Unlearning - Workshop Comparison Table\n\n")
    f.write("**Evaluation Protocol**: market1501, forget_id=2, seed=0, hard distractors=200\n\n")
    f.write("| Method | Mode | Retain mAP | Retain CMC@1 | Forget mAP | Forget CMC@1 | Test mAP | Test CMC@1 | Forgetting Drop (Abs) | Forgetting Drop (Ratio) |\n")
    f.write("|--------|------|------------|--------------|------------|--------------|----------|------------|----------------------|------------------------|\n")
    for r in rows:
        def fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        f.write(f"| {r['Method']} | {r['Mode']} | {fmt(r['retain_mAP'])} | {fmt(r['retain_CMC1'])} | {fmt(r['forget_mAP'])} | {fmt(r['forget_CMC1'])} | {fmt(r['test_mAP'])} | {fmt(r['test_CMC1'])} | {fmt(r['ForgettingDropAbs'])} | {fmt(r['ForgettingDropRatio'])} |\n")
    
    f.write("\n**Notes**:\n")
    f.write(f"- BASE_TEACHER forget mAP = {forget_mAP_base:.4f} (baseline, \"originally how much\")\n")
    f.write(f"- RETFIX_F best epoch = {best_epoch} (retain={best_with_retain:.4f}, forget={best_without_forget:.4f})\n")
    f.write(f"- ForgettingDropAbs = forget_mAP_base - forget_mAP_without\n")
    f.write(f"- ForgettingDropRatio = (forget_mAP_base - forget_mAP_without) / forget_mAP_base\n")

print(f"[OK] Wrote {csv_path}")
print(f"[OK] Wrote {md_path}")

# Print summary
print("\n" + "=" * 100)
print("WORKSHOP TABLE SUMMARY")
print("=" * 100)
for r in rows:
    print(f"{r['Method']:15s} | {r['Mode']:15s} | retain={fmt(r['retain_mAP']):6s} | forget={fmt(r['forget_mAP']):6s} | drop={fmt(r['ForgettingDropRatio']):6s}")

def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

print(f"\nBASE_TEACHER forget mAP (baseline): {forget_mAP_base:.4f}")
print(f"RETFIX_F WITHOUT forget mAP: {retfix_f_sanity.get('without_forget_mAP', 0):.4f}")
print(f"Forgetting Drop Ratio: {(forget_mAP_base - retfix_f_sanity.get('without_forget_mAP', 0)) / forget_mAP_base:.4f}")
