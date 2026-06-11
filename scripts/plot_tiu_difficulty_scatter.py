"""Difficulty scatter: hardness vs DropR. Data from difficulty_metrics + multitarget_summary."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.plotting.style import apply_tiu_style, save_tiu

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "output" / "compare" / "multitarget_v2"
FIG_PATH = REPO / "output" / "figures" / "tiu" / "fig_difficulty_scatter.pdf"


def load_data():
    diff_path = OUT_ROOT / "difficulty_metrics.csv"
    summary_path = OUT_ROOT / "multitarget_summary_by_target.csv"
    if not diff_path.exists() or not summary_path.exists():
        return []
    drop_by_tid = {}
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            tid = int(row["tid"])
            v = row.get("DropR_mean")
            drop_by_tid[tid] = float(v) if v else None
    rows = []
    with open(diff_path) as f:
        for r in csv.DictReader(f):
            tid = int(r["tid"])
            n = int(r.get("n_target", 0))
            h = float(r.get("hardness", np.nan)) if r.get("hardness") else np.nan
            d = float(r.get("distractor_strength", np.nan)) if r.get("distractor_strength") else np.nan
            dropr = drop_by_tid.get(tid)
            if dropr is None:
                continue
            rows.append({"tid": tid, "n_target": n, "hardness": h, "distractor_strength": d, "DropR": dropr})
    return rows


def main():
    apply_tiu_style()
    data = load_data()
    if not data:
        raise SystemExit("No difficulty data. Run run_multitarget_v2.py D stage first.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    x = [r["hardness"] for r in data]
    y = [r["DropR"] for r in data]
    s = [np.log1p(r["n_target"]) * 30 for r in data]
    c = [r["distractor_strength"] for r in data]
    tids = [r["tid"] for r in data]

    sc = ax.scatter(x, y, s=s, c=c, cmap="viridis", alpha=0.8)
    for i, tid in enumerate(tids):
        ax.annotate(str(tid), (x[i], y[i]), fontsize=9, xytext=(5, 5), textcoords="offset points")
    if 2 in tids:
        idx2 = tids.index(2)
        ax.scatter([x[idx2]], [y[idx2]], s=s[idx2]*1.5, facecolors="none", edgecolors="black", linewidths=2)
    ax.set_xlabel("Hardness (top-5 mean sim)")
    ax.set_ylabel("DropR (mean)")
    plt.colorbar(sc, ax=ax, label="Distractor strength")
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tiu(fig, str(FIG_PATH))
    print(f"[OK] {FIG_PATH}")


if __name__ == "__main__":
    main()
