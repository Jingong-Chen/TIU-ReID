"""Tune heatmap: 6 tid x 3x3 grid, 2x3 layout. Data from tune_grid_results.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.plotting.style import apply_tiu_style, save_tiu

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "output" / "compare" / "multitarget_v2"
FIG_PATH = REPO / "output" / "figures" / "tiu" / "fig_tune_heatmap.pdf"

TUNE_TIDS = [2, 7, 10, 20, 52, 100]
LB_VALS = [0.1, 0.5, 1.0]
LA_VALS = [0.05, 0.1, 0.2]


def load_grid():
    csv_path = OUT_ROOT / "tune_grid_results.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                lb = float(r.get("lb", 0))
                la = float(r.get("la", 0))
                tid = int(r.get("tid", 0))
            except (TypeError, ValueError):
                continue
            dropr = r.get("DropR")
            ret = r.get("ret_mAP")
            dropr = float(dropr) if dropr else np.nan
            ret = float(ret) if ret else np.nan
            rows.append({"tid": tid, "lb": lb, "la": la, "DropR": dropr, "ret_mAP": ret})
    return rows


def main():
    apply_tiu_style()
    import matplotlib as mpl
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["axes.titlesize"] = 14
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12

    grid_rows = load_grid()
    if not grid_rows:
        raise SystemExit("No tune grid data. Run run_multitarget_v2.py C stage first.")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), squeeze=True)
    axes = axes.flatten()

    for idx, tid in enumerate(TUNE_TIDS):
        if idx >= len(axes):
            break
        ax = axes[idx]
        Z = np.full((3, 3), np.nan)
        R = np.full((3, 3), np.nan)
        for r in grid_rows:
            if r["tid"] != tid:
                continue
            try:
                i = LB_VALS.index(r["lb"])
                j = LA_VALS.index(r["la"])
            except ValueError:
                continue
            Z[i, j] = r["DropR"] if not np.isnan(r["DropR"]) else np.nan
            R[i, j] = r["ret_mAP"] if not np.isnan(r["ret_mAP"]) else np.nan
        im = ax.imshow(Z, aspect="auto", cmap="viridis", vmin=0, vmax=0.9)
        for i in range(3):
            for j in range(3):
                v = Z[i, j]
                rv = R[i, j]
                t = f"{v:.2f}" if not np.isnan(v) else "-"
                if not np.isnan(rv):
                    t += f"\n({rv:.2f})"
                ax.text(j, i, t, ha="center", va="center", fontsize=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels([str(x) for x in LA_VALS], fontsize=11)
        ax.set_yticks(range(3))
        ax.set_yticklabels([str(x) for x in LB_VALS], fontsize=11)
        ax.set_xlabel(r"$\lambda_{\mathrm{adv}}$", fontsize=12)
        ax.set_ylabel(r"$\lambda_{\mathrm{baseonly}}$", fontsize=12)
        ax.set_title(f"Target ID {tid}", fontsize=13)

    for idx in range(len(TUNE_TIDS), len(axes)):
        axes[idx].set_visible(False)
    plt.subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.92, wspace=0.28, hspace=0.35)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tiu(fig, str(FIG_PATH))
    print(f"[OK] {FIG_PATH}")


if __name__ == "__main__":
    main()
