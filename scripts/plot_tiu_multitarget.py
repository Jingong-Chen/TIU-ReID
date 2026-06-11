"""Multi-target: 10 tid DropR + Ret/Fgt/Test mAP. Data from multitarget_v2."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.plotting.style import apply_tiu_style, color, save_tiu, setup_tiu_subplots

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "output" / "compare" / "multitarget_v2"
FIG_PATH = REPO / "output" / "figures" / "tiu" / "fig_multitarget.pdf"


def _f(x):
    if x is None or x == "" or str(x).strip() == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_multitarget():
    csv_path = OUT_ROOT / "multitarget_summary_by_target.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "tid": int(r["tid"]),
                "ret_mAP": _f(r.get("ret_mAP_mean")),
                "fgt_mAP": _f(r.get("fgt_mAP_mean")),
                "DropR": _f(r.get("DropR_mean")),
                "test_mAP": _f(r.get("test_mAP_mean")),
            })
    return rows


def main():
    apply_tiu_style()
    import matplotlib as mpl
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["axes.labelsize"] = 16
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams["ytick.labelsize"] = 14
    mpl.rcParams["legend.fontsize"] = 14

    data = load_multitarget()
    if not data:
        raise SystemExit("No multitarget data. Run run_multitarget_v2.py first.")

    fig, axes = setup_tiu_subplots(1, 2, figsize=(12, 5))
    ax_left, ax_right = axes.flat[0], axes.flat[1]
    tids = [r["tid"] for r in data]
    x = np.arange(len(tids))

    drop = [r["DropR"] if r["DropR"] is not None else np.nan for r in data]
    drop_vals = [v for v in drop if not np.isnan(v)]
    drop_max = min(1.0, max(drop_vals) + 0.02) if drop_vals else 0.7

    ax_left.plot(x, drop, "o-", color=color(0), linewidth=3.5, markersize=10, markeredgecolor=color(0, edge=True), clip_on=False)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([str(i) for i in tids])
    ax_left.set_xlabel("Target ID")
    ax_left.set_ylabel("DropR")
    ax_left.set_ylim(0.0, drop_max)
    ax_left.set_title("(a)", loc="left")

    ret = [r["ret_mAP"] if r["ret_mAP"] is not None else np.nan for r in data]
    fgt = [r["fgt_mAP"] if r["fgt_mAP"] is not None else np.nan for r in data]
    tst = [r["test_mAP"] if r["test_mAP"] is not None else np.nan for r in data]
    all_vals = [v for v in ret + fgt + tst if not np.isnan(v)]
    ymin = max(0.0, min(all_vals) - 0.02) if all_vals else 0.0
    ymax = min(1.0, max(all_vals) + 0.01) if all_vals else 1.0

    ax_right.plot(x, ret, "-o", color=color(0), linewidth=3.5, markersize=10, label="Ret mAP", markeredgecolor=color(0, edge=True), clip_on=False)
    ax_right.plot(x, tst, "--s", color=color(1), linewidth=3.5, markersize=10, label="Test mAP", markeredgecolor=color(1, edge=True), clip_on=False)
    ax_right.plot(x, fgt, "-.^", color=color(2), linewidth=3.5, markersize=10, label="Fgt mAP", markeredgecolor=color(2, edge=True), clip_on=False)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([str(i) for i in tids])
    ax_right.set_xlabel("Target ID")
    ax_right.set_ylabel("mAP")
    ax_right.set_ylim(ymin, ymax)
    ax_right.legend(loc="center left", bbox_to_anchor=(0, 0.58), framealpha=0.9, fontsize=14)
    ax_right.set_title("(b)", loc="left")

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tiu(fig, str(FIG_PATH))
    print(f"[OK] {FIG_PATH}")


if __name__ == "__main__":
    main()
