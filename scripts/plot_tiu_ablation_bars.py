"""Ablation: component comparison. Data from all_runs_master.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from matplotlib.ticker import FuncFormatter

from scripts.plotting.style import apply_tiu_style, color, save_tiu, setup_tiu_subplots
OUT_DIR = REPO / "output"
CSV_PATH = OUT_DIR / "compare" / "all_runs_master.csv"
FIG_PATH = OUT_DIR / "figures" / "tiu" / "fig_ablation_bars.pdf"

ORDER = ["Conf-only", "Disc-only", "L-empty-only", "Spr-only", "Full"]
SHORT = {"Conf-only": "Conf", "Disc-only": "Disc", "L-empty-only": "Ent", "Spr-only": "Spr", "Full": "Full"}


def _f(x):
    return float(x) if x not in (None, "") else None


def load_ablation():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = r.get("method", "")
            if m not in ORDER or (r.get("mode") or "").lower() != "without":
                continue
            rows.append({
                "method": m,
                "retain_mAP": _f(r.get("retain_mAP")),
                "forget_mAP": _f(r.get("forget_mAP")),
                "DropR": _f(r.get("ForgetDropRatio")),
            })
    by_m = {x["method"]: x for x in rows}
    return [by_m[m] for m in ORDER if m in by_m]


def main():
    apply_tiu_style()
    data = load_ablation()
    if not data:
        raise SystemExit("No ablation data found.")

    fig, axes = setup_tiu_subplots(1, 3)
    ax_a, ax_b, ax_c = axes.flat[0], axes.flat[1], axes.flat[2]
    n = len(data)
    x = np.arange(n)
    w = 0.6

    ax_a.bar(x, [r["retain_mAP"] for r in data], w, color=[color(i) for i in range(n)], edgecolor=[color(i, edge=True) for i in range(n)])
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([SHORT.get(r["method"], r["method"]) for r in data], rotation=20, ha="right")
    ax_a.set_ylabel("Ret mAP")
    ax_a.set_ylim(0.92, 0.96)
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax_a.set_title("(a)", loc="left", fontsize=12)

    ax_b.bar(x, [r["forget_mAP"] for r in data], w, color=[color(i) for i in range(n)], edgecolor=[color(i, edge=True) for i in range(n)])
    ax_b.axhline(1.0, linestyle="--", linewidth=1.5, alpha=0.6, color="gray")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([SHORT.get(r["method"], r["method"]) for r in data], rotation=20, ha="right")
    ax_b.set_ylabel("Fgt mAP")
    ax_b.set_ylim(0.0, 1.02)
    ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax_b.set_title("(b)", loc="left", fontsize=12)

    ax_c.bar(x, [r["DropR"] for r in data], w, color=[color(i) for i in range(n)], edgecolor=[color(i, edge=True) for i in range(n)])
    ax_c.axhline(0.0, linestyle="--", linewidth=1.5, alpha=0.6, color="gray")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([SHORT.get(r["method"], r["method"]) for r in data], rotation=20, ha="right")
    ax_c.set_ylabel("DropR")
    ax_c.set_ylim(0.0, 0.95)
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax_c.set_title("(c)", loc="left", fontsize=12)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tiu(fig, str(FIG_PATH))
    print(f"[OK] {FIG_PATH}")


if __name__ == "__main__":
    main()
