"""Main comparison: BASE / WITH_ONLY / TIU-ReID / baselines. Data from all_runs_master + removable_baselines."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.plotting.style import apply_tiu_style, color, save_tiu, setup_tiu_subplots

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output"
CSV_PATH = OUT_DIR / "compare" / "all_runs_master.csv"
FIG_PATH = OUT_DIR / "figures" / "tiu" / "fig_main_bars.pdf"

LABELS = ["BASE", "WONLY-W", "WONLY-R", "TIU-W", "TIU-R", "BEST"]
XTICK_6 = ["BASE", r"W$\Omega$", r"W$\varnothing$", r"T$\Omega$", r"T$\varnothing$", "BEST"]
XTICK_DROP = ["BASE", r"W$\varnothing$", r"T$\varnothing$", "BEST"]
HATCH = {"BASE": None, "WONLY-W": "///", "WONLY-R": None, "TIU-W": "///", "TIU-R": None, "BEST": None}
DROP_ONLY = {"BASE", "WONLY-R", "TIU-R", "BEST"}


def _f(x):
    return float(x) if x not in (None, "") else None


def load_main_data():
    rows = []
    base_forget = 1.0
    p = OUT_DIR / "removable_baselines" / "base_teacher_market1501_id2_seed0" / "summary_base.json"
    if p.exists():
        j = json.loads(p.read_text())
        rows.append({
            "label": "BASE",
            "retain_mAP": _f(j.get("retain_mAP")), "retain_R1": _f(j.get("retain_CMC1")),
            "forget_mAP": _f(j.get("forget_mAP")), "forget_R1": _f(j.get("forget_CMC1")),
            "test_mAP": None, "test_R1": None, "DropR": 0.0,
        })
    p = OUT_DIR / "removable_baselines" / "with_only_market1501_id2_seed0" / "summary_with_only.json"
    if p.exists():
        j = json.loads(p.read_text())
        for key, lbl in [("with", "WONLY-W"), ("without", "WONLY-R")]:
            b = j.get(key, {})
            f = _f(b.get("forget_mAP"))
            dropr = (base_forget - f) / base_forget if f is not None else None
            rows.append({
                "label": lbl,
                "retain_mAP": _f(b.get("retain_mAP")), "retain_R1": _f(b.get("retain_CMC1")),
                "forget_mAP": f, "forget_R1": _f(b.get("forget_CMC1")),
                "test_mAP": None, "test_R1": None, "DropR": dropr,
            })
    rows.append({"label": "TIU-W", "retain_mAP": 0.945, "retain_R1": None, "forget_mAP": 1.0, "forget_R1": None,
                 "test_mAP": 0.762, "test_R1": 0.897, "DropR": 0.0})
    rows.append({"label": "TIU-R", "retain_mAP": 0.945, "retain_R1": None, "forget_mAP": 0.473, "forget_R1": None,
                 "test_mAP": 0.762, "test_R1": 0.897, "DropR": 0.527})
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("method") == "Sweep1-lb0.1-la0.1" and (r.get("mode") or "").lower() == "without":
                    fv = _f(r.get("forget_mAP"))
                    dropr = _f(r.get("ForgetDropRatio")) or ((1.0 - fv) / 1.0 if fv is not None else None)
                    rows.append({
                        "label": "BEST",
                        "retain_mAP": _f(r.get("retain_mAP")), "retain_R1": _f(r.get("retain_CMC1")),
                        "forget_mAP": fv, "forget_R1": None,
                        "test_mAP": _f(r.get("test_mAP")), "test_R1": _f(r.get("test_CMC1")), "DropR": dropr,
                    })
                    break
    by_label = {x["label"]: x for x in rows}
    return [by_label[L] for L in LABELS if L in by_label]


def draw_panel(ax, data, key_mAP, key_R1, ylabel, xtick_labels, ylim, is_test=False, color_indices=None):
    n = len(data)
    x = np.arange(n)
    w = 0.28
    gap = 0.22
    off = [-gap / 2, gap / 2]
    cix = color_indices if color_indices is not None else list(range(n))
    mAP_vals = np.array([float(v) if v is not None else np.nan for v in [r.get(key_mAP) for r in data]])
    R1_vals = np.array([float(v) if v is not None else np.nan for v in [r.get(key_R1) for r in data]])
    mAP_done = R1_done = False
    for i in range(n):
        if np.isnan(mAP_vals[i]) and not is_test:
            continue
        if np.isnan(mAP_vals[i]) and is_test:
            continue
        h = HATCH.get(data[i]["label"])
        ax.bar(x[i] + off[0], mAP_vals[i], w, color=color(cix[i]), edgecolor=color(cix[i], edge=True),
               hatch=h, label="mAP" if not mAP_done else None)
        mAP_done = True
    for i in range(n):
        if np.isnan(R1_vals[i]) and not is_test:
            continue
        if np.isnan(R1_vals[i]) and is_test:
            continue
        h = HATCH.get(data[i]["label"])
        ax.bar(x[i] + off[1], R1_vals[i], w, color=color(cix[i]), edgecolor=color(cix[i], edge=True),
               hatch=h, label="R1" if not R1_done else None)
        R1_done = True
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)


def main():
    apply_tiu_style()
    data = load_main_data()
    if not data:
        raise SystemExit("No main-comparison data found.")

    xtick_6 = [XTICK_6[LABELS.index(r["label"])] for r in data]
    n = len(data)
    x = np.arange(n)
    figsize = (12, 8)
    fig, axes = setup_tiu_subplots(2, 2, figsize=figsize)
    ax_a, ax_b, ax_c, ax_d = axes.flat[0], axes.flat[1], axes.flat[2], axes.flat[3]

    draw_panel(ax_a, data, "retain_mAP", "retain_R1", "Retain mAP / R1", xtick_6, (0.92, 1.00), is_test=False)
    ax_a.set_title("(a)", loc="left", fontsize=12)
    draw_panel(ax_b, data, "forget_mAP", "forget_R1", "Forget mAP / R1", xtick_6, (0.0, 1.0), is_test=False)
    data_test = [r for r in data if r["label"] in {"TIU-R", "BEST"}]
    xtick_test = [r"T$\varnothing$", "BEST"]
    cix_test = [LABELS.index(r["label"]) for r in data_test]
    draw_panel(ax_c, data_test, "test_mAP", "test_R1", "Test mAP / R1", xtick_test, (0.65, 0.95), is_test=True, color_indices=cix_test)
    ax_c.set_title("(c)", loc="left", fontsize=12)

    sub = [r for r in data if r["label"] in DROP_ONLY]
    xs = np.arange(len(sub))
    xtick_d = [XTICK_DROP[0], XTICK_DROP[1], XTICK_DROP[2], XTICK_DROP[3]]
    label_to_idx = {r["label"]: i for i, r in enumerate(data)}
    w_bar = 0.28
    for i, r in enumerate(sub):
        v = r.get("DropR")
        if v is not None:
            ci = label_to_idx[r["label"]]
            ax_d.bar(xs[i], v, w_bar * 2, color=color(ci), edgecolor=color(ci, edge=True))
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels(xtick_d, rotation=15, ha="right")
    ax_d.set_ylabel("DropR")
    ax_d.set_ylim(0.0, 0.95)
    ax_d.set_title("(d)", loc="left", fontsize=12)

    from matplotlib.patches import Patch
    fig.legend(
        [Patch(facecolor=color(0), edgecolor=color(0, edge=True)), Patch(facecolor=color(1), edgecolor=color(1, edge=True))],
        ["mAP", "R1"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    footnote = fig.text(0.5, 0.02, r"$\Omega$: full deployment, $\varnothing$: removal deployment", ha="center", fontsize=9)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tiu(fig, str(FIG_PATH), extra_artists=[footnote])
    print(f"[OK] {FIG_PATH}")


if __name__ == "__main__":
    main()
