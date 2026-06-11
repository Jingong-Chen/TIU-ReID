"""Combined Sweep 1+2: 1 row × 4 columns. Sweep1 (lb×la) + Sweep2 (tbs×lc)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.plotting.style import apply_tiu_style, save_tiu

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output"
CSV_PATH = OUT_DIR / "compare" / "all_runs_master.csv"
FIG_PATH = OUT_DIR / "figures" / "tiu" / "fig_sweep_combined.pdf"

# Sweep1
S1_X = [0.1, 0.25, 0.5, 1.0, 2.0]
S1_Y = [1.0, 0.5, 0.2, 0.1, 0.05]
# Sweep2
S2_X = [0.0, 0.5, 1.0, 1.5, 2.0]
S2_Y = [5.0, 2.0, 1.0, 0.5, 0.0]
S2_Y_LABELS = ["5", "2", "1", ".5", "0"]

ELEV, AZIM = 40, -60
DX, DY = 0.45, 0.45
ALPHA = 0.90
EDGE_RGBA = (0, 0, 0, 0.25)
LW = 0.35


def _f(x):
    return float(x) if x not in (None, "") else None


def pastelize_cmap(base="viridis", blend=0.35):
    import matplotlib.pyplot as _plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = _plt.get_cmap(base)
    cols = cmap(np.linspace(0, 1, 256))
    cols[:, :3] = (1 - blend) * cols[:, :3] + blend * np.ones_like(cols[:, :3])
    return LinearSegmentedColormap.from_list(f"{base}_pastel", cols)


def load_sweep1():
    grid = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = r.get("method", "")
            if not m.startswith("Sweep1-") or (r.get("mode") or "").lower() != "without":
                continue
            s = m.replace("Sweep1-lb", "").strip().split("-la")
            if len(s) != 2:
                continue
            try:
                lb, la = float(s[0]), float(s[1])
            except Exception:
                continue
            grid[(lb, la)] = {
                "DropR": _f(r.get("ForgetDropRatio")),
                "retain_mAP": _f(r.get("retain_mAP")),
            }
    return grid


def load_sweep2():
    grid = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = r.get("method", "")
            if not m.startswith("Sweep2-") or (r.get("mode") or "").lower() != "without":
                continue
            s = m.replace("Sweep2-tbs", "").strip().split("-lc")
            if len(s) != 2:
                continue
            try:
                tbs, lc = float(s[0]), float(s[1])
            except Exception:
                continue
            grid[(tbs, lc)] = {
                "forget_mAP": _f(r.get("forget_mAP")),
                "test_mAP": _f(r.get("test_mAP")),
            }
    return grid


def draw_sweep1_panel(ax, grid, metric_key, zlabel, zlim, zticks, norm_vmin, norm_vmax, cmap):
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=norm_vmin, vmax=norm_vmax)
    z_bottom = zlim[0]
    nx, ny = len(S1_X), len(S1_Y)
    Z = np.full((nx, ny), np.nan)
    for xi in range(nx):
        for yi in range(ny):
            g = grid.get((S1_X[xi], S1_Y[yi]), {})
            v = g.get(metric_key)
            if v is not None:
                Z[xi, yi] = v

    ax.set_xlabel(r"$\lambda_{\varnothing}$")
    ax.set_ylabel(r"$\lambda_{\mathrm{adv}}$")
    ax.set_zlabel(zlabel)
    ax.zaxis.labelpad = 6
    ax.set_xticks(range(nx))
    ax.set_yticks(range(ny))
    ax.set_xticklabels([f"{v:g}" for v in S1_X], rotation=0)
    ax.set_yticklabels([f"{v:g}" for v in S1_Y], rotation=0)
    ax.set_zlim(zlim)
    if zticks is not None:
        ax.set_zticks(zticks)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_proj_type("ortho")
    ax.dist = 12
    ax.set_box_aspect((1, 1, 0.65))
    ax.computed_zorder = False

    bars = []
    for xi in range(nx):
        for yi in reversed(range(ny)):
            z = Z[xi, yi]
            missing = np.isnan(z)
            zf = float(z) if not missing else z_bottom
            h = max(zf - z_bottom, 1e-6)
            bars.append((xi, yi, zf, h, missing))

    for rank, (xi, yi, zf, h, missing) in enumerate(bars):
        if missing:
            hh = 1e-6
            fc = (0.86, 0.86, 0.86, 1.0)
            a = 0.18
        else:
            hh = h
            fc = cmap(norm(zf))
            a = ALPHA
        ax.bar3d(
            xi, yi, z_bottom, DX, DY, hh,
            color=fc, alpha=a,
            edgecolor=EDGE_RGBA, linewidth=LW,
            shade=False,
        )


def draw_sweep2_panel(ax, grid, metric_key, zlabel, zlim, norm_vmin, norm_vmax, cmap):
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=norm_vmin, vmax=norm_vmax)
    z_bottom = zlim[0]
    nx, ny = len(S2_X), len(S2_Y)
    Z = np.full((nx, ny), np.nan)
    for xi in range(nx):
        for yi in range(ny):
            g = grid.get((S2_X[xi], S2_Y[yi]), {})
            v = g.get(metric_key)
            if v is not None:
                Z[xi, yi] = v

    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$\lambda_{\mathrm{cons}}$")
    ax.set_zlabel(zlabel)
    ax.zaxis.labelpad = 6
    ax.set_xticks(range(nx))
    ax.set_yticks(range(ny))
    ax.set_xticklabels([f"{v:g}" for v in S2_X], rotation=0)
    ax.set_yticklabels(S2_Y_LABELS, rotation=0)
    ax.set_zlim(zlim)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_proj_type("ortho")
    ax.dist = 12
    ax.set_box_aspect((1, 1, 0.65))
    ax.computed_zorder = False

    bars = []
    for xi in reversed(range(nx)):
        for yi in reversed(range(ny)):
            z = Z[xi, yi]
            missing = np.isnan(z)
            zf = float(z) if not missing else z_bottom
            h = max(zf - z_bottom, 1e-6)
            bars.append((xi, yi, zf, h, missing))

    for rank, (xi, yi, zf, h, missing) in enumerate(bars):
        if missing:
            hh = 1e-6
            fc = (0.86, 0.86, 0.86, 1.0)
            a = 0.18
        else:
            hh = h
            fc = cmap(norm(zf))
            a = ALPHA
        x_phys = nx - 1 - xi
        ax.bar3d(
            x_phys, yi, z_bottom, DX, DY, hh,
            color=fc, alpha=a,
            edgecolor=EDGE_RGBA, linewidth=LW,
            shade=False,
        )


def main():
    apply_tiu_style()
    grid1 = load_sweep1()
    grid2 = load_sweep2()
    cmap = pastelize_cmap("viridis", 0.35)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 5))
    ax0 = fig.add_subplot(141, projection="3d")
    ax1 = fig.add_subplot(142, projection="3d")
    ax2 = fig.add_subplot(143, projection="3d")
    ax3 = fig.add_subplot(144, projection="3d")

    # (a) Sweep1 DropR
    draw_sweep1_panel(
        ax0, grid1, "DropR",
        "DropR", (0.0, 0.9), [0.0, 0.3, 0.6, 0.9], 0.0, 0.9, cmap,
    )
    ax0.set_title("(a) Sweep1 DropR")
    # (b) Sweep1 Ret mAP
    draw_sweep1_panel(
        ax1, grid1, "retain_mAP",
        "Ret mAP", (0.88, 0.96), [0.88, 0.90, 0.92, 0.94, 0.96], 0.88, 0.96, cmap,
    )
    ax1.set_title("(b) Sweep1 Ret mAP")
    # (c) Sweep2 Fgt mAP
    draw_sweep2_panel(
        ax2, grid2, "forget_mAP",
        "Fgt mAP", (0.25, 0.60), 0.25, 0.60, cmap,
    )
    ax2.set_title("(c) Sweep2 Fgt mAP")
    # (d) Sweep2 Test mAP
    draw_sweep2_panel(
        ax3, grid2, "test_mAP",
        "Test mAP", (0.72, 0.82), 0.72, 0.82, cmap,
    )
    ax3.set_title("(d) Sweep2 Test mAP")

    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.92, wspace=0.08)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tiu(fig, str(FIG_PATH), pad_inches=0.02)
    print(f"[OK] {FIG_PATH}")


if __name__ == "__main__":
    main()
