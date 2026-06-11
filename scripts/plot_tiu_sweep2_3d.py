"""Sweep2 3D: target_base_scale x lambda_consist. Data from all_runs_master.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.plotting.style import apply_tiu_style, save_tiu

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output"
CSV_PATH = OUT_DIR / "compare" / "all_runs_master.csv"
FIG_PATH = OUT_DIR / "figures" / "tiu" / "fig_sweep2_3d.pdf"

X_VALS = [0.0, 0.5, 1.0, 1.5, 2.0]
Y_VALS = [5.0, 2.0, 1.0, 0.5, 0.0]
Y_LABELS = ["5", "2", "1", ".5", "0"]
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
            grid[(tbs, lc)] = {"forget_mAP": _f(r.get("forget_mAP")), "test_mAP": _f(r.get("test_mAP"))}
    return grid


def main():
    apply_tiu_style()
    grid = load_sweep2()

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(16, 6))
    ax0 = fig.add_subplot(121, projection="3d")
    ax1 = fig.add_subplot(122, projection="3d")

    nx, ny = len(X_VALS), len(Y_VALS)
    Z_fgt = np.full((nx, ny), np.nan)
    Z_test = np.full((nx, ny), np.nan)
    for xi in range(nx):
        for yi in range(ny):
            xv, yv = X_VALS[xi], Y_VALS[yi]
            g = grid.get((xv, yv), {})
            if g.get("forget_mAP") is not None:
                Z_fgt[xi, yi] = g["forget_mAP"]
            if g.get("test_mAP") is not None:
                Z_test[xi, yi] = g["test_mAP"]

    cmap = pastelize_cmap("viridis", 0.35)

    def draw_panel(ax, Z, zlabel, zlim, norm_vmin, norm_vmax):
        norm = Normalize(vmin=norm_vmin, vmax=norm_vmax)
        z_bottom = zlim[0]
        ax.set_xlabel(r"$s$")
        ax.set_ylabel(r"$\lambda_{\mathrm{cons}}$")
        ax.set_zlabel(zlabel)
        ax.zaxis.labelpad = 6
        ax.set_xticks(range(nx))
        ax.set_yticks(range(ny))
        ax.set_xticklabels([f"{v:g}" for v in X_VALS], rotation=0)
        ax.set_yticklabels(Y_LABELS, rotation=0)
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
            pc = ax.bar3d(
                x_phys, yi, z_bottom, DX, DY, hh,
                color=fc, alpha=a,
                edgecolor=EDGE_RGBA, linewidth=LW,
                shade=False,
            )
            pc.set_zorder(rank + 10)

    draw_panel(ax0, Z_fgt, "Fgt mAP", (0.25, 0.60), 0.25, 0.60)
    draw_panel(ax1, Z_test, "Test mAP", (0.72, 0.82), 0.72, 0.82)

    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.98, wspace=0.10)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tiu(fig, str(FIG_PATH), pad_inches=0.02)
    print(f"[OK] {FIG_PATH}")


if __name__ == "__main__":
    main()
