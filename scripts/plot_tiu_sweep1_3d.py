"""Sweep1 3D: lambda_baseonly x lambda_adv. Data from all_runs_master.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.plotting.style import apply_tiu_style, save_tiu

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output"
CSV_PATH = OUT_DIR / "compare" / "all_runs_master.csv"
FIG_PATH = OUT_DIR / "figures" / "tiu" / "fig_sweep1_3d.pdf"

X_VALS = [0.1, 0.25, 0.5, 1.0, 2.0]
Y_VALS = [1.0, 0.5, 0.2, 0.1, 0.05]
ELEV, AZIM = 42, -60
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


def main():
    apply_tiu_style()
    grid = load_sweep1()

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(16, 6))
    ax0 = fig.add_subplot(121, projection="3d")
    ax1 = fig.add_subplot(122, projection="3d")

    nx, ny = len(X_VALS), len(Y_VALS)
    Z_dropr = np.full((nx, ny), np.nan)
    Z_ret = np.full((nx, ny), np.nan)

    for xi in range(nx):
        for yi in range(ny):
            xv, yv = X_VALS[xi], Y_VALS[yi]
            g = grid.get((xv, yv), {})
            if g.get("DropR") is not None:
                Z_dropr[xi, yi] = g["DropR"]
            if g.get("retain_mAP") is not None:
                Z_ret[xi, yi] = g["retain_mAP"]

    cmap = pastelize_cmap("viridis", 0.35)

    def draw_panel(ax, Z, zlabel, zlim, zticks, norm_vmin, norm_vmax, ylabel_math):
        norm = Normalize(vmin=norm_vmin, vmax=norm_vmax)
        z_bottom = zlim[0]
        ax.set_xlabel(r"$\lambda_{\varnothing}$")
        ax.set_ylabel(ylabel_math)
        ax.set_zlabel(zlabel)
        ax.zaxis.labelpad = 6
        ax.set_xticks(range(nx))
        ax.set_yticks(range(ny))
        ax.set_xticklabels([f"{v:g}" for v in X_VALS], rotation=0)
        ax.set_yticklabels([f"{v:g}" for v in Y_VALS], rotation=0)
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
            pc = ax.bar3d(
                xi, yi, z_bottom, DX, DY, hh,
                color=fc, alpha=a,
                edgecolor=EDGE_RGBA, linewidth=LW,
                shade=False,
            )
            pc.set_zorder(rank + 10)

    draw_panel(
        ax0, Z_dropr,
        "DropR", (0.0, 0.9), [0.0, 0.3, 0.6, 0.9], 0.0, 0.9,
        ylabel_math=r"$\lambda_{\mathrm{adv}}$",
    )
    draw_panel(
        ax1, Z_ret,
        "Ret mAP", (0.88, 0.96), [0.88, 0.90, 0.92, 0.94, 0.96], 0.88, 0.96,
        ylabel_math=r"$\lambda_{\mathrm{adv}}$",
    )

    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.98, wspace=0.10)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_tiu(fig, str(FIG_PATH), pad_inches=0.02)
    print(f"[OK] {FIG_PATH}")


if __name__ == "__main__":
    main()
