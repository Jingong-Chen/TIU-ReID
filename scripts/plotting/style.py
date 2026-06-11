"""Unified TIU-ReID plotting style. All figure scripts must call apply_tiu_style()."""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["mathtext.fontset"] = "stix"
import matplotlib.font_manager as _fm

# Light but distinguishable palette (ColorBrewer Set2 / Pastel1–like, edges darker)
PALETTE = [
    "#8dd3c7",  # teal pastel
    "#ffffb3",  # yellow pastel
    "#bebada",  # lavender
    "#fb8072",  # salmon
    "#80b1d3",  # sky blue
    "#fdb462",  # orange
    "#b3de69",  # light green
    "#fccde5",  # pink
    "#bc80bd",  # mauve
    "#ccebc5",  # pale green
]
# Darker edge colors for bars/lines (same hue, darker)
PALETTE_EDGE = [
    "#5a9d8f", "#c9c966", "#8e8db8", "#c95a4d", "#5a8fb1",
    "#c78d3a", "#8db84a", "#c99bb8", "#8d5a8d", "#8db87d",
]


def apply_tiu_style() -> None:
    """Apply global TIU-ReID figure style. Call at start of every plotting script."""
    _fam = "Times New Roman"
    if not any(f.name == _fam for f in _fm.fontManager.ttflist):
        _fam = "serif"
    mpl.rcParams["font.family"] = _fam
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["lines.markersize"] = 5
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["axes.axisbelow"] = True


def color(i: int, edge: bool = False) -> str:
    """Palette color by index. edge=True for darker border."""
    arr = PALETTE_EDGE if edge else PALETTE
    return arr[i % len(arr)]


def save_tiu(
    fig: "matplotlib.figure.Figure",
    path: str,
    png_too: bool = False,
    extra_artists: list | None = None,
    pad_inches: float | None = None,
) -> None:
    """Save figure as PDF (and optionally PNG 300dpi). bbox_inches='tight'. Use extra_artists so fig.text etc. are not clipped."""
    kwargs = {"bbox_inches": "tight", "pad_inches": pad_inches if pad_inches is not None else 0.05}
    if extra_artists:
        kwargs["bbox_extra_artists"] = extra_artists
    fig.savefig(path, **kwargs)
    if png_too:
        base = path.rsplit(".", 1)[0]
        fig.savefig(f"{base}.png", **{**kwargs, "dpi": 300})
    plt.close(fig)


def run_pdffonts_check(pdf_path: str | Path) -> str:
    """Run pdffonts on PDF; return summary. Warn if Type 3 fonts found."""
    p = Path(pdf_path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        return "skip (not PDF)"
    try:
        out = subprocess.run(
            ["pdffonts", str(p)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = (out.stdout or "") + (out.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"pdffonts not run: {e}"
    if "Type 3" in text or "Type3" in text:
        return "WARN: Type 3 font(s) present (embedding issue)"
    return "OK (no Type 3)"


def setup_tiu_subplots(nrows: int, ncols: int, figsize: tuple[float, float] | None = None) -> tuple["matplotlib.figure.Figure", np.ndarray]:
    """Create subplots, apply grid, hide top/right spines."""
    if figsize is None:
        figsize = (5.0 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.atleast_2d(axes)
    for ax in np.array(axes).flat:
        ax.grid(True, alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig, axes
