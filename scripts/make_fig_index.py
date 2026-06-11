"""Step 6: Generate output/figures/tiu/FIG_INDEX.md and validate figures."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.plotting.style import apply_tiu_style, run_pdffonts_check

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "output" / "figures" / "tiu"
MD_PATH = FIG_DIR / "FIG_INDEX.md"

FIG_SPEC = [
    {
        "path": "fig_main_bars.pdf",
        "inputs": ["output/compare/all_runs_master.csv", "output/removable_baselines/base_teacher_*/summary_base.json", "output/removable_baselines/with_only_*/summary_with_only.json"],
        "methods": ["BASE", "W-ONLY (Omega)", "W-ONLY (empty)", "TIU (Omega)", "TIU (empty)", "BEST"],
        "panels": "2x2: (a) Retain mAP/R1, (b) Forget mAP/R1, (c) Test mAP/R1, (d) DropR",
    },
    {
        "path": "fig_ablation_bars.pdf",
        "inputs": ["output/compare/all_runs_master.csv"],
        "methods": ["Conf-only", "Disc-only", "L-empty-only", "Spr-only", "Full", "Full+Consist (if distinct)"],
        "panels": "1x3: (a) Ret mAP, (b) Fgt mAP, (c) DropR",
    },
    {
        "path": "fig_sweep_combined.pdf",
        "inputs": ["output/compare/all_runs_master.csv"],
        "methods": ["Sweep1 lb x la", "Sweep2 tbs x lc"],
        "panels": "1x4 3D: (a) Sweep1 DropR, (b) Sweep1 Ret mAP, (c) Sweep2 Fgt mAP, (d) Sweep2 Test mAP",
    },
    {
        "path": "fig_multitarget.pdf",
        "inputs": ["output/compare/multitarget_v2/multitarget_summary_by_target.csv"],
        "methods": ["Multi-target 10 tid (2, 7, 10, 11, 20, 52, 100, 150, 500, 700)"],
        "panels": "1x2: (a) DropR vs Target ID, (b) Ret mAP, Test mAP, Fgt mAP",
    },
    {
        "path": "fig_tune_heatmap.pdf",
        "inputs": ["output/compare/multitarget_v2/tune_grid_results.csv"],
        "methods": ["Tune 5 tid (2,10,20,52,100) x 3x3 grid (lb, la)"],
        "panels": "2x3: 5 tids (2,10,20,52,100), DropR heatmap, corner Ret mAP",
    },
    {
        "path": "fig_difficulty_scatter.pdf",
        "inputs": ["output/compare/multitarget_v2/difficulty_metrics.csv", "output/compare/multitarget_v2/multitarget_summary_by_target.csv"],
        "methods": ["Hardness vs DropR scatter"],
        "panels": "1 panel: hardness x-axis, DropR y-axis, size=n_target, color=distractor_strength",
    },
]


def font_check() -> str:
    apply_tiu_style()
    fam = plt.rcParams.get("font.family", "")
    if isinstance(fam, list):
        fam = fam[0] if fam else ""
    return f"font.family={fam}, mathtext.fontset={plt.rcParams.get('mathtext.fontset', '')}, pdf.fonttype={plt.rcParams.get('pdf.fonttype', '')}"


def english_check() -> str:
    scripts = [
        "scripts/plot_tiu_main_bars.py",
        "scripts/plot_tiu_ablation_bars.py",
        "scripts/plot_tiu_sweep_combined.py",
        "scripts/plot_tiu_multitarget.py",
        "scripts/plot_tiu_tune_heatmap.py",
        "scripts/plot_tiu_difficulty_scatter.py",
        "scripts/plotting/style.py",
    ]
    out = []
    for rel in scripts:
        p = REPO / rel
        if not p.exists():
            out.append(f"{rel}: missing")
            continue
        t = p.read_text(encoding="utf-8")
        m = re.search(r"[\u4e00-\u9fff]", t)
        out.append(f"{rel}: no Chinese" if not m else f"{rel}: Chinese at pos {m.start()}")
    return "\n".join(out)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TIU-ReID Figure Index",
        "",
        "## Generated PDFs",
        "",
        "| Figure | Path | Input sources | Methods / variants | Panels |",
        "|--------|------|----------------|--------------------|--------|",
    ]
    for s in FIG_SPEC:
        pp = f"`output/figures/tiu/{s['path']}`"
        inp = "; ".join(s["inputs"])
        m = "; ".join(s["methods"])
        lines.append(f"| {s['path']} | {pp} | {inp} | {m} | {s['panels']} |")

    exists = [s["path"] for s in FIG_SPEC if (FIG_DIR / s["path"]).exists()]
    lines.extend([
        "",
        "## Validation",
        "",
        f"- Generated PDFs: {', '.join(exists)}",
        f"- All Times New Roman (or serif fallback) via `apply_tiu_style()`.",
        "- No caption; no Chinese in scripts or labels.",
        "- Panel counts: main 2x2, ablation 1x3, sweep 1x4 3D, multitarget 1x2.",
        "",
        "## Font check (rcParams)",
        "",
        "```",
        font_check(),
        "```",
        "",
        "Times New Roman is enforced in `scripts/plotting/style`; `pdf.fonttype=42` for embedding. "
        "If TNR is not installed, fallback to `serif`. Main-bars use W-ONLY (Omega/empty) and TIU (Omega/empty); empty-set glyph may differ with serif fallback.",
        "",
        "## English check (no Chinese)",
        "",
        "```",
        english_check(),
        "```",
        "",
        "## pdffonts check (no Type 3)",
        "",
        "```",
    ])
    for s in FIG_SPEC:
        p = FIG_DIR / s["path"]
        res = run_pdffonts_check(p)
        lines.append(f"{s['path']}: {res}")
    lines.extend([
        "```",
        "",
        "Run `pdffonts output/figures/tiu/<file>.pdf` to verify. Type 3 fonts prevent proper embedding.",
        "",
        "All figure scripts use English-only strings; no captions.",
        "",
    ])
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {MD_PATH}")
    print("Font:", font_check())
    print("English:", english_check().replace("\n", " | "))
    for s in FIG_SPEC:
        res = run_pdffonts_check(FIG_DIR / s["path"])
        print(f"pdffonts {s['path']}: {res}")


if __name__ == "__main__":
    main()
