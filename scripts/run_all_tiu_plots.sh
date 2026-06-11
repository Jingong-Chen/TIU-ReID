#!/usr/bin/env bash
# Run all TIU-ReID figure scripts. Requires PYTHONPATH="${REPO}".
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO}:${PYTHONPATH}"
cd "$REPO"
python scripts/plot_tiu_main_bars.py
python scripts/plot_tiu_ablation_bars.py
python scripts/plot_tiu_sweep_combined.py
python scripts/plot_tiu_multitarget.py
python scripts/plot_tiu_tune_heatmap.py
python scripts/plot_tiu_difficulty_scatter.py
python scripts/make_fig_index.py
echo "Done. Output: output/figures/tiu/*.pdf and FIG_INDEX.md"
