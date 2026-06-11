#!/usr/bin/env python3
"""
从实验日志中解析 epoch 与 loss，绘制「epoch 为横坐标、loss 为纵坐标」的折线图。
支持两种日志格式：
  [EPOCH N] loss=X.XXXX
  [EPOCH N] sched_factor=... total=X task_id=X base_anti=X baseonly=X consist=X ...
默认只画一个主要实验（Full），并画出各损失分量。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 主要实验的匹配名（只画这一个时用）
DEFAULT_MAIN_RUN = "2.7 Full"
# 要绘制的损失分量（按日志里的 key）
LOSS_KEYS = ["total", "task_id", "base_anti", "baseonly", "consist"]
LOSS_LABELS = {
    "total": "Total",
    "task_id": "Task ID (L_id)",
    "base_anti": "L_D (Discriminator / Base Anti)",
    "baseonly": "Base Only (forget)",
    "consist": "Consist",
}


def parse_log_detailed(log_path: Path):
    """
    解析带各损失分量的日志行，返回列表 of (run_name, [(epoch, {key: value}), ...])。
    仅解析形如 [EPOCH N] ... total=X task_id=X base_anti=X baseonly=X consist=X ... 的行。
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    section_pat = re.compile(r"^\[([\d.]+)\]\s*(.+?)(?:\.\.\.)?\s*$")
    # 匹配 [EPOCH N] 后面一系列 key=value（浮点数）
    epoch_detail_pat = re.compile(
        r"^\[EPOCH\s+(\d+)\]\s+(.+)$"
    )

    runs = []
    current_name = None
    current_points = []

    def flush_run():
        nonlocal current_points
        if current_name and current_points:
            runs.append((current_name, list(current_points)))
        current_points = []

    for line in lines:
        line_strip = line.strip()
        m = section_pat.match(line_strip)
        if m:
            flush_run()
            current_name = f"{m.group(1)} {m.group(2).strip()}"
            continue
        m = epoch_detail_pat.match(line_strip)
        if not m:
            continue
        epoch = int(m.group(1))
        rest = m.group(2)
        # 解析 key=float
        pairs = re.findall(r"(\w+)=([\d.]+)", rest)
        d = {}
        for k, v in pairs:
            try:
                d[k] = float(v)
            except ValueError:
                pass
        if "total" in d:
            current_points.append((epoch, d))

    flush_run()
    return runs


def plot_loss_components(
    runs: list[tuple[str, list[tuple[int, dict]]]],
    out_path: Path,
    run_filter: str = "Full",
    title: str | None = None,
):
    """只画一个实验，横轴 epoch，纵轴数值，多条线为各损失分量。"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    chosen = [(n, pts) for n, pts in runs if run_filter in n and pts]
    if not chosen:
        raise SystemExit(f"未找到包含 '{run_filter}' 且带详细 loss 的实验，请检查日志。")
    run_name, points = chosen[0]
    if not points:
        raise SystemExit("该实验无 [EPOCH N] ... total= ... 格式行。")

    epochs = [p[0] for p in points]
    keys = [k for k in LOSS_KEYS if any(k in p[1] for p in points)]
    if not keys:
        keys = [k for k in points[0][1].keys() if k not in ("sched_factor",)]
        if not keys:
            raise SystemExit("该实验未解析到任何损失键。")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(keys), 1)))
    for i, key in enumerate(keys):
        vals = [p[1].get(key, float("nan")) for p in points]
        label = LOSS_LABELS.get(key, key)
        ax.plot(epochs, vals, "o-", label=label, color=colors[i % len(colors)], markersize=6)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title or f"Loss components — {run_name}", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path}")


def parse_log(log_path: Path):
    """解析日志，返回列表 of (run_name, [(epoch, loss), ...])。"""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    section_pat = re.compile(r"^\[([\d.]+)\]\s*(.+?)(?:\.\.\.)?\s*$")
    epoch_loss_pat = re.compile(r"^\[EPOCH\s+(\d+)\]\s+loss=([\d.]+)\s*$")
    epoch_total_pat = re.compile(r"^\[EPOCH\s+(\d+)\].*?\btotal=([\d.]+)")

    runs = []
    current_name = "Run"
    current_points = []

    def flush_run():
        nonlocal current_points
        if current_points:
            runs.append((current_name, current_points))
            current_points = []

    for line in lines:
        line_strip = line.strip()
        m = section_pat.match(line_strip)
        if m:
            flush_run()
            current_name = f"{m.group(1)} {m.group(2).strip()}"
            continue
        m = epoch_loss_pat.match(line_strip)
        if m:
            epoch, loss = int(m.group(1)), float(m.group(2))
            current_points.append((epoch, loss))
            continue
        m = epoch_total_pat.match(line_strip)
        if m:
            epoch, loss = int(m.group(1)), float(m.group(2))
            current_points.append((epoch, loss))
            continue

    flush_run()
    return runs


def plot_loss_curves(
    runs: list[tuple[str, list[tuple[int, float]]]],
    out_path: Path,
    title: str = "Loss vs Epoch",
    single_run: str | None = None,
):
    """绘制折线图：横轴 epoch，纵轴 loss。"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))

    if single_run:
        runs = [(n, pts) for n, pts in runs if single_run in n or n == single_run]
    if not runs:
        plt.close()
        raise SystemExit("没有解析到任何 (epoch, loss) 数据，请检查日志格式或 --single-run。")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))
    for i, (name, points) in enumerate(runs):
        if not points:
            continue
        epochs = [p[0] for p in points]
        losses = [p[1] for p in points]
        label = name if len(name) <= 40 else name[:37] + "..."
        ax.plot(epochs, losses, "o-", label=label, color=colors[i % len(colors)], markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="从实验日志绘制 Loss vs Epoch 折线图")
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output" / "experiments_full.log",
        help="日志文件路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出图片路径（默认 loss_components.png）",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="图标题",
    )
    parser.add_argument(
        "--run",
        default="Full",
        help="只画哪一个实验（名称包含该字符串），默认 Full（主要实验）",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="改为画所有实验的总 loss 曲线（每条实验一条线），不画各损失分量",
    )
    parser.add_argument(
        "--single-run",
        default=None,
        help="[仅 --all-runs 时] 只绘制名称包含该字符串的一条总 loss 曲线",
    )
    args = parser.parse_args()

    log_path = args.log
    if not log_path.is_file():
        raise SystemExit(f"日志文件不存在: {log_path}")

    out_path = args.output
    if out_path is None:
        out_path = log_path.parent / ("loss_vs_epoch.png" if args.all_runs else "loss_components.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.all_runs:
        runs = parse_log(log_path)
        if not runs:
            raise SystemExit("未在日志中解析到 [EPOCH N] loss= 或 total= 行。")
        print(f"共解析到 {len(runs)} 条训练曲线。")
        plot_loss_curves(runs, out_path, title=args.title or "Loss vs Epoch", single_run=args.single_run)
    else:
        runs = parse_log_detailed(log_path)
        if not runs:
            raise SystemExit("未在日志中解析到 [EPOCH N] ... total= task_id= ... 格式行。")
        print(f"共解析到 {len(runs)} 个带详细 loss 的实验，绘制主要实验: {args.run}")
        plot_loss_components(runs, out_path, run_filter=args.run, title=args.title)


if __name__ == "__main__":
    main()
