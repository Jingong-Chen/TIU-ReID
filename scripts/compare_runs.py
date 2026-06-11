from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="MVP tags under output/mvp/")
    ap.add_argument("--out_dir", default=None, help="Defaults to output/compare")
    ap.add_argument("--name", default=None, help="Optional name suffix for outputs")
    ap.add_argument("--min_retain", type=float, default=0.0, help="Filter runs with retain mAP < min_retain")
    ap.add_argument("--sort_by", default="membership_auc_abs", help="Column to sort by")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_dir) if args.out_dir else repo_root / "output" / "compare"
    out_root.mkdir(parents=True, exist_ok=True)
    name = args.name or datetime.now().strftime("compare_%Y%m%d_%H%M%S")

    rows = []
    for tag in args.tags:
        mvp_dir = repo_root / "output" / "mvp" / tag
        summary = _read_json(mvp_dir / "summary.json") or {}
        sanity = _read_json(mvp_dir / "sanity_report.json") or {}

        unlearn_retain = summary.get("unlearn_retain") or {}
        unlearn_forget = summary.get("unlearn_forget") or {}
        test_unlearn = ((summary.get("test_eval") or {}).get("unlearn")) or {}

        row = {
            "tag": tag,
            "retain_mAP": unlearn_retain.get("mAP"),
            "forget_mAP": unlearn_forget.get("mAP"),
            "test_mAP": test_unlearn.get("mAP"),
            "membership_auc_abs": summary.get("membership_auc_abs"),
            "membership_acc_abs": summary.get("membership_acc_abs"),
            "a2_step_to_recover_50pct": ((summary.get("attack_A2") or {}).get("step_to_recover_50pct")),
            "score_pass_target": ((sanity.get("scorecard") or {}).get("pass_all_target")),
            "score_pass_relaxed": ((sanity.get("scorecard") or {}).get("pass_all_relaxed")),
            "run_args_path": str(mvp_dir / "unlearn" / "run_args.json"),
        }
        rows.append(row)

    if args.min_retain > 0:
        rows = [r for r in rows if r.get("retain_mAP") is not None and r["retain_mAP"] >= args.min_retain]

    sort_key = args.sort_by
    if sort_key in rows[0] if rows else []:
        rows = sorted(rows, key=lambda r: (r.get(sort_key) is None, r.get(sort_key)))

    csv_path = out_root / f"{name}.csv"
    md_path = out_root / f"{name}.md"

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "tag",
            "retain_mAP",
            "forget_mAP",
            "test_mAP",
            "membership_auc_abs",
            "membership_acc_abs",
            "a2_step_to_recover_50pct",
            "score_pass_target",
            "score_pass_relaxed",
            "run_args_path",
        ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    lines = []
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(k)) for k in fieldnames) + " |")
    md_path.write_text("\n".join(lines) + "\n")

    print("[OK] compare csv ->", csv_path)
    print("[OK] compare md  ->", md_path)


if __name__ == "__main__":
    main()


