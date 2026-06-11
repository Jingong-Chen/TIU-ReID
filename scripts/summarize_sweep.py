from __future__ import annotations

import argparse
import csv
import json
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
    ap.add_argument("--tags", nargs="*", default=None, help="explicit tags under output/mvp/")
    ap.add_argument("--prefix", default=None, help="glob prefix for tags")
    ap.add_argument("--out_dir", default=None, help="defaults to output/compare")
    ap.add_argument("--name", default="sweep_neighbor_attract")
    ap.add_argument("--include_fail", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_dir) if args.out_dir else repo_root / "output" / "compare"
    out_root.mkdir(parents=True, exist_ok=True)

    tags = []
    if args.tags:
        tags.extend(args.tags)
    if args.prefix:
        for p in sorted((repo_root / "output" / "mvp").glob(f"{args.prefix}*")):
            tags.append(p.name)
    tags = sorted(list(dict.fromkeys(tags)))

    rows = []
    for tag in tags:
        mvp_dir = repo_root / "output" / "mvp" / tag
        summary = _read_json(mvp_dir / "summary.json") or {}
        sanity = _read_json(mvp_dir / "sanity_report.json") or {}
        unlearn_retain = summary.get("unlearn_retain") or {}
        unlearn_forget = summary.get("unlearn_forget") or {}
        test_unlearn = (summary.get("test_eval") or {}).get("unlearn") or {}
        scorecard = sanity.get("scorecard") or {}
        retain_norm = sanity.get("retain_norm_stats") or {}
        retain_norm_query = retain_norm.get("query") or {}
        geom = sanity.get("forget_vs_retain_geometry") or {}
        pass_flag = bool(scorecard.get("pass_retain")) and not bool(scorecard.get("fail_retain_drift"))
        retain_delta = sanity.get("retain_delta_stats") or {}

        row = {
            "tag": tag,
            "retain_mAP": unlearn_retain.get("mAP"),
            "forget_mAP": unlearn_forget.get("mAP"),
            "test_mAP": test_unlearn.get("mAP"),
            "membership_auc_abs": summary.get("membership_auc_abs"),
            "membership_auc_abs_std": summary.get("membership_auc_abs_std"),
            "a2_step50": (summary.get("attack_A2") or {}).get("step_to_recover_50pct"),
            "retain_norm_mean": retain_norm_query.get("mean"),
            "retain_norm_p90": retain_norm_query.get("p90"),
            "retain_delta_mean": retain_delta.get("mean"),
            "retain_delta_p90": retain_delta.get("p90"),
            "forget_vs_retain_centroid_dist": geom.get("forget_vs_retain_centroid_cos_dist"),
            "PASS": pass_flag,
        }
        if args.include_fail or pass_flag:
            rows.append(row)

    rows = sorted(
        rows,
        key=lambda r: (
            not r["PASS"],
            (r["membership_auc_abs"] is None, r["membership_auc_abs"]),
            -(r["retain_mAP"] or 0.0),
            -(r["a2_step50"] or 0),
        ),
    )

    fieldnames = [
        "tag",
        "retain_mAP",
        "forget_mAP",
        "test_mAP",
        "membership_auc_abs",
        "membership_auc_abs_std",
        "a2_step50",
        "retain_norm_mean",
        "retain_norm_p90",
        "retain_delta_mean",
        "retain_delta_p90",
        "forget_vs_retain_centroid_dist",
        "PASS",
    ]

    csv_path = out_root / f"{args.name}.csv"
    md_path = out_root / f"{args.name}.md"

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

    print("[OK] sweep csv ->", csv_path)
    print("[OK] sweep md  ->", md_path)


if __name__ == "__main__":
    main()


