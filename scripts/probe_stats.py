from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    items = []
    with path.open("r") as f:
        for line in f:
            d = json.loads(line)
            items.append(d)
    return items


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def cam_distribution(items):
    return dict(Counter(int(x["camid"]) for x in items))


def compute_stats(query_items, gallery_items):
    gallery_by_pid = {}
    for g in gallery_items:
        gallery_by_pid.setdefault(int(g["pid"]), []).append(g)

    valid_queries = 0
    pos_counts = []
    neg_counts = []

    for q in query_items:
        q_pid = int(q["pid"])
        q_cam = int(q["camid"])
        g_list = gallery_by_pid.get(q_pid, [])
        pos = sum(1 for g in g_list if int(g["camid"]) != q_cam)
        # valid gallery excludes same pid+cam
        valid_gallery = [
            g for g in gallery_items if not (int(g["pid"]) == q_pid and int(g["camid"]) == q_cam)
        ]
        neg = sum(1 for g in valid_gallery if int(g["pid"]) != q_pid)
        pos_counts.append(pos)
        neg_counts.append(neg)
        if pos > 0:
            valid_queries += 1

    total_q = len(query_items)
    valid_ratio = valid_queries / total_q if total_q > 0 else 0.0
    return {
        "n_query": total_q,
        "n_gallery": len(gallery_items),
        "valid_query": valid_queries,
        "valid_query_ratio": valid_ratio,
        "pos_per_query_avg": sum(pos_counts) / total_q if total_q > 0 else 0.0,
        "pos_per_query_min": min(pos_counts) if pos_counts else 0,
        "pos_per_query_max": max(pos_counts) if pos_counts else 0,
        "neg_per_query_avg": sum(neg_counts) / total_q if total_q > 0 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--split_dir", required=False)
    args = ap.parse_args()

    probe_dir = Path(args.probe_dir)
    retain_q = load_jsonl(probe_dir / "retain_query.jsonl")
    retain_g = load_jsonl(probe_dir / "retain_gallery.jsonl")
    forget_q = load_jsonl(probe_dir / "forget_query.jsonl")
    forget_g = load_jsonl(probe_dir / "forget_gallery.jsonl")

    stats = {
        "retain": {
            "query_cam_dist": cam_distribution(retain_q),
            "gallery_cam_dist": cam_distribution(retain_g),
            "validity": compute_stats(retain_q, retain_g),
        },
        "forget": {
            "query_cam_dist": cam_distribution(forget_q),
            "gallery_cam_dist": cam_distribution(forget_g),
            "validity": compute_stats(forget_q, forget_g),
        },
    }

    probe_cfg = _read_json(probe_dir / "probe_config.json") if (probe_dir / "probe_config.json").exists() else None
    if probe_cfg:
        stats["probe_config"] = {
            "enforce_cross_cam": probe_cfg.get("enforce_cross_cam"),
            "forget_distractor_mode": probe_cfg.get("forget_distractor_mode"),
            "forget_distractor_seed": probe_cfg.get("forget_distractor_seed"),
            "forget_distractor_ids": probe_cfg.get("forget_distractor_ids"),
        }

    if args.split_dir:
        split_dir = Path(args.split_dir)
        forget_ids = set(int(x) for x in split_dir.joinpath("forget_ids.txt").read_text().split())
        retain_ids = set(int(x) for x in split_dir.joinpath("retain_ids.txt").read_text().split())
        distractor_file = probe_dir / "forget_distractor_ids.txt"
        distractors = []
        if distractor_file.exists():
            distractors = [int(x) for x in distractor_file.read_text().split() if x.strip()]
        stats["distractors"] = {
            "count": len(distractors),
            "in_forget": len(set(distractors) & forget_ids),
            "in_retain": len(set(distractors) & retain_ids),
        }

    out = probe_dir / "probe_stats.json"
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print("[OK] probe_stats.json ->", out)


if __name__ == "__main__":
    main()

