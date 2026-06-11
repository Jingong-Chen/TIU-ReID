from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from unlearning_reid.datasets.common import group_by_pid, read_train_items


def load_ids(p: Path) -> set[int]:
    text = p.read_text().strip()
    if not text:
        return set()
    return set(int(x) for x in text.splitlines() if x.strip())


def build_probe(by_pid, ids, seed: int, query_per_pid: int = 1, enforce_cross_cam: bool = True):
    rnd = random.Random(seed)
    query, gallery = [], []
    dropped = []
    for pid in sorted(ids):
        lst = list(by_pid[pid])
        if enforce_cross_cam:
            cam_map = {}
            for it in lst:
                cam_map.setdefault(it.camid, []).append(it)
            cams = list(cam_map.keys())
            if len(cams) < 2:
                dropped.append(pid)
                continue
            rnd.shuffle(cams)
            max_q = min(query_per_pid, len(cams) - 1)
            q_cams = cams[:max_q]
            for cam in q_cams:
                items = list(cam_map[cam])
                rnd.shuffle(items)
                query.append(items[0])
            for cam in cams:
                if cam in q_cams:
                    continue
                gallery += cam_map[cam]
        else:
            rnd.shuffle(lst)
            q = lst[:query_per_pid]
            g = lst[query_per_pid:]
            if len(g) == 0:
                q, g = lst[:1], lst[1:]
            query += q
            gallery += g
    return query, gallery, dropped


def dump_list(items, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for it in items:
            f.write(json.dumps({"path": it.path, "pid": it.pid, "camid": it.camid}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="market1501", choices=["market1501", "dukemtmc-reid"])
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--split_dir", required=True, help="output of make_splits.py (contains retain/forget ids)")
    ap.add_argument(
        "--out_dir", default=None, help="defaults to $REID_OUTPUT_DIR/probes/<dataset>/seedX"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query_per_pid", type=int, default=1)
    ap.add_argument("--enforce_cross_cam", type=int, default=1, help="1: enforce cross-cam, 0: disable")
    ap.add_argument(
        "--forget_distractor_ids",
        type=int,
        default=0,
        help="Number of retain IDs to add into forget gallery as distractors.",
    )
    ap.add_argument(
        "--distractor_mode",
        choices=["random", "hard"],
        default="random",
        help="random: sample from retain IDs; hard: pick nearest retain IDs by teacher centroids",
    )
    ap.add_argument(
        "--feat_train_npz",
        default=None,
        help="Required for hard mode. Teacher features npz for train set.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root) if args.data_root else Path(os.environ["REID_DATA_DIR"])
    split_dir = Path(args.split_dir)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(os.environ["REID_OUTPUT_DIR"]) / "probes" / args.dataset / split_dir.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_train_items(args.dataset, data_root)
    by_pid = group_by_pid(items)

    retain_ids = load_ids(split_dir / "retain_ids.txt")
    forget_ids = load_ids(split_dir / "forget_ids.txt")

    if not retain_ids or not forget_ids:
        raise RuntimeError("retain_ids or forget_ids is empty; check split_dir.")

    enforce_cross_cam = bool(args.enforce_cross_cam)
    rq, rg, dropped_r = build_probe(
        by_pid, retain_ids, args.seed, args.query_per_pid, enforce_cross_cam=enforce_cross_cam
    )
    fq, fg, dropped_f = build_probe(
        by_pid, forget_ids, args.seed, args.query_per_pid, enforce_cross_cam=enforce_cross_cam
    )

    distractor_ids = []
    distractor_seed = None
    distractor_mode = args.distractor_mode
    if args.forget_distractor_ids > 0:
        retain_list = sorted(retain_ids)
        if distractor_mode == "hard":
            if not args.feat_train_npz:
                raise RuntimeError("hard distractor mode requires --feat_train_npz")
            import numpy as np

            d = np.load(args.feat_train_npz, allow_pickle=True)
            feats = d["feats"].astype(np.float32)
            pids = d["pids"].astype(np.int64)
            uniq = sorted(set(int(x) for x in pids.tolist()))
            pid_to_idx = {pid: i for i, pid in enumerate(uniq)}
            centroids = np.zeros((len(uniq), feats.shape[1]), dtype=np.float32)
            counts = np.zeros((len(uniq),), dtype=np.int64)
            for feat, pid in zip(feats, pids):
                i = pid_to_idx[int(pid)]
                centroids[i] += feat
                counts[i] += 1
            counts = np.maximum(counts, 1)
            centroids = centroids / counts[:, None]
            centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
            forget_centroids = [centroids[pid_to_idx[pid]] for pid in sorted(forget_ids)]
            if not forget_centroids:
                raise RuntimeError("No forget centroids found for hard distractors.")
            forget_mean = np.mean(np.stack(forget_centroids, axis=0), axis=0)
            retain_centroids = {pid: centroids[pid_to_idx[pid]] for pid in retain_list}
            sims = [(pid, float(forget_mean @ retain_centroids[pid])) for pid in retain_list]
            sims.sort(key=lambda x: (-x[1], x[0]))
            distractor_ids = [pid for pid, _ in sims[: args.forget_distractor_ids]]
        else:
            distractor_seed = args.seed + 1337
            rnd = random.Random(distractor_seed)
            if args.forget_distractor_ids >= len(retain_list):
                distractor_ids = retain_list
            else:
                distractor_ids = sorted(rnd.sample(retain_list, args.forget_distractor_ids))

    fg_distractors = []
    for pid in distractor_ids:
        fg_distractors += list(by_pid[pid])

    if fg_distractors:
        fg = fg + fg_distractors
        (out_dir / "forget_distractor_ids.txt").write_text(
            "\n".join(map(str, distractor_ids)) + "\n"
        )

    dump_list(rq, out_dir / "retain_query.jsonl")
    dump_list(rg, out_dir / "retain_gallery.jsonl")
    dump_list(fq, out_dir / "forget_query.jsonl")
    dump_list(fg, out_dir / "forget_gallery.jsonl")

    meta = {
        "dataset": args.dataset,
        "seed": args.seed,
        "query_per_pid": args.query_per_pid,
        "enforce_cross_cam": enforce_cross_cam,
        "split_dir": str(split_dir),
        "counts": {
            "retain_query": len(rq),
            "retain_gallery": len(rg),
            "forget_query": len(fq),
            "forget_gallery": len(fg),
            "forget_gallery_distractors": len(fg_distractors),
        },
        "forget_distractor_ids": len(distractor_ids),
        "forget_distractor_seed": distractor_seed,
        "forget_distractor_mode": distractor_mode,
        "forget_distractor_feat_npz": str(args.feat_train_npz) if args.feat_train_npz else None,
        "forget_distractor_pool": "retain_ids",
        "forget_distractor_excludes_forget": True,
        "dropped_pids": {
            "retain": len(dropped_r),
            "forget": len(dropped_f),
        },
    }
    (out_dir / "probe_config.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("[OK] Wrote probes to:", out_dir)


if __name__ == "__main__":
    main()


