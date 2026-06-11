from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2_normalize(x: np.ndarray):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def read_ids(path: Path):
    return [int(x) for x in path.read_text().split() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_train_npz", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feat_path = Path(args.feat_train_npz)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(feat_path, allow_pickle=True)
    feats = d["feats"].astype(np.float32)
    pids = d["pids"].astype(np.int64)
    paths = d["paths"] if "paths" in d.files else None

    retain_ids = set(read_ids(split_dir / "retain_ids.txt"))
    forget_ids = set(read_ids(split_dir / "forget_ids.txt"))

    # retain centroids (L2)
    feats_l2 = l2_normalize(feats)
    retain_centroids = []
    retain_pids = []
    for pid in sorted(retain_ids):
        mask = pids == pid
        if not np.any(mask):
            continue
        c = feats_l2[mask].mean(axis=0)
        c = l2_normalize(c.reshape(1, -1))[0]
        retain_centroids.append(c)
        retain_pids.append(pid)
    retain_centroids = np.stack(retain_centroids, axis=0)
    retain_pids = np.array(retain_pids, dtype=np.int64)

    np.savez_compressed(
        out_dir / "retain_centroids.npz",
        centroids=retain_centroids.astype(np.float32),
        pids=retain_pids,
    )

    # forget samples: top-K retain centroids by cosine
    forget_mask = np.isin(pids, list(forget_ids))
    forget_indices = np.where(forget_mask)[0]
    forget_feats = feats_l2[forget_indices]

    sims = forget_feats @ retain_centroids.T
    k = min(args.topk, sims.shape[1])
    topk_idx = np.argsort(-sims, axis=1)[:, :k]

    records = []
    for i, gidx in enumerate(forget_indices.tolist()):
        topk_pids = [int(retain_pids[j]) for j in topk_idx[i].tolist()]
        rec = {
            "index": int(gidx),
            "pid": int(pids[gidx]),
            "topk_pids": topk_pids,
        }
        if paths is not None:
            rec["path"] = str(paths[gidx])
        records.append(rec)

    (out_dir / "forget_to_retain_topk.json").write_text(json.dumps(records, indent=2) + "\n")

    meta = {
        "feat_train_npz": str(feat_path),
        "split_dir": str(split_dir),
        "retain_centroids": str(out_dir / "retain_centroids.npz"),
        "forget_to_retain_topk": str(out_dir / "forget_to_retain_topk.json"),
        "retain_count": int(len(retain_pids)),
        "forget_samples": int(len(forget_indices)),
        "topk": int(k),
        "normalize": "l2",
        "seed": int(args.seed),
    }
    (out_dir / "neighbor_cache_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("[OK] neighbor cache ->", out_dir)


if __name__ == "__main__":
    main()


