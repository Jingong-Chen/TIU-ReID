from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    feats = d["feats"].astype(np.float32)
    pids = d["pids"].astype(np.int64)
    return feats, pids


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_npz", required=True, help="Teacher feature npz (feats,pids)")
    ap.add_argument("--out_dir", required=True, help="Output directory under output/")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--cluster_k", type=int, default=20)
    ap.add_argument("--use_tsne", action="store_true", help="If set, run t-SNE on ID centroids")
    ap.add_argument("--focus_id", type=int, default=None, help="Optional ID to print neighbors")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats, pids = load_npz(Path(args.feat_npz))
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
    centroids = l2_normalize(centroids)

    sim = centroids @ centroids.T
    neighbors = {}
    for i, pid in enumerate(uniq):
        order = np.argsort(-sim[i])
        top = []
        for j in order:
            if i == j:
                continue
            top.append({"pid": int(uniq[j]), "sim": float(sim[i, j])})
            if len(top) >= args.topk:
                break
        neighbors[int(pid)] = top

    clusters = {int(pid): 0 for pid in uniq}
    coords_2d = pca_2d(centroids)
    method = "pca"

    try:
        from sklearn.cluster import KMeans  # type: ignore

        if args.cluster_k > 1:
            km = KMeans(n_clusters=args.cluster_k, random_state=0, n_init=10)
            labels = km.fit_predict(centroids)
            clusters = {int(pid): int(labels[i]) for i, pid in enumerate(uniq)}
    except Exception:
        pass

    if args.use_tsne:
        try:
            from sklearn.manifold import TSNE  # type: ignore

            tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=0)
            coords_2d = tsne.fit_transform(centroids)
            method = "tsne"
        except Exception:
            method = "pca"

    np.savez_compressed(out_dir / "id_centroids.npz", pids=np.array(uniq), centroids=centroids)
    (out_dir / "id_neighbors.json").write_text(json.dumps(neighbors, indent=2) + "\n")
    (out_dir / "id_clusters.json").write_text(json.dumps(clusters, indent=2) + "\n")

    with (out_dir / "id_embed_2d.csv").open("w") as f:
        f.write("pid,x,y,cluster\n")
        for i, pid in enumerate(uniq):
            x, y = coords_2d[i]
            f.write(f"{pid},{x:.6f},{y:.6f},{clusters[int(pid)]}\n")

    summary = {
        "n_ids": len(uniq),
        "topk": args.topk,
        "cluster_k": args.cluster_k,
        "embed_method": method,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.focus_id is not None and int(args.focus_id) in neighbors:
        print("[FOCUS]", args.focus_id, "neighbors:", neighbors[int(args.focus_id)])

    print("[OK] id mapping saved ->", out_dir)


if __name__ == "__main__":
    main()

