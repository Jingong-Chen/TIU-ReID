from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from unlearning_reid.eval.reid_metrics import compute_cmc_map
from unlearning_reid.models.scrubber import LinearScrubber, ResidualScrubber


def load_npz(p: Path):
    d = np.load(p, allow_pickle=True)
    return d["feats"].astype(np.float32), d["pids"].astype(np.int64), d["camids"].astype(np.int64)


def apply_scrubber(feats: np.ndarray, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dim = int(ckpt["dim"])
    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    bottleneck = cfg.get("bottleneck", None)
    scrubber_type = cfg.get("scrubber_type", "linear")
    alpha_max = cfg.get("alpha_max", 0.1)
    state = ckpt.get("scrubber", {}) or {}
    if scrubber_type == "residual" or ("alpha_raw" in state):
        scrub = ResidualScrubber(dim=dim, bottleneck=bottleneck, alpha_max=alpha_max)
    else:
        scrub = LinearScrubber(dim=dim, bottleneck=bottleneck)
    scrub.load_state_dict(ckpt["scrubber"], strict=False)
    scrub.eval()
    with torch.no_grad():
        z = torch.from_numpy(feats)
        out = scrub(z).numpy()
    return out


def _compute_dist(qf: np.ndarray, gf: np.ndarray) -> np.ndarray:
    if torch.cuda.is_available():
        with torch.no_grad():
            q = torch.from_numpy(qf).cuda(non_blocking=True)
            g = torch.from_numpy(gf).cuda(non_blocking=True)
            q = q / (q.norm(dim=1, keepdim=True) + 1e-12)
            g = g / (g.norm(dim=1, keepdim=True) + 1e-12)
            dist = 1.0 - torch.mm(q, g.t())
            return dist.cpu().numpy()
    qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.matmul(qn, gn.T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q_npz", required=True)
    ap.add_argument("--g_npz", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--scrubber_ckpt", default=None, help="if set, apply scrubber to features before eval")
    ap.add_argument("--max_rank", type=int, default=50)
    args = ap.parse_args()

    qf, qp, qc = load_npz(Path(args.q_npz))
    gf, gp, gc = load_npz(Path(args.g_npz))

    if args.scrubber_ckpt:
        qf = apply_scrubber(qf, Path(args.scrubber_ckpt))
        gf = apply_scrubber(gf, Path(args.scrubber_ckpt))

    dist = _compute_dist(qf, gf)

    cmc, mAP = compute_cmc_map(dist, qp, gp, qc, gc, max_rank=args.max_rank)
    res = {
        "mAP": mAP,
        "CMC@1": float(cmc[0]),
        "CMC@5": float(cmc[4]) if len(cmc) > 4 else None,
        "CMC@10": float(cmc[9]) if len(cmc) > 9 else None,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(res, indent=2) + "\n")
    print("[OK]", res, "->", args.out_json)


if __name__ == "__main__":
    main()


