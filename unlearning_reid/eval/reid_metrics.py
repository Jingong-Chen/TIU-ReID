from __future__ import annotations

import numpy as np


def compute_cmc_map(distmat: np.ndarray, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    Standard single-query evaluation with same PID+same CAM filtered out.
    distmat: [num_q, num_g], smaller is more similar.
    Returns: cmc (max_rank,), mAP (float)
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_cam = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
        keep = ~remove

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        precisions = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        AP = (precisions * orig_cmc).sum() / num_rel
        all_AP.append(AP)

        num_valid_q += 1.0

    if num_valid_q == 0:
        return np.zeros(max_rank), 0.0

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / num_valid_q
    mAP = float(np.mean(all_AP))
    return cmc, mAP


