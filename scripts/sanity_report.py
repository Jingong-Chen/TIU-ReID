from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from unlearning_reid.models.scrubber import LinearScrubber, ResidualScrubber


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _curve_validity(curve_path: Path | None):
    if not curve_path or not curve_path.exists():
        return {"exists": False, "loss_nonzero": False, "metric_changed": False}
    rows = []
    if curve_path.suffix.lower() == ".csv":
        import csv

        with curve_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    else:
        rows = _read_json(curve_path) or []
    if not rows:
        return {"exists": True, "loss_nonzero": False, "metric_changed": False}
    losses = [float(r.get("loss", 0.0)) for r in rows]
    maps = [float(r.get("mAP", 0.0)) for r in rows]
    return {
        "exists": True,
        "loss_nonzero": any(l > 1e-8 for l in losses),
        "metric_changed": (max(maps) - min(maps)) > 1e-6 if maps else False,
    }


def _find_curve(dir_path: Path):
    if (dir_path / "curve.csv").exists():
        return dir_path / "curve.csv"
    if (dir_path / "curve.json").exists():
        return dir_path / "curve.json"
    return None


def _load_cfg_info(cfg_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    tr = repo_root / "third_party" / "TransReID"
    if str(tr) not in sys.path:
        sys.path.insert(0, str(tr))
    try:
        from config import cfg

        cfg.merge_from_file(str(cfg_path))
        return {
            "model_name": str(cfg.MODEL.NAME),
            "transformer_type": str(cfg.MODEL.TRANSFORMER_TYPE),
            "jpm": bool(cfg.MODEL.JPM),
            "neck": str(cfg.MODEL.NECK),
            "neck_feat": str(cfg.TEST.NECK_FEAT),
        }
    except Exception:
        return {
            "model_name": "unknown",
            "transformer_type": None,
            "jpm": None,
            "neck": None,
            "neck_feat": None,
        }


def _apply_scrubber_np(feats: np.ndarray, ckpt_path: Path):
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


def _norm_stats(feats: np.ndarray):
    norms = np.linalg.norm(feats, axis=1)
    return {
        "mean": float(norms.mean()),
        "p50": float(np.percentile(norms, 50)),
        "p90": float(np.percentile(norms, 90)),
    }


def _retain_cosine_stats(qf, qp, qc, gf, gp, gc):
    qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    sim = qn @ gn.T
    same_vals = []
    diff_vals = []
    for i in range(len(qp)):
        same_mask = (gp == qp[i]) & (gc != qc[i])
        diff_mask = gp != qp[i]
        if np.any(same_mask):
            same_vals.append(float(sim[i, same_mask].mean()))
        if np.any(diff_mask):
            diff_vals.append(float(sim[i, diff_mask].mean()))
    return {
        "same_mean": float(np.mean(same_vals)) if same_vals else None,
        "diff_mean": float(np.mean(diff_vals)) if diff_vals else None,
        "same_n": int(len(same_vals)),
        "diff_n": int(len(diff_vals)),
    }


def _centroid_geometry(feats: np.ndarray, pids: np.ndarray, forget_ids: set[int], retain_ids: set[int]):
    centroids = {}
    for pid in np.unique(pids):
        mask = pids == pid
        centroids[int(pid)] = feats[mask].mean(axis=0)
    f_list = [centroids[pid] for pid in forget_ids if pid in centroids]
    r_list = [centroids[pid] for pid in retain_ids if pid in centroids]
    if not f_list or not r_list:
        return None
    f = np.stack(f_list, axis=0)
    r = np.stack(r_list, axis=0)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)
    f_mean = f.mean(axis=0)
    f_mean = f_mean / (np.linalg.norm(f_mean) + 1e-12)
    r_mean = r.mean(axis=0)
    r_mean = r_mean / (np.linalg.norm(r_mean) + 1e-12)
    cos_dist = float(1.0 - np.dot(f_mean, r_mean))
    sims = f @ r.T
    dists = 1.0 - sims
    min_dists = dists.min(axis=1)
    return {
        "forget_vs_retain_centroid_cos_dist": cos_dist,
        "forget_to_nearest_retain_centroid_cos_dist": {
            "p50": float(np.percentile(min_dists, 50)),
            "p90": float(np.percentile(min_dists, 90)),
        },
    }

def _nn_hit_rate(qf, qp, qc, gf, gp, gc, k=1):
    qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    dist = 1.0 - np.matmul(qn, gn.T)
    hits = 0
    for i in range(len(qp)):
        q_pid = qp[i]
        q_cam = qc[i]
        mask = ~((gp == q_pid) & (gc == q_cam))
        if not np.any(mask):
            continue
        d = dist[i][mask]
        gpid = gp[mask]
        order = np.argsort(d)
        topk = gpid[order[:k]]
        if np.any(topk == q_pid):
            hits += 1
    return hits / len(qp) if len(qp) > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvp_dir", required=True)
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--teacher_cfg", default=None)
    ap.add_argument("--teacher_weights", default=None)
    ap.add_argument("--probe_dir", default=None)
    ap.add_argument("--split_dir", default=None)
    args = ap.parse_args()

    mvp_dir = Path(args.mvp_dir)
    teacher_dir = Path(args.teacher_dir)

    cfg_txt = teacher_dir / "teacher_cfg_path.txt"
    w_txt = teacher_dir / "teacher_weights_path.txt"
    cfg_from_txt = cfg_txt.read_text().strip() if cfg_txt.exists() else None
    w_from_txt = w_txt.read_text().strip() if w_txt.exists() else None

    cfg_used = args.teacher_cfg or cfg_from_txt
    w_used = args.teacher_weights or w_from_txt

    cfg_info = _load_cfg_info(Path(cfg_used)) if cfg_used else {}
    model_name = cfg_info.get("model_name")
    jpm = cfg_info.get("jpm")
    neck_feat = cfg_info.get("neck_feat")
    transformer_type = cfg_info.get("transformer_type")

    feats_path = mvp_dir / "features" / "train_teacher.npz"
    dim = None
    norm_mean = None
    train_input_normalized = None
    if feats_path.exists():
        d = np.load(feats_path, allow_pickle=True)
        feats = d["feats"]
        dim = int(feats.shape[1])
        norms = np.linalg.norm(feats, axis=1)
        norm_mean = float(norms.mean())
        train_input_normalized = bool(np.allclose(norms, 1.0, atol=1e-4))

    teacher_retain = _read_json(mvp_dir / "metrics_teacher_retain.json")
    teacher_forget = _read_json(mvp_dir / "metrics_teacher_forget.json")
    unlearn_retain = _read_json(mvp_dir / "metrics_unlearn_retain.json")
    unlearn_forget = _read_json(mvp_dir / "metrics_unlearn_forget.json")

    test_eval_dir = mvp_dir / "test_eval"
    test_teacher = _read_json(test_eval_dir / "metrics_test_teacher.json")
    test_unlearn = _read_json(test_eval_dir / "metrics_test_unlearn.json")

    probe_stats = None
    if args.probe_dir:
        probe_stats = _read_json(Path(args.probe_dir) / "probe_stats.json")

    unlearn_args = _read_json(mvp_dir / "unlearn" / "run_args.json")
    mem_attack = _read_json(mvp_dir / "membership_attack.json")
    summary = _read_json(mvp_dir / "summary.json")
    forget_balance = None
    if unlearn_args:
        forget_balance = unlearn_args.get("forget_balance")
    if forget_balance is None:
        forget_balance = "proportional"

    moment_stats = None
    train_log = _read_json(mvp_dir / "unlearn" / "train_log.json")
    if isinstance(train_log, list) and train_log:
        last = train_log[-1]
        moment_stats = {
            "loss_moment": last.get("loss_moment"),
            "mean_diff_norm": last.get("mean_diff_norm"),
            "cov_diff_norm": last.get("cov_diff_norm"),
            "lambda_moment_eff": last.get("lambda_moment_eff"),
            "lambda_mem_eff": last.get("lambda_mem_eff"),
        }

    forget_counts = None
    retain_ids = set()
    forget_ids = set()
    if args.split_dir and feats_path.exists():
        split_dir = Path(args.split_dir)
        forget_ids = set(int(x) for x in split_dir.joinpath("forget_ids.txt").read_text().split() if x.strip())
        retain_ids = set(int(x) for x in split_dir.joinpath("retain_ids.txt").read_text().split() if x.strip())
        if forget_ids:
            d = np.load(feats_path, allow_pickle=True)
            pids = d["pids"].astype(np.int64)
            counts = {pid: int(np.sum(pids == pid)) for pid in forget_ids}
            total_count = int(len(pids))
            forget_total = int(sum(counts.values()))
            retain_total = int(total_count - forget_total)
            forget_counts = {
                "counts": counts,
                "min": min(counts.values()),
                "max": max(counts.values()),
                "mean": float(np.mean(list(counts.values()))),
                "total_forget": forget_total,
                "total_retain": retain_total,
                "total": total_count,
                "forget_ratio": float(forget_total / max(total_count, 1)),
            }

    nn_stats = None
    retain_norm_stats = None
    retain_cosine_stats = None
    retain_delta_stats = None
    forget_geometry = None
    feat_dir = mvp_dir / "features"
    ckpt = mvp_dir / "unlearn" / "ckpt_final.pt"
    if feat_dir.exists() and ckpt.exists() and args.split_dir:
        q = np.load(feat_dir / "forget_query_teacher.npz", allow_pickle=True)
        g = np.load(feat_dir / "forget_gallery_teacher.npz", allow_pickle=True)
        qf, qp, qc = q["feats"].astype(np.float32), q["pids"].astype(np.int64), q["camids"].astype(np.int64)
        gf, gp, gc = g["feats"].astype(np.float32), g["pids"].astype(np.int64), g["camids"].astype(np.int64)
        split_dir = Path(args.split_dir)
        forget_ids = set(int(x) for x in split_dir.joinpath("forget_ids.txt").read_text().split())
        forget_mask = np.isin(gp, list(forget_ids))
        gf_f, gp_f, gc_f = gf[forget_mask], gp[forget_mask], gc[forget_mask]

        qf_u = _apply_scrubber_np(qf, ckpt)
        gf_u = _apply_scrubber_np(gf, ckpt)
        gf_u_f = _apply_scrubber_np(gf_f, ckpt)

        nn_stats = {
            "teacher": {
                "all_gallery": {
                    "hit@1": _nn_hit_rate(qf, qp, qc, gf, gp, gc, k=1),
                    "hit@5": _nn_hit_rate(qf, qp, qc, gf, gp, gc, k=5),
                },
                "forget_only_gallery": {
                    "hit@1": _nn_hit_rate(qf, qp, qc, gf_f, gp_f, gc_f, k=1),
                    "hit@5": _nn_hit_rate(qf, qp, qc, gf_f, gp_f, gc_f, k=5),
                },
            },
            "unlearn": {
                "all_gallery": {
                    "hit@1": _nn_hit_rate(qf_u, qp, qc, gf_u, gp, gc, k=1),
                    "hit@5": _nn_hit_rate(qf_u, qp, qc, gf_u, gp, gc, k=5),
                },
                "forget_only_gallery": {
                    "hit@1": _nn_hit_rate(qf_u, qp, qc, gf_u_f, gp_f, gc_f, k=1),
                    "hit@5": _nn_hit_rate(qf_u, qp, qc, gf_u_f, gp_f, gc_f, k=5),
                },
            },
        }

        rq_path = feat_dir / "retain_query_teacher.npz"
        rg_path = feat_dir / "retain_gallery_teacher.npz"
        if rq_path.exists() and rg_path.exists():
            rq = np.load(rq_path, allow_pickle=True)
            rg = np.load(rg_path, allow_pickle=True)
            rq_f, rq_p, rq_c = rq["feats"].astype(np.float32), rq["pids"].astype(np.int64), rq["camids"].astype(np.int64)
            rg_f, rg_p, rg_c = rg["feats"].astype(np.float32), rg["pids"].astype(np.int64), rg["camids"].astype(np.int64)
            rq_s = _apply_scrubber_np(rq_f, ckpt)
            rg_s = _apply_scrubber_np(rg_f, ckpt)
            retain_norm_stats = {"query": _norm_stats(rq_s), "gallery": _norm_stats(rg_s)}
            retain_cosine_stats = _retain_cosine_stats(rq_s, rq_p, rq_c, rg_s, rg_p, rg_c)

        if feats_path.exists() and forget_ids and retain_ids:
            d = np.load(feats_path, allow_pickle=True)
            train_feats = d["feats"].astype(np.float32)
            train_pids = d["pids"].astype(np.int64)
            train_s = _apply_scrubber_np(train_feats, ckpt)
            forget_geometry = _centroid_geometry(train_s, train_pids, forget_ids, retain_ids)
            # retain delta stats: ||scrub(z)-z|| on retain train samples
            retain_mask = np.isin(train_pids, list(retain_ids))
            if np.any(retain_mask):
                delta = train_s[retain_mask] - train_feats[retain_mask]
                dn = np.linalg.norm(delta, axis=1)
                retain_delta_stats = {
                    "mean": float(dn.mean()),
                    "p50": float(np.percentile(dn, 50)),
                    "p90": float(np.percentile(dn, 90)),
                    "n": int(dn.shape[0]),
                }

    a1_curve = _find_curve(mvp_dir / "attack_A1")
    a2_curve = _find_curve(mvp_dir / "attack_A2")

    if model_name == "transformer":
        if jpm:
            embed_source = "JPM concat: global CLS + 4 local CLS"
            model_class = "build_transformer_local"
        else:
            embed_source = "global CLS token"
            model_class = "build_transformer"
    else:
        embed_source = "global pooled feature"
        model_class = "Backbone"
    bnneck_used = bool(neck_feat == "after")

    # scorecard
    retain_map = unlearn_retain["mAP"] if unlearn_retain else None
    forget_map = unlearn_forget["mAP"] if unlearn_forget else None
    membership_auc_abs = None
    a2_step_to_recover_50 = None
    if summary:
        membership_auc_abs = summary.get("membership_auc_abs")
        a2 = summary.get("attack_A2") or {}
        a2_step_to_recover_50 = a2.get("step_to_recover_50pct")
    if membership_auc_abs is None and mem_attack and "metrics" in mem_attack:
        m = mem_attack["metrics"].get("l2") or mem_attack["metrics"].get("raw") or {}
        membership_auc_abs = m.get("auc_abs")

    retain_norm_mean = None
    retain_norm_p90 = None
    if retain_norm_stats and retain_norm_stats.get("query"):
        retain_norm_mean = retain_norm_stats["query"].get("mean")
        retain_norm_p90 = retain_norm_stats["query"].get("p90")

    scorecard = {
        "retain_mAP": retain_map,
        "forget_mAP": forget_map,
        "membership_auc_abs": membership_auc_abs,
        "a2_step_to_recover_50pct": a2_step_to_recover_50,
        "retain_norm_mean": retain_norm_mean,
        "retain_norm_p90": retain_norm_p90,
        "pass_retain": retain_map is not None and retain_map >= 0.98,
        "pass_forget": forget_map is not None and forget_map <= 0.10,
        "pass_mem_target": membership_auc_abs is not None and membership_auc_abs <= 0.55,
        "pass_mem_relaxed": membership_auc_abs is not None and membership_auc_abs <= 0.60,
    }
    scorecard["fail_retain_drift"] = (
        (retain_norm_mean is not None and retain_norm_mean < 15.0)
        or (retain_norm_p90 is not None and retain_norm_p90 < 15.0)
    )
    scorecard["pass_all_target"] = (
        scorecard["pass_retain"]
        and scorecard["pass_forget"]
        and scorecard["pass_mem_target"]
        and not scorecard["fail_retain_drift"]
    )
    scorecard["pass_all_relaxed"] = (
        scorecard["pass_retain"]
        and scorecard["pass_forget"]
        and scorecard["pass_mem_relaxed"]
        and not scorecard["fail_retain_drift"]
    )

    report = {
        "teacher_extractor": {
            "backend": "TransReID",
            "model_name": model_name,
            "model_class": model_class,
            "transformer_type": transformer_type,
            "jpm": jpm,
            "neck_feat": neck_feat,
            "embedding_source": embed_source,
            "bnneck_used": bnneck_used,
            "cfg_path_used": cfg_used,
            "weights_path_used": w_used,
            "cfg_path_from_txt": cfg_from_txt,
            "weights_path_from_txt": w_from_txt,
            "cfg_match_txt": bool(cfg_used and cfg_from_txt and Path(cfg_used).resolve() == Path(cfg_from_txt).resolve()),
            "weights_match_txt": bool(
                w_used and w_from_txt and Path(w_used).resolve() == Path(w_from_txt).resolve()
            ),
            "embedding_dim": dim,
            "l2_normalize_in_extractor": False,
            "l2_normalize_in_eval": True,
            "unlearn_train_input_normalized": train_input_normalized,
            "unlearn_train_feat_norm_mean": norm_mean,
        },
        "probe_validity": probe_stats,
        "metrics": {
            "teacher_retain": teacher_retain,
            "teacher_forget": teacher_forget,
            "unlearn_retain": unlearn_retain,
            "unlearn_forget": unlearn_forget,
            "test_teacher": test_teacher,
            "test_unlearn": test_unlearn,
        },
        "unlearn_config": {
            "forget_mode": unlearn_args.get("forget_mode") if unlearn_args else None,
            "forget_balance": forget_balance,
            "lambda_moment": unlearn_args.get("lambda_moment") if unlearn_args else None,
            "mem_warmup_frac": unlearn_args.get("mem_warmup_frac") if unlearn_args else None,
            "alpha_max": unlearn_args.get("alpha_max") if unlearn_args else None,
            "lambda_delta": unlearn_args.get("lambda_delta") if unlearn_args else None,
            "forget_target": unlearn_args.get("forget_target") if unlearn_args else None,
            "lambda_impostor": unlearn_args.get("lambda_impostor") if unlearn_args else None,
            "impostor_topk": unlearn_args.get("impostor_topk") if unlearn_args else None,
            "neighbor_attract": unlearn_args.get("neighbor_attract") if unlearn_args else None,
            "lambda_nbr": unlearn_args.get("lambda_nbr") if unlearn_args else None,
            "nbr_k": unlearn_args.get("nbr_k") if unlearn_args else None,
            "nbr_sample": unlearn_args.get("nbr_sample") if unlearn_args else None,
            "neighbor_cache_dir": unlearn_args.get("neighbor_cache_dir") if unlearn_args else None,
        },
        "membership_training": {
            "enabled": bool(unlearn_args.get("membership_adv")) if unlearn_args else False,
            "lambda_mem": unlearn_args.get("lambda_mem") if unlearn_args else None,
            "mem_hidden": unlearn_args.get("mem_hidden") if unlearn_args else None,
            "mem_steps_per_iter": unlearn_args.get("mem_steps_per_iter") if unlearn_args else None,
            "mem_adv_mode": unlearn_args.get("mem_adv_mode") if unlearn_args else None,
        },
        "moment_stats": moment_stats,
        "membership_attack": mem_attack,
        "attack_validity": {
            "A1": _curve_validity(a1_curve),
            "A2": _curve_validity(a2_curve),
        },
        "forget_id_counts": forget_counts,
        "retain_norm_stats": retain_norm_stats,
        "retain_delta_stats": retain_delta_stats,
        "retain_cosine_stats": retain_cosine_stats,
        "forget_vs_retain_geometry": forget_geometry,
        "forget_nn_hit_rate": nn_stats,
        "scorecard": scorecard,
    }

    out = mvp_dir / "sanity_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    lines = []
    tr = teacher_retain["mAP"] if teacher_retain else None
    uf = unlearn_forget["mAP"] if unlearn_forget else None
    tt = test_teacher["mAP"] if test_teacher else None
    tu = test_unlearn["mAP"] if test_unlearn else None
    lines.append(f"[SANITY] extractor=TransReID model={report['teacher_extractor']['model_name']}")
    lines.append(
        f"[SANITY] emb={report['teacher_extractor']['embedding_source']} bnneck={report['teacher_extractor']['bnneck_used']}"
    )
    lines.append(f"[SANITY] dim={dim} l2_eval=True train_norm={report['teacher_extractor']['unlearn_train_input_normalized']}")
    if probe_stats:
        rq = probe_stats["retain"]["validity"]["valid_query_ratio"]
        fq = probe_stats["forget"]["validity"]["valid_query_ratio"]
        lines.append(f"[SANITY] valid_query retain={rq:.3f} forget={fq:.3f}")
    if tr is not None and uf is not None:
        lines.append(f"[SANITY] retain_mAP={tr:.4f} forget_mAP={uf:.4f}")
    if tt is not None and tu is not None:
        lines.append(f"[SANITY] test_mAP teacher={tt:.4f} unlearn={tu:.4f}")
    a1v = report["attack_validity"]["A1"]
    a2v = report["attack_validity"]["A2"]
    lines.append(f"[SANITY] A1 loss_nonzero={a1v['loss_nonzero']} metric_changed={a1v['metric_changed']}")
    lines.append(f"[SANITY] A2 loss_nonzero={a2v['loss_nonzero']} metric_changed={a2v['metric_changed']}")
    if mem_attack and "metrics" in mem_attack:
        m = mem_attack["metrics"].get("l2") or mem_attack["metrics"].get("raw")
        if m:
            auc = m.get("auc")
            acc = m.get("acc")
            auc_abs = m.get("auc_abs", auc)
            acc_abs = m.get("acc_abs", acc)
            lines.append(
                f"[SANITY] mem_attack auc={auc:.4f} acc={acc:.4f} auc_abs={auc_abs:.4f} acc_abs={acc_abs:.4f}"
            )
    if retain_norm_mean is not None:
        lines.append(f"[SANITY] retain_norm_mean={retain_norm_mean:.2f} p90={retain_norm_p90:.2f}")
    if retain_delta_stats and retain_delta_stats.get("mean") is not None:
        lines.append(
            f"[SANITY] retain_delta_mean={retain_delta_stats['mean']:.4f} p90={retain_delta_stats['p90']:.4f}"
        )
    if forget_geometry and "forget_vs_retain_centroid_cos_dist" in forget_geometry:
        d = forget_geometry["forget_vs_retain_centroid_cos_dist"]
        p50 = forget_geometry["forget_to_nearest_retain_centroid_cos_dist"]["p50"]
        lines.append(f"[SANITY] centroid_dist={d:.3f} nearest_p50={p50:.3f}")
    if scorecard:
        lines.append(
            f"[SANITY] scorecard retain_ok={scorecard['pass_retain']} forget_ok={scorecard['pass_forget']} mem_ok={scorecard['pass_mem_target']}"
        )

    for line in lines[:10]:
        print(line, flush=True)

    print("[OK] sanity_report.json ->", out)


if __name__ == "__main__":
    main()

