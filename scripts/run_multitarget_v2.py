#!/usr/bin/env python
"""
Multitarget v2: A) Base per target, B) Method retfix_F stats, C) 3x3 tune, D) Difficulty analysis.
All outputs under output/compare/multitarget_v2/.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "output" / "compare" / "multitarget_v2"
DATASET = "market1501"
SEEDS = [0, 1, 2]
INITIAL_TIDS = [2, 10, 20, 52, 100, 150, 200, 300, 500, 700]
TUNE_TIDS = [2, 7, 10, 20, 52, 100]  # C) 6 tids for 3x3 sweep, 2x3 heatmap
MIN_TIDS = 10
HARD_DISTRACTORS = 200
ENFORCE_CROSS_CAM = True


def _env():
    env = os.environ.copy()
    env.setdefault("REID_DATA_DIR", str(REPO / "data"))
    env.setdefault("REID_OUTPUT_DIR", str(REPO / "output"))
    # Ensure PYTHONPATH includes repo root for subprocess imports
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{REPO}:{pythonpath}"
    else:
        env["PYTHONPATH"] = str(REPO)
    return env


def _get_teacher_paths():
    teacher_dir = Path(os.environ.get("REID_OUTPUT_DIR", REPO / "output")) / "transreid" / "market_teacher_r50"
    cfg_path = teacher_dir / "teacher_cfg_path.txt"
    weights_path = teacher_dir / "teacher_weights_path.txt"
    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Teacher not found: {teacher_dir}. Run teacher training first.")
    return cfg_path.read_text().strip(), weights_path.read_text().strip()


def _get_train_ids():
    from unlearning_reid.datasets.common import group_by_pid, read_train_items
    data_root = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
    items = read_train_items(DATASET, data_root)
    by_pid = group_by_pid(items)
    pids = sorted(pid for pid, lst in by_pid.items() if len(lst) >= 4)
    return pids


def _get_train_feat_npz():
    # Prefer existing teacher train features (e.g. from mvp run)
    cand = [
        REPO / "output" / "mvp" / "auto_market1501_id2_seed0_t3" / "features" / "train_teacher.npz",
        REPO / "output" / "mvp" / "auto_market1501_id2_seed0" / "features" / "train_teacher.npz",
        REPO / "output" / "mvp" / "mvp_market_single2_seed0" / "features" / "train_teacher.npz",
    ]
    for p in cand:
        if p.exists():
            return str(p)
    # Generate: write train_list.jsonl then extract
    out_dir = OUT_ROOT / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = out_dir / "train_list.jsonl"
    if not train_jsonl.exists():
        from unlearning_reid.datasets.common import read_train_items
        data_root = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
        items = read_train_items(DATASET, data_root)
        with train_jsonl.open("w") as f:
            for it in items:
                f.write(json.dumps({"path": it.path, "pid": it.pid, "camid": it.camid}) + "\n")
    cfg, weights = _get_teacher_paths()
    npz_path = out_dir / "train_teacher.npz"
    if not npz_path.exists():
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "extract_teacher_features.py"),
             "--cfg", cfg, "--weights", weights,
             "--jsonl", str(train_jsonl), "--out", str(npz_path)],
            cwd=str(REPO), env=_env(), check=True,
        )
    return str(npz_path)


def run_A():
    """A) Base per target: splits, probes, probe_stats, eval_baseline -> base_per_target.csv"""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_csv_path = OUT_ROOT / "base_per_target.csv"
    
    # Skip A if base_per_target.csv already exists and has data
    if base_csv_path.exists():
        with base_csv_path.open() as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            if len(existing_rows) >= MIN_TIDS * len(SEEDS):
                print(f"[A] Skipping: base_per_target.csv already exists with {len(existing_rows)} rows")
                # Convert to list of dicts (csv.DictReader rows are already dicts, but ensure they're reusable)
                return [dict(r) for r in existing_rows], []
    
    cfg, weights = _get_teacher_paths()
    feat_npz = _get_train_feat_npz()
    train_ids = _get_train_ids()
    skip_log_path = OUT_ROOT / "skip_log.csv"

    skip_rows = []
    base_rows = []
    # Full candidate list: INITIAL_TIDS then train_ids not in INITIAL_TIDS (order preserved)
    full_candidates = list(INITIAL_TIDS) + [p for p in train_ids if p not in set(INITIAL_TIDS)]
    # Per-seed: collect valid tids (need at least MIN_TIDS)
    for seed in SEEDS:
        split_base = REPO / "output" / "splits" / DATASET
        probe_base = REPO / "output" / "probes" / DATASET
        base_out_base = REPO / "output" / "removable_baselines"
        valid_tids_this_seed = []

        for tid in full_candidates:
            if len(valid_tids_this_seed) >= MIN_TIDS:
                break

            split_dir = split_base / f"seed{seed}_tid{tid}"
            probe_dir = probe_base / f"seed{seed}_hard200_tid{tid}"
            base_out_dir = base_out_base / f"base_teacher_{DATASET}_seed{seed}_tid{tid}"

            # 1) make_splits
            r1 = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "make_splits.py"),
                 "--dataset", DATASET, "--seed", str(seed), "--forget_ids", str(tid), "--out_dir", str(split_dir)],
                cwd=str(REPO), env=_env(), capture_output=True, text=True,
            )
            if r1.returncode != 0:
                skip_rows.append({"seed": seed, "tid": tid, "reason": f"make_splits_fail:{r1.stderr[:80] if r1.stderr else r1.returncode}"})
                continue

            # 2) make_probe_sets (hard200, enforce_cross_cam)
            r2 = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "make_probe_sets.py"),
                 "--dataset", DATASET, "--split_dir", str(split_dir), "--out_dir", str(probe_dir),
                 "--seed", str(seed), "--distractor_mode", "hard", "--forget_distractor_ids", str(HARD_DISTRACTORS),
                 "--enforce_cross_cam", "1", "--feat_train_npz", feat_npz],
                cwd=str(REPO), env=_env(), capture_output=True, text=True,
            )
            if r2.returncode != 0:
                skip_rows.append({"seed": seed, "tid": tid, "reason": f"make_probe_fail:{r2.stderr[:80] if r2.stderr else r2.returncode}"})
                continue

            # 3) probe_stats
            subprocess.run(
                [sys.executable, str(REPO / "scripts" / "probe_stats.py"), "--probe_dir", str(probe_dir)],
                cwd=str(REPO), env=_env(), capture_output=True,
            )
            stats_path = probe_dir / "probe_stats.json"
            if not stats_path.exists():
                skip_rows.append({"seed": seed, "tid": tid, "reason": "no_probe_stats"})
                continue
            stats = json.loads(stats_path.read_text())
            forget_validity = stats.get("forget", {}).get("validity", {})
            valid_ratio = forget_validity.get("valid_query_ratio", 0.0)
            n_forget_q = forget_validity.get("n_query", 0)
            if valid_ratio < 1.0 or n_forget_q == 0:
                skip_rows.append({"seed": seed, "tid": tid, "reason": f"valid_query_ratio={valid_ratio}"})
                continue

            # 4) eval_baseline (WITHOUT)
            base_out_dir.mkdir(parents=True, exist_ok=True)
            r4 = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "eval_baseline_probes.py"),
                 "--mode", "without", "--probe_dir", str(probe_dir), "--out_dir", str(base_out_dir),
                 "--cfg", cfg, "--weights", weights, "--skip_if_exists"],
                cwd=str(REPO), env=_env(), capture_output=True, text=True,
            )
            if r4.returncode != 0:
                skip_rows.append({"seed": seed, "tid": tid, "reason": "eval_baseline_fail"})
                continue
            summary_path = base_out_dir / "summary_base.json"
            if not summary_path.exists():
                skip_rows.append({"seed": seed, "tid": tid, "reason": "eval_failed"})
                continue
            summary = json.loads(summary_path.read_text())
            base_rows.append({
                "seed": seed, "tid": tid,
                "base_ret_mAP": summary.get("retain_mAP"), "base_ret_R1": summary.get("retain_CMC1"),
                "base_fgt_mAP": summary.get("forget_mAP"), "base_fgt_R1": summary.get("forget_CMC1"),
            })
            valid_tids_this_seed.append(tid)
            if len(valid_tids_this_seed) >= MIN_TIDS:
                break

    # Write skip_log.csv
    with skip_log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "tid", "reason"])
        w.writeheader()
        w.writerows(skip_rows)

    # Write base_per_target.csv
    with base_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "tid", "base_ret_mAP", "base_ret_R1", "base_fgt_mAP", "base_fgt_R1"])
        w.writeheader()
        w.writerows(base_rows)

    print(f"[A] base_per_target.csv: {len(base_rows)} rows, skip_log: {len(skip_rows)} rows")
    # Return rows as list of dicts (not csv.DictReader rows which are not reusable)
    result_rows = []
    for r in base_rows:
        result_rows.append({
            "seed": r["seed"], "tid": r["tid"],
            "base_ret_mAP": r["base_ret_mAP"], "base_ret_R1": r["base_ret_R1"],
            "base_fgt_mAP": r["base_fgt_mAP"], "base_fgt_R1": r["base_fgt_R1"],
        })
    return result_rows, skip_rows


def _parse_float(s):
    if s is None or s == "" or str(s).strip() == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def run_B(base_rows):
    """B) Method retfix_F per target, summary tables, multitarget_table.md/.tex"""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    method_csv = OUT_ROOT / "method_retfixF_per_target.csv"
    if not base_rows:
        for p in ["method_retfixF_per_target.csv", "multitarget_summary_by_target.csv", "multitarget_summary_overall.csv"]:
            (OUT_ROOT / p).write_text("")
        (OUT_ROOT / "multitarget_table.md").write_text("| Target ID | Ret mAP | Fgt mAP | DropR | Test mAP |\n")
        (OUT_ROOT / "multitarget_table.tex").write_text("\\begin{tabular}{c|cccc}\n\\end{tabular}\n")
        print("[B] No base rows, wrote empty tables")
        return
    # Skip training if method_retfixF_per_target.csv already exists with enough rows
    method_rows = []
    if method_csv.exists():
        with method_csv.open() as f:
            for row in csv.DictReader(f):
                try:
                    tid_val = int(row.get("tid", 0))
                except (TypeError, ValueError):
                    tid_val = row.get("tid")
                try:
                    seed_val = int(row.get("seed", 0))
                except (TypeError, ValueError):
                    seed_val = row.get("seed")
                method_rows.append({
                    "seed": seed_val, "tid": tid_val,
                    "ret_mAP": _parse_float(row.get("ret_mAP")), "ret_R1": _parse_float(row.get("ret_R1")),
                    "fgt_mAP": _parse_float(row.get("fgt_mAP")), "fgt_R1": _parse_float(row.get("fgt_R1")),
                    "test_mAP": _parse_float(row.get("test_mAP")), "test_R1": _parse_float(row.get("test_R1")),
                    "base_fgt_mAP": _parse_float(row.get("base_fgt_mAP")), "DropR": _parse_float(row.get("DropR")),
                })
        if len(method_rows) >= len(base_rows):
            print(f"[B] Skipping: method_retfixF_per_target.csv already exists with {len(method_rows)} rows")
            # Fall through to generate summary tables from existing method_rows
        else:
            method_rows = []
    if not method_rows:
        cfg, weights = _get_teacher_paths()
        feat_npz = _get_train_feat_npz()
        print(f"[B] Starting training for {len(base_rows)} (seed, tid) combinations...")
        pbar = tqdm(base_rows, desc="B: Training models", unit="model", disable=not HAS_TQDM)
        for r in pbar:
            seed, tid = r["seed"], r["tid"]
            if HAS_TQDM:
                pbar.set_description(f"B: Training tid={tid} seed={seed}")
            probe_dir = REPO / "output" / "probes" / DATASET / f"seed{seed}_hard200_tid{tid}"
            out_dir = REPO / "output" / "removable" / f"removable_market1501_tid{tid}_seed{seed}_retfixF"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Train retfix_F (default hyperparams: lambda_baseonly_forget ~2, lambda_adv ~0.1 from run_single_retfix)
            cmd = [
                sys.executable, str(REPO / "scripts" / "train_removable_unlearn_v3.py"),
                "--cfg", cfg, "--weights", weights, "--out_dir", str(out_dir), "--dataset", DATASET,
                "--forget_id", str(tid), "--seed", str(seed), "--probe_dir", str(probe_dir),
                "--teacher_feat_npz", feat_npz,
                "--lambda_baseonly_forget", "2.0", "--lambda_adv", "0.1", "--epochs", "10",
                "--eval_per_epoch", "0",
            ]
            tr = subprocess.run(cmd, cwd=str(REPO), env=_env(), capture_output=True, text=True)
            # Eval WITH/WITHOUT from run dir
            sansy = out_dir / "sanity_report_removable.json"
            if tr.returncode != 0 and not sansy.exists():
                method_rows.append({
                    "seed": seed, "tid": tid,
                    "ret_mAP": None, "ret_R1": None, "fgt_mAP": None, "fgt_R1": None,
                    "test_mAP": None, "test_R1": None,
                    "base_fgt_mAP": r.get("base_fgt_mAP"), "DropR": None,
                })
                continue
            if sansy.exists():
                j = json.loads(sansy.read_text())
                ret_mAP = j.get("without_retain_mAP")
                fgt_mAP = j.get("without_forget_mAP")
            else:
                ret_mAP = fgt_mAP = None
            test_mAP = test_R1 = None
            # eval_test_split if desired (writes test_without_mAP/CMC1 into sanity_report)
            try:
                subprocess.run(
                    [sys.executable, str(REPO / "scripts" / "eval_test_split.py"),
                     "--mvp_dir", str(out_dir), "--cfg", cfg, "--forget_id", str(tid), "--target_base_scale", "0.0"],
                    cwd=str(REPO), env=_env(), capture_output=True, timeout=120,
                )
                if sansy.exists():
                    j2 = json.loads(sansy.read_text())
                    test_mAP = j2.get("test_without_mAP"); test_R1 = j2.get("test_without_CMC1")
            except Exception:
                pass
            base_fgt = r.get("base_fgt_mAP") or 1.0
            drop_r = (float(base_fgt) - float(fgt_mAP)) / float(base_fgt) if (base_fgt and fgt_mAP is not None) else None
            method_rows.append({
                "seed": seed, "tid": tid,
                "ret_mAP": ret_mAP, "ret_R1": None, "fgt_mAP": fgt_mAP, "fgt_R1": None,
                "test_mAP": test_mAP, "test_R1": test_R1,
                "base_fgt_mAP": base_fgt, "DropR": drop_r,
            })
            if HAS_TQDM:
                completed = len([x for x in method_rows if x.get("ret_mAP") is not None])
                pbar.set_postfix({"completed": completed, "failed": len(method_rows) - completed})

        if HAS_TQDM:
            pbar.close()
        with method_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "tid", "ret_mAP", "ret_R1", "fgt_mAP", "fgt_R1", "test_mAP", "test_R1", "base_fgt_mAP", "DropR"])
            w.writeheader()
            w.writerows(method_rows)

    # Summary by target (mean±std across seeds)
    import numpy as np
    by_tid = {}
    for row in method_rows:
        tid = row["tid"]
        if tid not in by_tid:
            by_tid[tid] = []
        by_tid[tid].append(row)

    def mean_std(vals):
        v = [float(x) for x in vals if x is not None]
        if not v:
            return None, None
        return np.mean(v), np.std(v)

    summary_by_target = []
    for tid in sorted(by_tid.keys()):
        rows = by_tid[tid]
        ret_m, ret_s = mean_std([r["ret_mAP"] for r in rows])
        fgt_m, fgt_s = mean_std([r["fgt_mAP"] for r in rows])
        drop_m, drop_s = mean_std([r["DropR"] for r in rows])
        test_m, test_s = mean_std([r["test_mAP"] for r in rows])
        summary_by_target.append({
            "tid": tid,
            "ret_mAP_mean": ret_m, "ret_mAP_std": ret_s,
            "fgt_mAP_mean": fgt_m, "fgt_mAP_std": fgt_s,
            "DropR_mean": drop_m, "DropR_std": drop_s,
            "test_mAP_mean": test_m, "test_mAP_std": test_s,
        })

    with (OUT_ROOT / "multitarget_summary_by_target.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tid", "ret_mAP_mean", "ret_mAP_std", "fgt_mAP_mean", "fgt_mAP_std", "DropR_mean", "DropR_std", "test_mAP_mean", "test_mAP_std"])
        w.writeheader()
        w.writerows(summary_by_target)

    all_ret = [r["ret_mAP"] for r in method_rows if r["ret_mAP"] is not None]
    all_fgt = [r["fgt_mAP"] for r in method_rows if r["fgt_mAP"] is not None]
    all_drop = [r["DropR"] for r in method_rows if r["DropR"] is not None]
    overall = {
        "ret_mAP_mean": np.mean(all_ret) if all_ret else None, "ret_mAP_std": np.std(all_ret) if all_ret else None,
        "fgt_mAP_mean": np.mean(all_fgt) if all_fgt else None, "fgt_mAP_std": np.std(all_fgt) if all_fgt else None,
        "DropR_mean": np.mean(all_drop) if all_drop else None, "DropR_std": np.std(all_drop) if all_drop else None,
    }
    with (OUT_ROOT / "multitarget_summary_overall.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ret_mAP_mean", "ret_mAP_std", "fgt_mAP_mean", "fgt_mAP_std", "DropR_mean", "DropR_std"])
        w.writeheader()
        w.writerow(overall)

    # Tables md + tex
    md_lines = ["| Target ID | Ret mAP (mean±std) | Fgt mAP (mean±std) | DropR (mean±std) | Test mAP (mean±std) |", "|" + "---|" * 5]
    for row in summary_by_target:
        md_lines.append("| {} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} |".format(
            row["tid"],
            row["ret_mAP_mean"] or 0, row["ret_mAP_std"] or 0,
            row["fgt_mAP_mean"] or 0, row["fgt_mAP_std"] or 0,
            row["DropR_mean"] or 0, row["DropR_std"] or 0,
            row["test_mAP_mean"] or 0, row["test_mAP_std"] or 0,
        ))
    (OUT_ROOT / "multitarget_table.md").write_text("\n".join(md_lines) + "\n")
    tex_lines = ["\\begin{tabular}{c|cccc}\nTarget ID & Ret mAP & Fgt mAP & DropR & Test mAP \\\\\n\\hline"]
    for row in summary_by_target:
        tex_lines.append("{} & ${:.3f}\\pm{:.3f}$ & ${:.3f}\\pm{:.3f}$ & ${:.3f}\\pm{:.3f}$ & ${:.3f}\\pm{:.3f}$ \\\\".format(
            row["tid"], row["ret_mAP_mean"] or 0, row["ret_mAP_std"] or 0,
            row["fgt_mAP_mean"] or 0, row["fgt_mAP_std"] or 0,
            row["DropR_mean"] or 0, row["DropR_std"] or 0,
            row["test_mAP_mean"] or 0, row["test_mAP_std"] or 0,
        ))
    tex_lines.append("\\end{tabular}")
    (OUT_ROOT / "multitarget_table.tex").write_text("\n".join(tex_lines) + "\n")
    print(f"[B] method_retfixF_per_target.csv, summary tables, md/tex done")


def run_C():
    """C) 3x3 tune for tids 2,10,20,52,100 seed=0 only -> tune_grid_results, tune_best_per_tid, fig_tune_heatmap.pdf"""
    lb_vals = [0.1, 0.5, 1.0]
    la_vals = [0.05, 0.1, 0.2]
    cfg, weights = _get_teacher_paths()
    feat_npz = _get_train_feat_npz()
    # Load base_fgt_mAP per (seed, tid) for correct DropR denominator
    base_fgt_by_key = {}
    base_csv = OUT_ROOT / "base_per_target.csv"
    if base_csv.exists():
        with base_csv.open() as f:
            for row in csv.DictReader(f):
                k = (int(row["seed"]), int(row["tid"]))
                try:
                    base_fgt_by_key[k] = float(row.get("base_fgt_mAP") or 1.0)
                except (TypeError, ValueError):
                    base_fgt_by_key[k] = 1.0
    # Build list of (tid, lb, la) tasks for progress bar
    tasks = []
    for tid in TUNE_TIDS:
        seed = 0
        probe_dir = REPO / "output" / "probes" / DATASET / f"seed{seed}_hard200_tid{tid}"
        if not probe_dir.exists():
            continue
        for lb in lb_vals:
            for la in la_vals:
                tasks.append((tid, seed, lb, la, probe_dir, base_fgt_by_key.get((seed, tid), 1.0) or 1.0))
    if not tasks:
        print("[C] No probe_dir for TUNE_TIDS, skip")
        return
    print(f"[C] Starting 3x3 tune for {len(TUNE_TIDS)} tids, {len(tasks)} grid points (skip if already done)...")
    grid_rows = []
    n_skip, n_run = 0, 0
    pbar = tqdm(tasks, desc="C: Tune grid", unit="point", disable=not HAS_TQDM)
    for tid, seed, lb, la, probe_dir, base_fgt in pbar:
        if HAS_TQDM:
            pbar.set_description(f"C: tid={tid} lb={lb} la={la}")
        out_dir = (REPO / "output" / "removable" / f"sweep_tune_tid{tid}" / f"lb{lb}_la{la}").resolve()
        sanity_path = out_dir / "sanity_report_removable.json"
        already_done = sanity_path.exists()
        if already_done:
            n_skip += 1
            j = json.loads(sanity_path.read_text(encoding="utf-8"))
            ret = j.get("without_retain_mAP")
            fgt = j.get("without_forget_mAP")
            drop_r = (float(base_fgt) - float(fgt)) / float(base_fgt) if (base_fgt and fgt is not None) else None
            test_mAP = j.get("test_without_mAP")
            if test_mAP is None:
                try:
                    subprocess.run(
                        [sys.executable, str(REPO / "scripts" / "eval_test_split.py"),
                         "--mvp_dir", str(out_dir), "--cfg", cfg, "--forget_id", str(tid), "--target_base_scale", "0.0"],
                        cwd=str(REPO), env=_env(), capture_output=True, timeout=120,
                    )
                    if sanity_path.exists():
                        test_mAP = json.loads(sanity_path.read_text()).get("test_without_mAP")
                except Exception:
                    pass
            grid_rows.append({"tid": tid, "lb": lb, "la": la, "ret_mAP": ret, "fgt_mAP": fgt, "DropR": drop_r, "test_mAP": test_mAP})
            if HAS_TQDM:
                pbar.set_postfix({"done": len(grid_rows), "skip": n_skip})
            continue
        n_run += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "train_removable_unlearn_v3.py"),
             "--cfg", cfg, "--weights", weights, "--out_dir", str(out_dir), "--dataset", DATASET,
             "--forget_id", str(tid), "--seed", str(seed), "--probe_dir", str(probe_dir),
             "--teacher_feat_npz", feat_npz,
             "--lambda_baseonly_forget", str(lb), "--lambda_adv", str(la), "--epochs", "10",
             "--eval_per_epoch", "0"],
            cwd=str(REPO), env=_env(), check=True, capture_output=True,
        )
        j = json.loads((out_dir / "sanity_report_removable.json").read_text())
        ret = j.get("without_retain_mAP")
        fgt = j.get("without_forget_mAP")
        drop_r = (float(base_fgt) - float(fgt)) / float(base_fgt) if (base_fgt and fgt is not None) else None
        test_mAP = None
        try:
            subprocess.run(
                [sys.executable, str(REPO / "scripts" / "eval_test_split.py"),
                 "--mvp_dir", str(out_dir), "--cfg", cfg, "--forget_id", str(tid), "--target_base_scale", "0.0"],
                cwd=str(REPO), env=_env(), capture_output=True, timeout=120,
            )
            te = out_dir / "sanity_report_removable.json"
            if te.exists():
                test_mAP = json.loads(te.read_text()).get("test_without_mAP")
        except Exception:
            pass
        grid_rows.append({"tid": tid, "lb": lb, "la": la, "ret_mAP": ret, "fgt_mAP": fgt, "DropR": drop_r, "test_mAP": test_mAP})
        if HAS_TQDM:
            pbar.set_postfix({"done": len(grid_rows), "DropR": f"{drop_r:.3f}" if drop_r is not None else "-"})
    if HAS_TQDM:
        pbar.close()
    print(f"[C] Done: {n_skip} skipped (cached), {n_run} newly run")

    with (OUT_ROOT / "tune_grid_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tid", "lb", "la", "ret_mAP", "fgt_mAP", "DropR", "test_mAP"])
        w.writeheader()
        w.writerows(grid_rows)

    # Best per tid: Ret mAP >= 0.93, max DropR
    best_rows = []
    for tid in TUNE_TIDS:
        rows = [r for r in grid_rows if r["tid"] == tid and (r.get("ret_mAP") or 0) >= 0.93]
        if not rows:
            rows = [r for r in grid_rows if r["tid"] == tid]
        if not rows:
            continue
        best = max(rows, key=lambda x: float(x["DropR"] or 0))
        best_rows.append({"tid": tid, "lb": best["lb"], "la": best["la"], "ret_mAP": best["ret_mAP"], "DropR": best["DropR"]})
    with (OUT_ROOT / "tune_best_per_tid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tid", "lb", "la", "ret_mAP", "DropR"])
        w.writeheader()
        w.writerows(best_rows)

    # fig_tune_heatmap.pdf: 2x3 panel (5 tids), 3x3 heatmap DropR, corner text Ret mAP
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scripts.plotting.style import apply_tiu_style, save_tiu
        apply_tiu_style()
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes = axes.flatten()
        for idx, tid in enumerate(TUNE_TIDS):
            if idx >= len(axes):
                break
            ax = axes[idx]
            Z = np.full((3, 3), np.nan)
            R = np.full((3, 3), np.nan)
            for r in grid_rows:
                if r["tid"] != tid:
                    continue
                i = lb_vals.index(r["lb"])
                j = la_vals.index(r["la"])
                Z[i, j] = r["DropR"] or np.nan
                R[i, j] = r["ret_mAP"] or np.nan
            im = ax.imshow(Z, aspect="auto", cmap="viridis", vmin=0, vmax=0.9)
            for i in range(3):
                for j in range(3):
                    v = Z[i, j]
                    r = R[i, j]
                    t = f"{v:.2f}" if not np.isnan(v) else "-"
                    if not np.isnan(r):
                        t += f"\n({r:.2f})"
                    ax.text(j, i, t, ha="center", va="center", fontsize=8)
            ax.set_xticks(range(3)); ax.set_xticklabels([str(x) for x in la_vals])
            ax.set_yticks(range(3)); ax.set_yticklabels([str(x) for x in lb_vals])
            ax.set_xlabel(r"$\lambda_{\mathrm{adv}}$"); ax.set_ylabel(r"$\lambda_{\mathrm{baseonly}}$")
            ax.set_title(f"Target ID {tid}")
        for idx in range(len(TUNE_TIDS), len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        save_tiu(fig, str(OUT_ROOT / "fig_tune_heatmap.pdf"))
        plt.close()
    except Exception as e:
        print(f"[C] fig_tune_heatmap.pdf skip: {e}")
    print("[C] tune_grid_results, tune_best_per_tid, fig_tune_heatmap.pdf done")


def run_D(summary_by_target_path):
    """D) Difficulty metrics + fig_difficulty_scatter.pdf + README_SUMMARY.txt"""
    import numpy as np
    # Use tids from B (summary_by_target) if available; else INITIAL_TIDS
    tids_for_difficulty = list(INITIAL_TIDS)
    if summary_by_target_path and summary_by_target_path.exists():
        with summary_by_target_path.open() as f:
            tids_for_difficulty = sorted([int(row["tid"]) for row in csv.DictReader(f)])
        if not tids_for_difficulty:
            tids_for_difficulty = list(INITIAL_TIDS)
    print(f"[D] Computing difficulty metrics for {len(tids_for_difficulty)} tids...")
    feat_npz = _get_train_feat_npz()
    d = np.load(feat_npz, allow_pickle=True)
    feats = d["feats"].astype(np.float32)
    pids = d["pids"].astype(np.int64)
    uniq = sorted(set(pids.tolist()))
    pid_to_idx = {int(p): i for i, p in enumerate(uniq)}
    centroids = np.zeros((len(uniq), feats.shape[1]), dtype=np.float32)
    counts = np.zeros((len(uniq),), dtype=np.int64)
    for f, p in zip(feats, pids):
        i = pid_to_idx[int(p)]
        centroids[i] += f
        counts[i] += 1
    counts = np.maximum(counts, 1)
    centroids = centroids / counts[:, None]
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    centroids = centroids / norms
    pid_list = uniq

    diff_rows = []
    pbar_d = tqdm(tids_for_difficulty, desc="D: Difficulty metrics", unit="tid", disable=not HAS_TQDM)
    for tid in pbar_d:
        if tid not in pid_to_idx:
            diff_rows.append({"tid": tid, "n_target": 0, "hardness": np.nan, "distractor_strength": np.nan})
            continue
        tc = centroids[pid_to_idx[tid]]
        n_target = int(counts[pid_to_idx[tid]])
        retain_idx = [pid_to_idx[p] for p in pid_list if p != tid]
        if not retain_idx:
            diff_rows.append({"tid": tid, "n_target": n_target, "hardness": np.nan, "distractor_strength": np.nan})
            continue
        retain_c = centroids[retain_idx]
        sims = np.matmul(tc.reshape(1, -1), retain_c.T).flatten()
        top5 = np.sort(sims)[-5:]
        hardness = float(np.mean(top5))
        # distractor strength: from probe we don't have here; use mean sim to all retain as proxy
        distractor_strength = float(np.mean(sims))
        diff_rows.append({"tid": tid, "n_target": n_target, "hardness": hardness, "distractor_strength": distractor_strength})
        if HAS_TQDM:
            pbar_d.set_postfix({"hardness": f"{hardness:.3f}", "n": n_target})

    if HAS_TQDM:
        pbar_d.close()
    with (OUT_ROOT / "difficulty_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tid", "n_target", "hardness", "distractor_strength"])
        w.writeheader()
        w.writerows(diff_rows)

    # Merge DropR from summary_by_target
    drop_by_tid = {}
    if summary_by_target_path and summary_by_target_path.exists():
        with summary_by_target_path.open() as f:
            for row in csv.DictReader(f):
                drop_by_tid[int(row["tid"])] = float(row.get("DropR_mean") or 0)
    for row in diff_rows:
        row["DropR_mean"] = drop_by_tid.get(row["tid"])

    # For Pearson and README (defined even if plot fails)
    x = [r["hardness"] for r in diff_rows if r.get("DropR_mean") is not None]
    y = [r["DropR_mean"] for r in diff_rows if r.get("DropR_mean") is not None]
    if HAS_TQDM:
        print("[D] Writing fig_difficulty_scatter.pdf and README_SUMMARY.txt...")

    # fig_difficulty_scatter.pdf
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scripts.plotting.style import apply_tiu_style, save_tiu
        apply_tiu_style()
        fig, ax = plt.subplots(figsize=(6, 4))
        s = [np.log1p(r["n_target"]) * 30 for r in diff_rows if r.get("DropR_mean") is not None]
        c = [r["distractor_strength"] for r in diff_rows if r.get("DropR_mean") is not None]
        tids = [r["tid"] for r in diff_rows if r.get("DropR_mean") is not None]
        sc = ax.scatter(x, y, s=s, c=c, cmap="viridis", alpha=0.8)
        for i, tid in enumerate(tids):
            ax.annotate(str(tid), (x[i], y[i]), fontsize=9, xytext=(5, 5), textcoords="offset points")
        if 2 in tids:
            idx2 = tids.index(2)
            ax.scatter([x[idx2]], [y[idx2]], s=s[idx2]*1.5, facecolors="none", edgecolors="black", linewidths=2)
        ax.set_xlabel("Hardness (top-5 mean sim)"); ax.set_ylabel("DropR (mean)")
        plt.colorbar(sc, ax=ax, label="Distractor strength")
        plt.tight_layout()
        save_tiu(fig, str(OUT_ROOT / "fig_difficulty_scatter.pdf"))
        plt.close()
    except Exception as e:
        print(f"[D] fig_difficulty_scatter skip: {e}")

    # Pearson r
    if len(x) >= 3:
        r_pearson = np.corrcoef(x, y)[0, 1]
    else:
        r_pearson = np.nan
    readme_lines = [
        "multitarget_v2 summary (auto-generated)",
        "1. Overall mean±std (Ret mAP / Fgt mAP / DropR): see multitarget_summary_overall.csv",
        "2. Tune 后是否改善: see tune_best_per_tid.csv and fig_tune_heatmap.pdf",
        "3. DropR 与 hardness 的相关性 (Pearson r): {:.4f}".format(r_pearson),
        "4. Difficulty metrics: difficulty_metrics.csv, fig_difficulty_scatter.pdf",
        "5. 多目标有效 tid 数: see base_per_target.csv row count and skip_log.csv",
        "6. 最终表格: multitarget_table.md, multitarget_table.tex",
    ]
    (OUT_ROOT / "README_SUMMARY.txt").write_text("\n".join(readme_lines) + "\n")
    print("[D] difficulty_metrics.csv, fig_difficulty_scatter.pdf, README_SUMMARY.txt done")


def main():
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_rows, _ = run_A()
    run_B(base_rows)
    run_C()
    run_D(OUT_ROOT / "multitarget_summary_by_target.csv")
    print("\n[OK] multitarget_v2 done. Outputs under", OUT_ROOT)


if __name__ == "__main__":
    main()
