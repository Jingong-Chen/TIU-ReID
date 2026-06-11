#!/usr/bin/env python
"""
Removable Module Unlearning - V3 (with warmup scheduling + consistency)

Key improvements over V2:
- Warmup + ramp scheduling for baseonly_forget and anti-target losses
- Non-target consistency constraint
- Better retain preservation
"""
from __future__ import annotations

# Compatibility: timm (via TransReID) may require torch._six on older PyTorch
import sys
import collections.abc
import types
if "torch._six" not in sys.modules:
    _m = types.ModuleType("torch._six")
    _m.container_abcs = collections.abc
    _m.string_classes = (str, bytes)
    _m.int_classes = (int,)
    sys.modules["torch._six"] = _m

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: simple progress indicator
    def tqdm(iterable, **kwargs):
        return iterable

from unlearning_reid.datasets.common import read_train_items
from unlearning_reid.models.removable import TargetDiscriminator, TargetModule, grad_reverse


def _ensure_transreid_imports(repo_root: Path) -> None:
    tr = repo_root / "third_party" / "TransReID"
    if str(tr) not in sys.path:
        sys.path.insert(0, str(tr))


def _cuda_works() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.randn(4, 4, device="cuda")
        return True
    except:
        return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrainImageDataset(Dataset):
    def __init__(self, items, pid_to_label, transform, teacher_feats=None):
        self.items = items
        self.pid_to_label = pid_to_label
        self.transform = transform
        self.teacher_feats = teacher_feats

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, pid, camid = self.items[idx]
        img = self._read_image(path)
        if self.transform is not None:
            img = self.transform(img)
        label = self.pid_to_label[pid]
        cam_label = max(0, int(camid) - 1)
        view_label = 0
        if self.teacher_feats is None:
            return img, int(pid), int(label), cam_label, view_label
        teacher_feat = torch.from_numpy(self.teacher_feats[idx]).float()
        return img, int(pid), int(label), cam_label, view_label, teacher_feat

    @staticmethod
    def _read_image(path: str):
        from datasets.bases import read_image
        return read_image(path)


class TargetPKSampler(Sampler[int]):
    def __init__(self, pid_list, target_pid, batch_size, num_instances, seed=0):
        self.index_by_pid = {}
        for idx, pid in enumerate(pid_list):
            self.index_by_pid.setdefault(pid, []).append(idx)
        if target_pid not in self.index_by_pid:
            raise RuntimeError(f"Target pid {target_pid} not found.")
        self.target_pid = int(target_pid)
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.p = max(2, self.batch_size // self.num_instances)
        self.seed = int(seed)
        self.epoch = 0
        self.other_pids = [pid for pid in self.index_by_pid.keys() if pid != self.target_pid]
        self.num_batches = max(1, math.ceil(len(self.other_pids) / max(1, self.p - 1)))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        other = list(self.other_pids)
        rng.shuffle(other)
        step = 0
        for _ in range(self.num_batches):
            batch_pids = [self.target_pid]
            chunk = other[step : step + (self.p - 1)]
            step += self.p - 1
            if len(chunk) < (self.p - 1):
                chunk += rng.sample(self.other_pids, (self.p - 1) - len(chunk))
            batch_pids += chunk
            for pid in batch_pids:
                indices = self.index_by_pid[pid]
                if len(indices) >= self.num_instances:
                    chosen = rng.sample(indices, self.num_instances)
                else:
                    chosen = rng.choices(indices, k=self.num_instances)
                for idx in chosen:
                    yield idx

    def __len__(self):
        return self.num_batches * self.p * self.num_instances


def build_train_transform(cfg, device="cpu"):
    from datasets.make_dataloader import Compose, Normalize, Pad, RandomCrop, RandomHorizontalFlip, Resize, ToTensor
    from timm.data.random_erasing import RandomErasing
    # RandomErasing must stay on CPU: DataLoader workers are forked, CUDA cannot be re-initialized there
    return Compose([
        ToTensor(),
        Resize(cfg.INPUT.SIZE_TRAIN),
        RandomHorizontalFlip(p=cfg.INPUT.PROB),
        Pad(cfg.INPUT.PADDING),
        RandomCrop(cfg.INPUT.SIZE_TRAIN),
        Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
    ])


def build_eval_transform(cfg):
    from datasets.make_dataloader import Compose, Normalize, Resize, ToTensor
    return Compose([
        ToTensor(),
        Resize(cfg.INPUT.SIZE_TEST),
        Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


def load_transreid_weights(model, weights_path):
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_state = model.state_dict()
    cleaned = {}
    for k, v in state.items():
        key = k.replace("module.", "")
        if key in model_state and model_state[key].shape == v.shape:
            cleaned[key] = v
    model.load_state_dict(cleaned, strict=False)


def write_env_snapshot(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (out_dir / "pip_freeze.txt").write_text(freeze)
    except:
        pass
    (out_dir / "python.txt").write_text(f"{sys.version}\n{sys.executable}\n")


# ========== Evaluation utilities ==========
class JsonlImageDataset(Dataset):
    def __init__(self, jsonl_path, transform, data_root=None):
        self.items = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.items.append(json.loads(line))
        self.transform = transform
        self.data_root = Path(data_root) if data_root is not None else None

    def __len__(self):
        return len(self.items)

    def _resolve_path(self, path: str) -> str:
        if os.path.exists(path):
            return path
        # Probe jsonl may have paths from another machine; rebase under data_root or REID_DATA_DIR
        if "data" in path:
            suffix = path.split("data/", 1)[-1]
            if self.data_root is not None:
                candidate = str(self.data_root / suffix)
                if os.path.exists(candidate):
                    return candidate
            data_dir = os.environ.get("REID_DATA_DIR")
            if data_dir:
                candidate = os.path.join(data_dir, suffix)
                if os.path.exists(candidate):
                    return candidate
        return path

    def __getitem__(self, idx):
        d = self.items[idx]
        path = self._resolve_path(d["path"])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, int(d["pid"]), max(0, int(d["camid"]) - 1), 0, path


def _compute_dist(qf, gf):
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.matmul(qf, gf.T)


def compute_cmc_map(dist, qp, gp, qc, gc, max_rank=50):
    num_q = dist.shape[0]
    indices = np.argsort(dist, axis=1)
    matches = (gp[indices] == qp[:, np.newaxis]).astype(np.int32)
    all_cmc, all_AP = [], []
    for q_idx in range(num_q):
        order = indices[q_idx]
        remove = (gp[order] == qp[q_idx]) & (gc[order] == qc[q_idx])
        keep = ~remove
        m = matches[q_idx][keep]
        if not np.any(m):
            continue
        cmc = m.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_rel = m.sum()
        tmp_cmc = m.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * m
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.mean(axis=0) if len(all_cmc) else np.zeros(max_rank)
    mAP = np.mean(all_AP) if all_AP else 0.0
    return cmc, mAP


@torch.no_grad()
def quick_eval_probe(model, target_module, mode, probe_dir, transform, device, target_pid, target_base_scale, batch_size=64, num_workers=2, data_root=None):
    model.eval()
    if target_module is not None:
        target_module.eval()

    results = {}
    for probe_type in ["retain", "forget"]:
        q_path = probe_dir / f"{probe_type}_query.jsonl"
        g_path = probe_dir / f"{probe_type}_gallery.jsonl"
        if not q_path.exists() or not g_path.exists():
            continue

        feats_all, pids_all, camids_all = [], [], []
        for jsonl_path in [q_path, g_path]:
            ds = JsonlImageDataset(jsonl_path, transform, data_root=data_root)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
            feats_gpu, pids_list, camids_list = [], [], []
            for x, pid, cam_label, view_label, _ in loader:
                x = x.to(device)
                cam_label = cam_label.to(device)
                view_label = view_label.to(device)
                pid_t = torch.as_tensor(pid, device=device)

                z = model(x, cam_label=cam_label, view_label=view_label)
                if isinstance(z, tuple):
                    z = z[1]
                if mode == "with" and target_module is not None:
                    delta = target_module(z)
                    mask = (pid_t == int(target_pid)).float().unsqueeze(1)
                    if target_base_scale != 0.0:
                        z = z + delta * mask - target_base_scale * z * mask
                    else:
                        z = z + delta * mask

                feats_gpu.append(z.float())
                pids_list.append(pid.numpy() if hasattr(pid, 'numpy') else np.array(pid))
                camids_list.append(cam_label.cpu().numpy())

            feats_all.append(torch.cat(feats_gpu, dim=0).cpu().float().numpy())
            pids_all.append(np.concatenate(pids_list, axis=0))
            camids_all.append(np.concatenate(camids_list, axis=0))

        qf, qp, qc = feats_all[0], pids_all[0], camids_all[0]
        gf, gp, gc = feats_all[1], pids_all[1], camids_all[1]
        dist = _compute_dist(qf, gf)
        cmc, mAP = compute_cmc_map(dist, qp, gp, qc, gc)
        results[probe_type] = {"mAP": float(mAP), "CMC@1": float(cmc[0]) if len(cmc) > 0 else 0.0}

    return results


def get_schedule_factor(epoch, total_epochs, warmup_frac):
    """Get scheduling factor: 0 during warmup, linear ramp after."""
    warmup_epochs = int(total_epochs * warmup_frac)
    if epoch <= warmup_epochs:
        return 0.0
    ramp_epochs = total_epochs - warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, ramp_epochs)
    return min(1.0, progress)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", default="market1501")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--teacher_feat_npz", default=None)
    ap.add_argument("--forget_id", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_instances", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    # Learning rates
    ap.add_argument("--lr_base", type=float, default=1e-4)
    ap.add_argument("--lr_module", type=float, default=3e-4)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--lr_disc", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    # Task losses
    ap.add_argument("--lambda_id", type=float, default=1.0)
    ap.add_argument("--lambda_tri", type=float, default=1.0)
    ap.add_argument("--triplet_margin", type=float, default=0.3)
    # Forgetting losses (with warmup)
    ap.add_argument("--lambda_adv", type=float, default=0.5)
    ap.add_argument("--lambda_baseonly_forget", type=float, default=2.0)
    ap.add_argument("--baseonly_forget_mode", default="entropy", choices=["entropy", "low_target_prob"])
    ap.add_argument("--lambda_feat_confuse", type=float, default=1.0)
    ap.add_argument("--lambda_feat_spread", type=float, default=1.0)
    # NEW: Warmup scheduling
    ap.add_argument("--baseonly_warmup_frac", type=float, default=0.5, help="Fraction of epochs for warmup (forgetting losses off)")
    ap.add_argument("--anti_warmup_frac", type=float, default=0.5, help="Fraction of epochs for anti-target warmup")
    # NEW: Non-target consistency
    ap.add_argument("--lambda_nontarget_consist", type=float, default=1.0, help="Non-target z_with==z_without consistency")
    # Delta regularization
    ap.add_argument("--lambda_delta", type=float, default=0.01)
    ap.add_argument("--lambda_delta_non", type=float, default=0.1)
    # Other
    ap.add_argument("--grl_lambda", type=float, default=1.0)
    ap.add_argument("--disc_hidden", type=int, default=256)
    ap.add_argument("--module_hidden", type=int, default=512)
    ap.add_argument("--module_dropout", type=float, default=0.1)
    ap.add_argument("--detach_base_for_target", type=int, default=1)
    ap.add_argument("--detach_base_in_delta", type=int, default=1)
    ap.add_argument("--target_base_scale", type=float, default=1.0)
    ap.add_argument("--neck_feat", default="before")
    ap.add_argument("--freeze_base", type=int, default=0)
    ap.add_argument("--probe_dir", default=None)
    ap.add_argument("--split_dir", default=None)
    ap.add_argument("--eval_per_epoch", type=int, default=0, choices=[0, 1],
                    help="0: only evaluate at end (faster); 1: evaluate every epoch (slower, for monitoring)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_transreid_imports(repo_root)

    from config import cfg
    from model import make_model
    from loss.triplet_loss import TripletLoss

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root) if args.data_root else Path(os.environ.get("REID_DATA_DIR", repo_root / "data"))
    forget_id = args.forget_id or int(os.environ.get("FORGET_ID", "2"))

    set_seed(args.seed)

    cfg.merge_from_file(args.cfg)
    cfg.defrost()
    cfg.TEST.NECK_FEAT = args.neck_feat
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.freeze()

    device = "cuda" if _cuda_works() else "cpu"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    items = read_train_items(args.dataset, data_root)
    pids = sorted({it.pid for it in items})
    pid_to_label = {pid: i for i, pid in enumerate(pids)}
    label_to_pid = {i: pid for pid, i in pid_to_label.items()}

    if forget_id not in pid_to_label:
        raise RuntimeError(f"forget_id={forget_id} not found.")
    target_label = pid_to_label[forget_id]

    # Teacher features
    teacher_feats = None
    if args.teacher_feat_npz:
        feat_path = Path(args.teacher_feat_npz)
        if feat_path.exists():
            npz = np.load(feat_path, allow_pickle=True)
            feats = npz["feats"].astype(np.float32)
            paths = npz["paths"]
            feat_map = {str(p): feats[i] for i, p in enumerate(paths)}
            teacher_feats = []
            for it in items:
                key = str(it.path)
                teacher_feats.append(feat_map.get(key, np.zeros((feats.shape[1],), dtype=np.float32)))
            teacher_feats = np.stack(teacher_feats)

    dataset = TrainImageDataset(
        [(it.path, it.pid, it.camid) for it in items],
        pid_to_label,
        build_train_transform(cfg, device),
        teacher_feats=teacher_feats,
    )

    pid_list = [it.pid for it in items]
    sampler = TargetPKSampler(pid_list, forget_id, args.batch, args.num_instances, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True)

    num_classes = len(pid_to_label)
    model = make_model(cfg, num_class=num_classes, camera_num=0, view_num=0)
    load_transreid_weights(model, args.weights)
    model.to(device)

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1], device=device)
        cam_label = torch.zeros(2, dtype=torch.long, device=device)
        view_label = torch.zeros(2, dtype=torch.long, device=device)
        out = model(dummy, cam_label=cam_label, view_label=view_label)
        feat_dim = int(out[1].shape[1] if isinstance(out, tuple) else out.shape[1])

    target_module = TargetModule(dim=feat_dim, hidden=args.module_hidden, dropout=args.module_dropout).to(device)
    discriminator = TargetDiscriminator(dim=feat_dim, hidden=args.disc_hidden).to(device)
    classifier = nn.Linear(feat_dim, num_classes, bias=False).to(device)
    nn.init.normal_(classifier.weight, std=0.001)

    if args.freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    params = []
    if not args.freeze_base:
        params.append({"params": model.parameters(), "lr": args.lr_base})
    params.append({"params": target_module.parameters(), "lr": args.lr_module})
    params.append({"params": classifier.parameters(), "lr": args.lr_head})
    params.append({"params": discriminator.parameters(), "lr": args.lr_disc})
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    triplet_loss_fn = TripletLoss(margin=args.triplet_margin)
    scaler = amp.GradScaler(enabled=(device == "cuda"))

    train_log = []
    gate_stats_all = []
    loss_breakdown_all = []
    epoch_eval_short = []

    eval_transform = build_eval_transform(cfg)
    probe_dir = Path(args.probe_dir) if args.probe_dir else None

    print(f"[INFO] V3 Training with Warmup Scheduling")
    print(f"[INFO] baseonly_warmup_frac={args.baseonly_warmup_frac}, anti_warmup_frac={args.anti_warmup_frac}")
    print(f"[INFO] lambda_baseonly_forget={args.lambda_baseonly_forget}, lambda_adv={args.lambda_adv}")
    print(f"[INFO] lambda_nontarget_consist={args.lambda_nontarget_consist}")
    print(f"[INFO] Target: pid={forget_id}, label={target_label}")

    start = time.time()
    # Overall progress bar
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc=f"Training (tid={forget_id}, seed={args.seed})", unit="epoch", disable=not HAS_TQDM)
    for epoch in epoch_pbar:
        # Compute schedule factors
        baseonly_factor = get_schedule_factor(epoch, args.epochs, args.baseonly_warmup_frac)
        anti_factor = get_schedule_factor(epoch, args.epochs, args.anti_warmup_frac)
        
        effective_lambda_baseonly = args.lambda_baseonly_forget * baseonly_factor
        effective_lambda_adv = args.lambda_adv * anti_factor
        effective_lambda_feat_confuse = args.lambda_feat_confuse * baseonly_factor
        effective_lambda_feat_spread = args.lambda_feat_spread * baseonly_factor

        model.eval()  # Keep eval for feature extraction
        target_module.train()
        discriminator.train()
        classifier.train()

        epoch_losses = {
            "loss_task_id": 0.0,
            "loss_task_metric": 0.0,
            "loss_base_anti": 0.0,
            "loss_baseonly_forget": 0.0,
            "loss_feat_confuse": 0.0,
            "loss_feat_spread": 0.0,
            "loss_nontarget_consist": 0.0,
            "loss_delta_reg": 0.0,
            "loss_total": 0.0,
        }
        epoch_counts = {"n": 0, "n_target": 0, "n_non_target": 0}
        epoch_gate = {"gate_1_count": [], "gate_1_pids": [], "batches_with_target": 0}
        delta_ratio_tensors = []
        base_norm_tensors = []

        # Batch progress bar
        batch_pbar = tqdm(loader, desc=f"  Epoch {epoch}/{args.epochs}", leave=False, disable=not HAS_TQDM or args.eval_per_epoch)
        for step, batch in enumerate(batch_pbar, start=1):
            if args.max_steps and step > args.max_steps:
                break

            if len(batch) == 6:
                img, pid, label, cam_label, view_label, teacher_feat = batch
            else:
                img, pid, label, cam_label, view_label = batch
                teacher_feat = None

            img = img.to(device)
            pid = pid.to(device)
            label = label.to(device)
            cam_label = cam_label.to(device)
            view_label = view_label.to(device)

            target_mask = (pid == int(forget_id)).float().unsqueeze(1)
            is_target = (pid == int(forget_id)).long()

            n_target = int(target_mask.sum().item())
            n_non_target = img.size(0) - n_target
            if n_target > 0:
                epoch_gate["gate_1_count"].append(n_target)
                epoch_gate["gate_1_pids"].extend(pid[is_target == 1].cpu().tolist())
                epoch_gate["batches_with_target"] += 1

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=(device == "cuda")):
                z_base = model(img, cam_label=cam_label, view_label=view_label)
                if isinstance(z_base, tuple):
                    z_base = z_base[1]
                z_base_det = z_base.detach() if args.detach_base_in_delta else z_base
                delta_raw = target_module(z_base_det)
                delta = delta_raw * target_mask

                # z_on: WITH_MODULE
                z_on = z_base + delta
                if args.target_base_scale != 0.0:
                    z_on = z_base + target_mask * (delta - args.target_base_scale * z_base_det)
                if args.detach_base_for_target:
                    z_on = z_on + target_mask * (z_base.detach() - z_base)

                # z_off: WITHOUT_MODULE (should equal z_base for all samples)
                z_off = z_base

                # ========== Task Loss (on z_on) ==========
                logits_on = classifier(z_on)
                ce = F.cross_entropy(logits_on, label, reduction="none")
                loss_task_id = ce.mean()
                loss_task_metric, _, _ = triplet_loss_fn(z_on, label)

                # ========== Base Anti-Target Adversarial (with warmup) ==========
                adv_logits = discriminator(grad_reverse(z_base, args.grl_lambda))
                loss_base_anti = F.cross_entropy(adv_logits, is_target)

                # ========== Base-only Forget Loss (with warmup) ==========
                loss_baseonly_forget = torch.tensor(0.0, device=device)
                if n_target > 0 and effective_lambda_baseonly > 0:
                    logits_off = classifier(z_off[is_target == 1])
                    probs_off = F.softmax(logits_off, dim=1)
                    if args.baseonly_forget_mode == "entropy":
                        entropy = (-probs_off * (probs_off + 1e-12).log()).sum(dim=1)
                        max_entropy = math.log(num_classes)
                        loss_baseonly_forget = (max_entropy - entropy).mean()
                    else:
                        p_target = probs_off[:, target_label]
                        loss_baseonly_forget = p_target.mean()

                # ========== Feature-level forgetting (with warmup) ==========
                loss_feat_confuse = torch.tensor(0.0, device=device)
                loss_feat_spread = torch.tensor(0.0, device=device)

                if n_target > 0 and n_non_target > 0 and effective_lambda_feat_confuse > 0:
                    z_t = z_base[is_target == 1]
                    z_nt = z_base[is_target == 0]
                    centroid_nt = z_nt.mean(dim=0, keepdim=True)
                    z_t_norm = F.normalize(z_t, p=2, dim=1)
                    centroid_nt_norm = F.normalize(centroid_nt, p=2, dim=1)
                    cos_sim = (z_t_norm * centroid_nt_norm).sum(dim=1)
                    loss_feat_confuse = 1.0 - cos_sim.mean()

                if n_target > 1 and effective_lambda_feat_spread > 0:
                    z_t = z_base[is_target == 1]
                    z_t_norm = F.normalize(z_t, p=2, dim=1)
                    sim_matrix = torch.matmul(z_t_norm, z_t_norm.t())
                    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
                    loss_feat_spread = sim_matrix[mask].mean()

                # ========== NEW: Non-target consistency ==========
                loss_nontarget_consist = torch.tensor(0.0, device=device)
                if n_non_target > 0 and args.lambda_nontarget_consist > 0:
                    # For non-target, z_on should equal z_off (since gate=0)
                    # This helps prevent drift
                    z_on_nt = z_on[is_target == 0]
                    z_off_nt = z_off[is_target == 0]
                    loss_nontarget_consist = ((z_on_nt - z_off_nt) ** 2).sum(dim=1).mean()

                # ========== Delta Regularization ==========
                loss_delta_reg = torch.tensor(0.0, device=device)
                loss_delta_non_reg = torch.tensor(0.0, device=device)
                if n_target > 0:
                    loss_delta_reg = (delta_raw[is_target == 1].pow(2).sum(dim=1)).mean()
                    with torch.no_grad():
                        dn = torch.norm(delta_raw[is_target == 1], dim=1)
                        bn = torch.norm(z_base[is_target == 1], dim=1)
                        delta_ratio_tensors.append((dn / (bn + 1e-12)))
                        base_norm_tensors.append(bn)
                if n_non_target > 0:
                    loss_delta_non_reg = (delta_raw[is_target == 0].pow(2).sum(dim=1)).mean()

                # ========== Total Loss ==========
                loss = (
                    args.lambda_id * loss_task_id
                    + args.lambda_tri * loss_task_metric
                    + effective_lambda_adv * loss_base_anti
                    + effective_lambda_baseonly * loss_baseonly_forget
                    + effective_lambda_feat_confuse * loss_feat_confuse
                    + effective_lambda_feat_spread * loss_feat_spread
                    + args.lambda_nontarget_consist * loss_nontarget_consist
                    + args.lambda_delta * loss_delta_reg
                    + args.lambda_delta_non * loss_delta_non_reg
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = img.size(0)
            epoch_losses["loss_task_id"] += float(loss_task_id.detach()) * bs
            epoch_losses["loss_task_metric"] += float(loss_task_metric.detach()) * bs
            epoch_losses["loss_base_anti"] += float(loss_base_anti.detach()) * bs
            epoch_losses["loss_baseonly_forget"] += float(loss_baseonly_forget.detach()) * n_target if n_target > 0 else 0
            epoch_losses["loss_feat_confuse"] += float(loss_feat_confuse.detach()) * n_target if n_target > 0 else 0
            epoch_losses["loss_feat_spread"] += float(loss_feat_spread.detach()) * n_target if n_target > 0 else 0
            epoch_losses["loss_nontarget_consist"] += float(loss_nontarget_consist.detach()) * n_non_target if n_non_target > 0 else 0
            epoch_losses["loss_delta_reg"] += float(loss_delta_reg.detach()) * n_target if n_target > 0 else 0
            epoch_losses["loss_total"] += float(loss.detach()) * bs

            epoch_counts["n"] += bs
            epoch_counts["n_target"] += n_target
            epoch_counts["n_non_target"] += n_non_target
        
        # Close batch progress bar
        if HAS_TQDM and not args.eval_per_epoch:
            batch_pbar.close()

        # Single GPU->CPU sync for delta/base stats (was per-batch)
        if delta_ratio_tensors:
            delta_ratio_samples = torch.cat(delta_ratio_tensors).cpu().tolist()
        else:
            delta_ratio_samples = []
        if base_norm_tensors:
            base_norm_samples = torch.cat(base_norm_tensors).cpu().tolist()
        else:
            base_norm_samples = []

        # Normalize
        n = epoch_counts["n"]
        n_t = epoch_counts["n_target"]
        n_nt = epoch_counts["n_non_target"]
        for k in ["loss_task_id", "loss_task_metric", "loss_base_anti", "loss_total"]:
            epoch_losses[k] /= max(n, 1)
        for k in ["loss_baseonly_forget", "loss_feat_confuse", "loss_feat_spread", "loss_delta_reg"]:
            epoch_losses[k] /= max(n_t, 1)
        epoch_losses["loss_nontarget_consist"] /= max(n_nt, 1)

        gate_summary = {
            "epoch": epoch,
            "batches_with_target": epoch_gate["batches_with_target"],
            "all_gate1_pids_are_target": all(p == forget_id for p in epoch_gate["gate_1_pids"]) if epoch_gate["gate_1_pids"] else True,
        }
        gate_stats_all.append(gate_summary)

        delta_ratio_stats = {
            "mean": float(np.mean(delta_ratio_samples)) if delta_ratio_samples else 0,
            "p50": float(np.percentile(delta_ratio_samples, 50)) if delta_ratio_samples else 0,
            "p90": float(np.percentile(delta_ratio_samples, 90)) if delta_ratio_samples else 0,
        }
        base_norm_stats = {
            "mean": float(np.mean(base_norm_samples)) if base_norm_samples else 0,
            "p90": float(np.percentile(base_norm_samples, 90)) if base_norm_samples else 0,
        }

        loss_breakdown_all.append({
            "epoch": epoch,
            "losses": epoch_losses,
            "effective_lambdas": {
                "baseonly": effective_lambda_baseonly,
                "adv": effective_lambda_adv,
                "feat_confuse": effective_lambda_feat_confuse,
                "feat_spread": effective_lambda_feat_spread,
            },
            "delta_ratio": delta_ratio_stats,
            "base_norm": base_norm_stats,
        })

        # Epoch evaluation (optional: skip when --eval_per_epoch 0 to save time)
        epoch_eval = {"epoch": epoch, "with": {}, "without": {}}
        if args.eval_per_epoch and probe_dir and probe_dir.exists():
            model.eval()
            target_module.eval()
            with torch.no_grad():
                epoch_eval["with"] = quick_eval_probe(model, target_module, "with", probe_dir, eval_transform, device, forget_id, args.target_base_scale, data_root=data_root)
                epoch_eval["without"] = quick_eval_probe(model, None, "without", probe_dir, eval_transform, device, forget_id, 0.0, data_root=data_root)
        epoch_eval_short.append(epoch_eval)

        # Update epoch progress bar
        if HAS_TQDM:
            epoch_pbar.set_postfix({
                "loss": f"{epoch_losses['loss_total']:.4f}",
                "baseonly": f"{epoch_losses['loss_baseonly_forget']:.4f}",
                "delta_r": f"{delta_ratio_stats['mean']:.3f}"
            })
            if epoch_eval["with"]:
                epoch_pbar.write(f"  Epoch {epoch}: WITH retain={epoch_eval['with'].get('retain', {}).get('mAP', 0):.4f} "
                                 f"forget={epoch_eval['with'].get('forget', {}).get('mAP', 0):.4f} | "
                                 f"WITHOUT retain={epoch_eval['without'].get('retain', {}).get('mAP', 0):.4f} "
                                 f"forget={epoch_eval['without'].get('forget', {}).get('mAP', 0):.4f}")
        
        if not HAS_TQDM or args.eval_per_epoch:
            print(f"[EPOCH {epoch}] sched_factor={baseonly_factor:.2f} total={epoch_losses['loss_total']:.4f} "
                  f"task_id={epoch_losses['loss_task_id']:.4f} base_anti={epoch_losses['loss_base_anti']:.4f} "
                  f"baseonly={epoch_losses['loss_baseonly_forget']:.4f} consist={epoch_losses['loss_nontarget_consist']:.4f} "
                  f"delta_ratio={delta_ratio_stats['mean']:.3f} base_norm={base_norm_stats['mean']:.1f}")
            if epoch_eval["with"]:
                print(f"         WITH: retain={epoch_eval['with'].get('retain', {}).get('mAP', 0):.4f} "
                      f"forget={epoch_eval['with'].get('forget', {}).get('mAP', 0):.4f}")
                print(f"         WITHOUT: retain={epoch_eval['without'].get('retain', {}).get('mAP', 0):.4f} "
                      f"forget={epoch_eval['without'].get('forget', {}).get('mAP', 0):.4f}")

        train_log.append({
            "epoch": epoch,
            "losses": epoch_losses,
            "schedule_factor": baseonly_factor,
            "gate": gate_summary,
            "delta_ratio": delta_ratio_stats,
            "base_norm": base_norm_stats,
            "eval": epoch_eval,
        })

    elapsed = time.time() - start
    print(f"[OK] Training done in {elapsed:.1f}s")

    # Save checkpoints
    torch.save(model.state_dict(), out_dir / "base_ckpt.pth")
    torch.save({
        "dim": feat_dim,
        "config": {"hidden": args.module_hidden, "dropout": args.module_dropout},
        "state_dict": target_module.state_dict(),
    }, out_dir / "target_module.pth")
    torch.save(discriminator.state_dict(), out_dir / "discriminator.pth")
    torch.save(classifier.state_dict(), out_dir / "classifier_head.pth")

    # Save logs
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2) + "\n")
    (out_dir / "gate_stats.json").write_text(json.dumps(gate_stats_all, indent=2) + "\n")
    (out_dir / "loss_breakdown.json").write_text(json.dumps(loss_breakdown_all, indent=2) + "\n")
    (out_dir / "epoch_eval_short.json").write_text(json.dumps(epoch_eval_short, indent=2) + "\n")

    run_args = vars(args)
    run_args.update({
        "target_pid": int(forget_id),
        "target_label": int(target_label),
        "num_classes": num_classes,
        "feat_dim": feat_dim,
    })
    (out_dir / "run_args.json").write_text(json.dumps(run_args, indent=2) + "\n")
    (out_dir / "teacher_cfg_path.txt").write_text(str(Path(args.cfg).resolve()) + "\n")
    (out_dir / "teacher_weights_path.txt").write_text(str(Path(args.weights).resolve()) + "\n")
    write_env_snapshot(out_dir)

    # Final evaluation and diagnosis
    print("[INFO] Generating final diagnostic report...")
    model.eval()
    target_module.eval()
    discriminator.eval()
    classifier.eval()

    final_eval = {"with": {}, "without": {}}
    if probe_dir and probe_dir.exists():
        final_eval["with"] = quick_eval_probe(model, target_module, "with", probe_dir, eval_transform, device, forget_id, args.target_base_scale, data_root=data_root)
        final_eval["without"] = quick_eval_probe(model, None, "without", probe_dir, eval_transform, device, forget_id, 0.0, data_root=data_root)

    # Discriminator AUC on training data
    diag_loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
    disc_preds, disc_labels = [], []
    base_target_probs, base_target_entropies = [], []
    with torch.no_grad():
        for batch in diag_loader:
            if len(batch) == 6:
                img, pid, label, cam_label, view_label, _ = batch
            else:
                img, pid, label, cam_label, view_label = batch
            img = img.to(device)
            cam_label = cam_label.to(device)
            view_label = view_label.to(device)
            pid_t = torch.as_tensor(pid, device=device)

            z_base = model(img, cam_label=cam_label, view_label=view_label)
            if isinstance(z_base, tuple):
                z_base = z_base[1]

            d_logits = discriminator(z_base)
            # ROC-AUC requires continuous scores: use probability of target (positive) class
            d_probs = F.softmax(d_logits, dim=1)
            target_cls = 1 if d_probs.size(1) > 1 else 0
            disc_preds.append(d_probs[:, target_cls])
            disc_labels.append((pid_t == int(forget_id)).long())

            target_mask = pid_t == int(forget_id)
            if torch.any(target_mask):
                logits_base = classifier(z_base[target_mask])
                probs_base = F.softmax(logits_base, dim=1)
                base_target_probs.append(probs_base[:, target_label])
                base_target_entropies.append((-probs_base * (probs_base + 1e-12).log()).sum(dim=1))

    disc_labels = torch.cat(disc_labels, dim=0).cpu().numpy()
    disc_scores = torch.cat(disc_preds, dim=0).cpu().numpy()  # probability of target class for AUC
    disc_preds_class = (disc_scores >= 0.5).astype(np.int64)   # for accuracy
    if base_target_probs:
        base_target_probs = torch.cat(base_target_probs, dim=0).cpu().numpy().tolist()
    else:
        base_target_probs = []
    if base_target_entropies:
        base_target_entropies = torch.cat(base_target_entropies, dim=0).cpu().numpy().tolist()
    else:
        base_target_entropies = []
    disc_acc = float((disc_preds_class == disc_labels).mean())
    try:
        from sklearn.metrics import roc_auc_score
        disc_auc = float(roc_auc_score(disc_labels, disc_scores))
    except:
        disc_auc = float("nan")

    diagnose_report = {
        "target_pid": forget_id,
        "discriminator": {"acc": disc_acc, "auc": disc_auc},
        "base_only_confidence": {
            "target_prob_mean": float(np.mean(base_target_probs)) if base_target_probs else None,
            "entropy_mean": float(np.mean(base_target_entropies)) if base_target_entropies else None,
        },
        "delta_ratio_final": delta_ratio_stats,
        "base_norm_final": base_norm_stats,
        "final_probe_metrics": final_eval,
        "gate_correctness": all(g["all_gate1_pids_are_target"] for g in gate_stats_all),
    }
    (out_dir / "removable_diagnose_report.json").write_text(json.dumps(diagnose_report, indent=2) + "\n")

    # Sanity report
    sanity = {
        "target_id": forget_id,
        "with_retain_mAP": final_eval.get("with", {}).get("retain", {}).get("mAP"),
        "with_forget_mAP": final_eval.get("with", {}).get("forget", {}).get("mAP"),
        "without_retain_mAP": final_eval.get("without", {}).get("retain", {}).get("mAP"),
        "without_forget_mAP": final_eval.get("without", {}).get("forget", {}).get("mAP"),
        "disc_auc": disc_auc,
        "base_norm_mean": base_norm_stats["mean"],
    }
    (out_dir / "sanity_report_removable.json").write_text(json.dumps(sanity, indent=2) + "\n")

    # PASS/FAIL check
    with_retain = final_eval.get("with", {}).get("retain", {}).get("mAP", 0)
    without_retain = final_eval.get("without", {}).get("retain", {}).get("mAP", 0)
    without_forget = final_eval.get("without", {}).get("forget", {}).get("mAP", 1)
    
    pass_retain = with_retain >= 0.95 and without_retain >= 0.95
    pass_forget = without_forget <= 0.10
    pass_disc = disc_auc <= 0.60
    pass_norm = base_norm_stats["mean"] >= 15 if base_norm_stats["mean"] else False
    
    overall_pass = pass_retain and pass_forget and pass_norm

    scorecard = {
        "PASS": overall_pass,
        "pass_retain": pass_retain,
        "pass_forget": pass_forget,
        "pass_disc": pass_disc,
        "pass_norm": pass_norm,
        "with_retain_mAP": with_retain,
        "without_retain_mAP": without_retain,
        "without_forget_mAP": without_forget,
        "disc_auc": disc_auc,
        "base_norm_mean": base_norm_stats["mean"],
    }
    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2) + "\n")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"WITH retain mAP:    {with_retain:.4f}")
    print(f"WITH forget mAP:    {final_eval.get('with', {}).get('forget', {}).get('mAP', 'N/A')}")
    print(f"WITHOUT retain mAP: {without_retain:.4f}")
    print(f"WITHOUT forget mAP: {without_forget:.4f}")
    print(f"Discriminator AUC:  {disc_auc:.4f}")
    print(f"Base norm mean:     {base_norm_stats['mean']:.1f}")
    print(f"PASS: {overall_pass}")
    print("=" * 60)

    (out_dir / "inference_modes.md").write_text(
        "# Removable Module Inference Modes\n\n"
        "WITH_MODULE: Load base_ckpt.pth + target_module.pth\n"
        "WITHOUT_MODULE: Load base_ckpt.pth only\n"
    )


if __name__ == "__main__":
    main()
