#!/usr/bin/env python
"""Train WITH_ONLY baseline: only train TargetModule, no forget loss."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset, Sampler

from unlearning_reid.datasets.common import read_train_items
from unlearning_reid.models.removable import TargetModule


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
    def __init__(self, items, pid_to_label, transform):
        self.items = items
        self.pid_to_label = pid_to_label
        self.transform = transform

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
        return img, int(pid), int(label), cam_label, view_label

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


def build_train_transform(cfg):
    from datasets.make_dataloader import Compose, Normalize, Pad, RandomCrop, RandomHorizontalFlip, Resize, ToTensor
    from timm.data.random_erasing import RandomErasing
    return Compose([
        ToTensor(),
        Resize(cfg.INPUT.SIZE_TRAIN),
        RandomHorizontalFlip(p=cfg.INPUT.PROB),
        Pad(cfg.INPUT.PADDING),
        RandomCrop(cfg.INPUT.SIZE_TRAIN),
        Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", default="market1501")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--forget_id", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_instances", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr_module", type=float, default=3e-4)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--lambda_id", type=float, default=1.0)
    ap.add_argument("--lambda_tri", type=float, default=1.0)
    ap.add_argument("--triplet_margin", type=float, default=0.3)
    ap.add_argument("--module_hidden", type=int, default=512)
    ap.add_argument("--module_dropout", type=float, default=0.1)
    ap.add_argument("--target_base_scale", type=float, default=1.0)
    ap.add_argument("--neck_feat", default="before")
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

    dataset = TrainImageDataset(
        [(it.path, it.pid, it.camid) for it in items],
        pid_to_label,
        build_train_transform(cfg),
    )

    pid_list = [it.pid for it in items]
    sampler = TargetPKSampler(pid_list, forget_id, args.batch, args.num_instances, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True)

    num_classes = len(pid_to_label)
    model = make_model(cfg, num_class=num_classes, camera_num=0, view_num=0)
    load_transreid_weights(model, args.weights)
    model.to(device)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1], device=device)
        cam_label = torch.zeros(2, dtype=torch.long, device=device)
        view_label = torch.zeros(2, dtype=torch.long, device=device)
        out = model(dummy, cam_label=cam_label, view_label=view_label)
        feat_dim = int(out[1].shape[1] if isinstance(out, tuple) else out.shape[1])

    target_module = TargetModule(dim=feat_dim, hidden=args.module_hidden, dropout=args.module_dropout).to(device)
    classifier = nn.Linear(feat_dim, num_classes, bias=False).to(device)
    nn.init.normal_(classifier.weight, std=0.001)

    params = [
        {"params": target_module.parameters(), "lr": args.lr_module},
        {"params": classifier.parameters(), "lr": args.lr_head},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)

    triplet_loss_fn = TripletLoss(margin=args.triplet_margin)
    scaler = amp.GradScaler(enabled=(device == "cuda"))

    print(f"[INFO] WITH_ONLY Training (no forget loss)")
    print(f"[INFO] Target: pid={forget_id}, label={target_label}")
    print(f"[INFO] Base model: FROZEN")

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.eval()
        target_module.train()
        classifier.train()

        epoch_loss = 0.0
        n_samples = 0

        for step, batch in enumerate(loader, start=1):
            img, pid, label, cam_label, view_label = batch
            img = img.to(device)
            pid = pid.to(device)
            label = label.to(device)
            cam_label = cam_label.to(device)
            view_label = view_label.to(device)

            target_mask = (pid == int(forget_id)).float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=(device == "cuda")):
                z_base = model(img, cam_label=cam_label, view_label=view_label)
                if isinstance(z_base, tuple):
                    z_base = z_base[1]

                delta = target_module(z_base)
                delta = delta * target_mask

                z_on = z_base + delta
                if args.target_base_scale != 0.0:
                    z_on = z_base + target_mask * (delta - args.target_base_scale * z_base)

                logits = classifier(z_on)
                loss_id = F.cross_entropy(logits, label)
                loss_metric, _, _ = triplet_loss_fn(z_on, label)

                loss = args.lambda_id * loss_id + args.lambda_tri * loss_metric

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = img.size(0)
            epoch_loss += float(loss.detach()) * bs
            n_samples += bs

        avg_loss = epoch_loss / max(n_samples, 1)
        print(f"[EPOCH {epoch}] loss={avg_loss:.4f}")

    elapsed = time.time() - start
    print(f"[OK] Training done in {elapsed:.1f}s")

    # Save checkpoints
    torch.save(model.state_dict(), out_dir / "base_ckpt.pth")
    torch.save({
        "dim": feat_dim,
        "config": {"hidden": args.module_hidden, "dropout": args.module_dropout},
        "state_dict": target_module.state_dict(),
    }, out_dir / "target_module.pth")

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


if __name__ == "__main__":
    main()
