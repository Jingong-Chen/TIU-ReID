#!/usr/bin/env python
"""Baseline: Fine-tune without target samples (naive delete then finetune)."""
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


class PKSampler(Sampler[int]):
    """PK sampler for triplet loss: P identities, K instances each."""
    def __init__(self, pid_list, batch_size, num_instances, seed=0):
        self.index_by_pid = {}
        for idx, pid in enumerate(pid_list):
            self.index_by_pid.setdefault(pid, []).append(idx)
        self.pids = list(self.index_by_pid.keys())
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.p = max(2, self.batch_size // self.num_instances)
        self.seed = int(seed)
        self.epoch = 0
        self.num_batches = max(1, math.ceil(len(self.pids) / self.p))
    
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pids = list(self.pids)
        rng.shuffle(pids)
        
        step = 0
        for _ in range(self.num_batches):
            batch_pids = pids[step : step + self.p]
            step += self.p
            if len(batch_pids) < self.p:
                batch_pids += rng.sample(self.pids, self.p - len(batch_pids))
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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_instances", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_id", type=float, default=1.0)
    ap.add_argument("--lambda_tri", type=float, default=1.0)
    ap.add_argument("--triplet_margin", type=float, default=0.3)
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

    # Filter out target identity
    items = read_train_items(args.dataset, data_root)
    items_filtered = [it for it in items if it.pid != forget_id]
    print(f"[INFO] Filtered out target ID {forget_id}: {len(items)} -> {len(items_filtered)} samples")

    pids = sorted({it.pid for it in items_filtered})
    pid_to_label = {pid: i for i, pid in enumerate(pids)}

    dataset = TrainImageDataset(
        [(it.path, it.pid, it.camid) for it in items_filtered],
        pid_to_label,
        build_train_transform(cfg),
    )

    # Get num_classes from filtered dataset, camera/view from original items
    num_classes = len(pid_to_label)
    camera_ids = {it.camid for it in items_filtered}
    camera_num = max(camera_ids) + 1 if camera_ids else 6
    view_num = 0  # Market1501 doesn't have view labels
    
    # Use PK sampler for proper triplet loss
    pid_list = [it.pid for it in items_filtered]
    sampler = PKSampler(pid_list, args.batch, args.num_instances, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    load_transreid_weights(model, args.weights)
    model.to(device)

    classifier = nn.Linear(model.in_planes if hasattr(model, "in_planes") else 768, num_classes, bias=False).to(device)
    nn.init.normal_(classifier.weight, std=0.001)

    params = [
        {"params": model.parameters(), "lr": args.lr},
        {"params": classifier.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    triplet_loss_fn = TripletLoss(margin=args.triplet_margin)
    scaler = amp.GradScaler(enabled=(device == "cuda"))

    print(f"[INFO] Fine-tune without target ID {forget_id}")

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        classifier.train()

        epoch_loss = 0.0
        n_samples = 0

        for step, batch in enumerate(loader, start=1):
            img, pid, label, cam_label, view_label = batch
            img = img.to(device)
            label = label.to(device)
            cam_label = cam_label.to(device)
            view_label = view_label.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=(device == "cuda")):
                z = model(img, cam_label=cam_label, view_label=view_label)
                if isinstance(z, tuple):
                    z = z[1] if len(z) > 1 else z[0]
                if isinstance(z, list):
                    z = z[0]

                logits = classifier(z)
                loss_id = F.cross_entropy(logits, label)
                loss_metric, _, _ = triplet_loss_fn(z, label)
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

    torch.save(model.state_dict(), out_dir / "base_ckpt.pth")
    torch.save(classifier.state_dict(), out_dir / "classifier_head.pth")

    run_args = vars(args)
    run_args.update({
        "target_pid": int(forget_id),
        "num_classes": num_classes,
    })
    (out_dir / "run_args.json").write_text(json.dumps(run_args, indent=2) + "\n")
    (out_dir / "teacher_cfg_path.txt").write_text(str(Path(args.cfg).resolve()) + "\n")
    (out_dir / "teacher_weights_path.txt").write_text(str(Path(args.weights).resolve()) + "\n")


if __name__ == "__main__":
    main()
