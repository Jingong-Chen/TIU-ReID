#!/usr/bin/env python
"""
Probe AUC: freeze removal-deployment embedding z, train a post-hoc probe (linear/MLP)
on target vs non-target; report val ROC AUC. This measures separability with a strong
probe, not the in-training discriminator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

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
    except Exception:
        return False


def load_base_ckpt(model, weights_path: str) -> None:
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_state = model.state_dict()
    cleaned = {k.replace("module.", ""): v for k, v in state.items() if k.replace("module.", "") in model_state and model_state[k.replace("module.", "")].shape == v.shape}
    model.load_state_dict(cleaned, strict=False)


class ProbeMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory (base_ckpt.pth, scorecard.json)")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--forget_id", type=int, default=2)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--dataset", default="market1501")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--probe_seed", type=int, default=42)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--probe_hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_transreid_imports(repo_root)

    import collections.abc
    import types
    if "torch._six" not in sys.modules:
        m = types.ModuleType("torch._six")
        m.container_abcs = collections.abc
        m.string_classes = (str, bytes)
        m.int_classes = (int,)
        sys.modules["torch._six"] = m

    from config import cfg
    from model import make_model
    from datasets.make_dataloader import Compose, Normalize, Resize, ToTensor

    cfg.merge_from_file(args.cfg)
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.TEST.NECK_FEAT = "before"
    reid_data = os.environ.get("REID_DATA_DIR", str(repo_root / "data"))
    if args.data_root:
        reid_data = args.data_root
    market_root = Path(reid_data) / "market1501"
    inner = market_root / "Market-1501-v15.09.15"
    link = market_root / "market1501"
    if inner.exists() and (not link.exists() or link.resolve() != inner.resolve()):
        if link.exists():
            link.unlink()
        link.symlink_to(inner, target_is_directory=True)
    cfg.DATASETS.ROOT_DIR = str(market_root)
    cfg.freeze()

    device = "cuda" if _cuda_works() else "cpu"
    run_dir = Path(args.run_dir)
    base_ckpt = run_dir / "base_ckpt.pth"
    if not base_ckpt.exists():
        print(f"[ERROR] {base_ckpt} not found", file=sys.stderr)
        return 1

    items = read_train_items(args.dataset, Path(reid_data).resolve())
    forget_id = args.forget_id
    transform = Compose([
        ToTensor(),
        Resize(cfg.INPUT.SIZE_TEST),
        Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    class TrainImageDataset(torch.utils.data.Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            it = self.items[idx]
            path = it.path
            if not Path(path).exists() and os.environ.get("REID_DATA_DIR"):
                data_dir = os.environ["REID_DATA_DIR"]
                if "data" in path:
                    path = str(Path(data_dir) / path.split("data/", 1)[-1])
            img = Image.open(path).convert("RGB")
            x = self.transform(img)
            return x, it.pid

    train_ds = TrainImageDataset(items, transform)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))

    num_classes = len({it.pid for it in items})
    model = make_model(cfg, num_class=num_classes, camera_num=6, view_num=1)
    load_base_ckpt(model, str(base_ckpt))
    model.to(device)
    model.eval()

    z_list, pid_list = [], []
    with torch.no_grad():
        for batch in loader:
            x, pid = batch
            x = x.to(device)
            cam_label = torch.zeros(x.size(0), dtype=torch.long, device=device)
            view_label = torch.zeros(x.size(0), dtype=torch.long, device=device)
            out = model(x, cam_label=cam_label, view_label=view_label)
            z = out[1] if isinstance(out, tuple) else out
            z_list.append(z.cpu().numpy())
            pid_list.append(pid.numpy())
    Z = np.concatenate(z_list, axis=0).astype(np.float32)
    P = np.concatenate(pid_list, axis=0)
    labels = (P == forget_id).astype(np.int64)

    rng = np.random.default_rng(args.probe_seed)
    n = len(Z)
    idx = np.arange(n)
    rng.shuffle(idx)
    nval = max(1, int(n * args.val_frac))
    val_idx = idx[:nval]
    train_idx = idx[nval:]
    Z_tr, Y_tr = Z[train_idx], labels[train_idx]
    Z_val, Y_val = Z[val_idx], labels[val_idx]

    feat_dim = Z.shape[1]
    probe = ProbeMLP(feat_dim, hidden=args.probe_hidden).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    ds_tr = TensorDataset(torch.from_numpy(Z_tr), torch.from_numpy(Y_tr))
    dl_tr = DataLoader(ds_tr, batch_size=min(256, len(Z_tr)), shuffle=True)
    for _ in range(args.probe_epochs):
        probe.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            logits = probe(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(torch.from_numpy(Z_val).to(device))
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().numpy()

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(Y_val, scores))
    except Exception:
        auc = float("nan")
    auc_sym = max(auc, 1.0 - auc) if not np.isnan(auc) else float("nan")

    scorecard_path = run_dir / "scorecard.json"
    if scorecard_path.exists():
        sc = json.loads(scorecard_path.read_text())
    else:
        sc = {}
    sc["ProbeAUC"] = auc
    sc["ProbeAUC_symmetric"] = auc_sym
    scorecard_path.write_text(json.dumps(sc, indent=2) + "\n")
    print(f"[OK] ProbeAUC={auc:.4f} (symmetric={auc_sym:.4f}) -> {scorecard_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
