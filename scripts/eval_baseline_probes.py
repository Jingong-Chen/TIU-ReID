#!/usr/bin/env python
"""Evaluate baseline teacher on retain/forget probes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from unlearning_reid.eval.reid_metrics import compute_cmc_map


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


class JsonlImageDataset(Dataset):
    def __init__(self, jsonl_path: Path, transform):
        self.items = []
        with jsonl_path.open("r") as f:
            for line in f:
                d = json.loads(line)
                self.items.append(d)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        img = Image.open(d["path"]).convert("RGB")
        x = self.transform(img)
        pid = int(d["pid"])
        camid = int(d["camid"])
        cam_label = max(0, camid - 1)
        view_label = 0
        return x, pid, cam_label, view_label, d["path"]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, img: Image.Image):
        return img.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)


class ToTensor:
    def __call__(self, img: Image.Image):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor):
        return (tensor - self.mean) / self.std


def build_val_transform(cfg):
    return Compose(
        [
            Resize(cfg.INPUT.SIZE_TEST),
            ToTensor(),
            Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )


def load_transreid_weights(model, weights_path: str) -> None:
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


def configure_data_root(cfg, repo_root):
    reid_data = os.environ.get("REID_DATA_DIR", str(repo_root / "data"))
    reid_out = os.environ.get("REID_OUTPUT_DIR", str(repo_root / "output"))
    tmp_root = Path(reid_out) / "transreid_data"
    tmp_root.mkdir(parents=True, exist_ok=True)
    ds_name = cfg.DATASETS.NAMES
    if isinstance(ds_name, (list, tuple)):
        ds_name = ds_name[0]
    ds_name = str(ds_name).lower()
    if ds_name == "market1501":
        src = Path(reid_data) / "market1501" / "Market-1501-v15.09.15"
        link = tmp_root / "market1501"
        if not link.exists():
            link.symlink_to(src, target_is_directory=True)
        cfg.DATASETS.ROOT_DIR = str(tmp_root)


@torch.no_grad()
def extract_features(model, loader, device, target_module=None, mode="without", target_pid=None, target_base_scale=0.0):
    model.eval()
    if target_module is not None:
        target_module.eval()

    feats, pids, camids = [], [], []
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

        feats.append(z.cpu().float().numpy())
        pids.append(np.array(pid))
        camids.append(cam_label.cpu().numpy())

    return np.concatenate(feats), np.concatenate(pids), np.concatenate(camids)


def _compute_dist(qf, gf):
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.matmul(qf, gf.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_module", default=None, help="Path to target_module.pth (optional)")
    ap.add_argument("--mode", default="without", choices=["with", "without"], help="WITH_MODULE or WITHOUT_MODULE")
    ap.add_argument("--target_pid", type=int, default=None)
    ap.add_argument("--target_base_scale", type=float, default=0.0)
    ap.add_argument("--skip_if_exists", action="store_true", help="Skip if output files already exist")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_transreid_imports(repo_root)

    from config import cfg
    from model import make_model
    from datasets import make_dataloader
    from unlearning_reid.models.removable import TargetModule

    cfg.merge_from_file(args.cfg)
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.TEST.NECK_FEAT = "before"
    configure_data_root(cfg, repo_root)
    cfg.freeze()

    device = "cuda" if _cuda_works() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe_dir)
    
    # Skip if already exists
    if args.skip_if_exists and (out_dir / "summary_base.json").exists():
        print(f"[SKIP] {out_dir} already evaluated")
        return

    # Load model
    _, _, _, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    load_transreid_weights(model, args.weights)
    model.to(device)

    # Load target module if needed
    target_module = None
    if args.mode == "with" and args.target_module and Path(args.target_module).exists():
        try:
            tm_ckpt = torch.load(args.target_module, map_location="cpu")
            if isinstance(tm_ckpt, dict) and "state_dict" in tm_ckpt:
                tm_dim = int(tm_ckpt.get("dim", 768))
                tm_cfg = tm_ckpt.get("config", {})
                target_module = TargetModule(dim=tm_dim, hidden=tm_cfg.get("hidden", 512), dropout=tm_cfg.get("dropout", 0.0))
                target_module.load_state_dict(tm_ckpt["state_dict"])
            else:
                # Fallback: assume it's a state dict
                model.eval()
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1], device=device)
                    cam_label = torch.zeros(1, dtype=torch.long, device=device)
                    view_label = torch.zeros(1, dtype=torch.long, device=device)
                    out = model(dummy, cam_label=cam_label, view_label=view_label)
                    feat_dim = int(out[1].shape[1] if isinstance(out, tuple) else out.shape[1])
                target_module = TargetModule(dim=feat_dim, hidden=512, dropout=0.0)
                target_module.load_state_dict(tm_ckpt)
            target_module.to(device)
        except Exception as e:
            print(f"[WARN] Failed to load target_module: {e}")
            target_module = None

    transform = build_val_transform(cfg)
    results = {}

    for probe_type in ["retain", "forget"]:
        q_path = probe_dir / f"{probe_type}_query.jsonl"
        g_path = probe_dir / f"{probe_type}_gallery.jsonl"
        if not q_path.exists() or not g_path.exists():
            print(f"[WARN] Missing {probe_type} probe files")
            continue

        q_ds = JsonlImageDataset(q_path, transform)
        g_ds = JsonlImageDataset(g_path, transform)
        q_loader = DataLoader(q_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
        g_loader = DataLoader(g_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))

        qf, qp, qc = extract_features(model, q_loader, device, target_module, args.mode, args.target_pid, args.target_base_scale)
        gf, gp, gc = extract_features(model, g_loader, device, target_module, args.mode, args.target_pid, args.target_base_scale)

        dist = _compute_dist(qf, gf)
        cmc, mAP = compute_cmc_map(dist, qp, gp, qc, gc)
        num_valid = len(qp)  # All queries are valid

        results[probe_type] = {
            "mAP": float(mAP),
            "CMC@1": float(cmc[0]) if len(cmc) > 0 else 0.0,
            "CMC@5": float(cmc[4]) if len(cmc) > 4 else 0.0,
            "CMC@10": float(cmc[9]) if len(cmc) > 9 else 0.0,
            "num_valid_q": int(num_valid),
        }

        # Save individual metrics
        out_path = out_dir / f"metrics_base_{probe_type}.json"
        out_path.write_text(json.dumps(results[probe_type], indent=2) + "\n")
        print(f"[{probe_type.upper()}] mAP={mAP:.4f}, CMC@1={cmc[0]:.4f}, num_valid_q={num_valid}")

    # Save summary
    summary = {
        "mode": args.mode,
        "retain_mAP": results.get("retain", {}).get("mAP", 0),
        "retain_CMC1": results.get("retain", {}).get("CMC@1", 0),
        "forget_mAP": results.get("forget", {}).get("mAP", 0),
        "forget_CMC1": results.get("forget", {}).get("CMC@1", 0),
    }
    (out_dir / "summary_base.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n[OK] Evaluation complete: {args.mode}")
    print(f"  Retain: mAP={summary['retain_mAP']:.4f}, CMC@1={summary['retain_CMC1']:.4f}")
    print(f"  Forget: mAP={summary['forget_mAP']:.4f}, CMC@1={summary['forget_CMC1']:.4f}")


if __name__ == "__main__":
    main()
