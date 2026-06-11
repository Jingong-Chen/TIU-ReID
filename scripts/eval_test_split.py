#!/usr/bin/env python
"""Evaluate model on standard test split (WITH/WITHOUT modes)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

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
    def __call__(self, img):
        return img.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)


class ToTensor:
    def __call__(self, img):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def build_val_transform(cfg):
    return Compose([
        Resize(cfg.INPUT.SIZE_TEST),
        ToTensor(),
        Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


def _compute_dist(qf, gf):
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.matmul(qf, gf.T)


def compute_cmc_map(dist, qp, gp, qc, gc, max_rank=50):
    num_q = dist.shape[0]
    indices = np.argsort(dist, axis=1)
    matches = (gp[indices] == qp[:, np.newaxis]).astype(np.int32)
    all_cmc, all_AP = [], []
    num_valid = 0
    for q_idx in range(num_q):
        order = indices[q_idx]
        remove = (gp[order] == qp[q_idx]) & (gc[order] == qc[q_idx])
        keep = ~remove
        m = matches[q_idx][keep]
        if not np.any(m):
            continue
        num_valid += 1
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
    return cmc, mAP, num_valid


@torch.no_grad()
def extract_features(model, target_module, mode, loader, device, target_pid, target_base_scale):
    model.eval()
    if target_module is not None:
        target_module.eval()
    
    feats, pids, camids = [], [], []
    for x, pid, camid in loader:
        x = x.to(device)
        pid_t = torch.as_tensor(pid, device=device)
        cam_label = torch.zeros(len(pid), dtype=torch.long, device=device)
        view_label = torch.zeros(len(pid), dtype=torch.long, device=device)
        
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
        camids.append(np.array(camid))
    
    return np.concatenate(feats), np.concatenate(pids), np.concatenate(camids)


class TestDataset(Dataset):
    def __init__(self, data_list, transform):
        self.data = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Handle both (path, pid, camid) and (path, pid, camid, ...) formats
        if len(item) >= 3:
            path, pid, camid = item[0], item[1], item[2]
        else:
            path, pid, camid = item
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, int(pid), int(camid)


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
        market_root = Path(reid_data) / "market1501"
        # TransReID expects root s.t. join(root, "market1501") has bounding_box_train
        inner = market_root / "Market-1501-v15.09.15"
        link = market_root / "market1501"
        if inner.exists() and (not link.exists() or link.resolve() != inner.resolve()):
            if link.exists():
                link.unlink()
            link.symlink_to(inner, target_is_directory=True)
        cfg.DATASETS.ROOT_DIR = str(market_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvp_dir", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--forget_id", type=int, default=2)
    ap.add_argument("--target_base_scale", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="", choices=["", "cuda", "cpu"],
                    help="Device for eval. Default: cuda if available and has memory, else cpu. Set to 'cpu' when GPU is OOM.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_transreid_imports(repo_root)

    from config import cfg
    from model import make_model
    from datasets import make_dataloader

    cfg.merge_from_file(args.cfg)
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.TEST.NECK_FEAT = "before"
    configure_data_root(cfg, repo_root)
    cfg.freeze()

    if args.device:
        device = args.device
    else:
        device = "cuda" if _cuda_works() else "cpu"
    mvp_dir = Path(args.mvp_dir)

    # Load model (use CPU when GPU is busy to avoid OOM)
    _, _, _, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    load_transreid_weights(model, str(mvp_dir / "base_ckpt.pth"))
    try:
        model.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "OutOfMemoryError" in type(e).__name__:
            print("[WARN] CUDA OOM, falling back to CPU (slower).", file=sys.stderr)
            device = "cpu"
            model.to(device)
        else:
            raise

    # Load target module
    tm_ckpt = torch.load(mvp_dir / "target_module.pth", map_location="cpu")
    tm_dim = int(tm_ckpt["dim"])
    tm_cfg = tm_ckpt.get("config", {})
    target_module = TargetModule(dim=tm_dim, hidden=tm_cfg.get("hidden"), dropout=tm_cfg.get("dropout", 0.0))
    target_module.load_state_dict(tm_ckpt["state_dict"])
    target_module.to(device)

    transform = build_val_transform(cfg)

    # Get test data from TransReID dataloader
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    # Extract query and gallery from val_loader's dataset
    val_dataset = val_loader.dataset
    query_data = []
    gallery_data = []
    
    # Parse the dataset structure
    if hasattr(val_dataset, 'query') and hasattr(val_dataset, 'gallery'):
        query_data = val_dataset.query
        gallery_data = val_dataset.gallery
    else:
        # Fallback: use the dataset directly
        all_data = list(val_dataset.dataset) if hasattr(val_dataset, 'dataset') else []
        query_data = all_data[:num_query]
        gallery_data = all_data[num_query:]

    print(f"[INFO] Query: {len(query_data)}, Gallery: {len(gallery_data)}")

    results = {}
    for mode in ["with", "without"]:
        tm = target_module if mode == "with" else None
        tbs = args.target_base_scale if mode == "with" else 0.0

        # Extract query features
        q_ds = TestDataset(query_data, transform)
        q_loader = DataLoader(q_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
        qf, qp, qc = extract_features(model, tm, mode, q_loader, device, args.forget_id, tbs)

        # Extract gallery features
        g_ds = TestDataset(gallery_data, transform)
        g_loader = DataLoader(g_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
        gf, gp, gc = extract_features(model, tm, mode, g_loader, device, args.forget_id, tbs)

        dist = _compute_dist(qf, gf)
        cmc, mAP, num_valid = compute_cmc_map(dist, qp, gp, qc, gc)

        results[mode] = {
            "mAP": float(mAP),
            "CMC@1": float(cmc[0]) if len(cmc) > 0 else 0.0,
            "CMC@5": float(cmc[4]) if len(cmc) > 4 else 0.0,
            "CMC@10": float(cmc[9]) if len(cmc) > 9 else 0.0,
            "num_valid_q": int(num_valid),
        }
        print(f"[{mode.upper()}] mAP={mAP:.4f}, CMC@1={cmc[0]:.4f}, num_valid_q={num_valid}")

        # Save individual metrics
        out_path = mvp_dir / f"metrics_test_{mode}.json"
        out_path.write_text(json.dumps(results[mode], indent=2) + "\n")

    # Update sanity report
    sanity_path = mvp_dir / "sanity_report_removable.json"
    if sanity_path.exists():
        sanity = json.loads(sanity_path.read_text())
    else:
        sanity = {}
    
    sanity["test_with_mAP"] = results["with"]["mAP"]
    sanity["test_with_CMC1"] = results["with"]["CMC@1"]
    sanity["test_without_mAP"] = results["without"]["mAP"]
    sanity["test_without_CMC1"] = results["without"]["CMC@1"]
    sanity_path.write_text(json.dumps(sanity, indent=2) + "\n")

    # Also update scorecard.json if present (for collect_group_results compatibility)
    scorecard_path = mvp_dir / "scorecard.json"
    if scorecard_path.exists():
        scorecard = json.loads(scorecard_path.read_text())
        scorecard["test_with_mAP"] = results["with"]["mAP"]
        scorecard["test_without_mAP"] = results["without"]["mAP"]
        scorecard_path.write_text(json.dumps(scorecard, indent=2) + "\n")

    print(f"\n[OK] Test split evaluation complete")
    print(f"  WITH: mAP={results['with']['mAP']:.4f}, CMC@1={results['with']['CMC@1']:.4f}")
    print(f"  WITHOUT: mAP={results['without']['mAP']:.4f}, CMC@1={results['without']['CMC@1']:.4f}")


if __name__ == "__main__":
    main()
