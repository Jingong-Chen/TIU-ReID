from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _ensure_transreid_imports(repo_root: Path) -> None:
    tr = repo_root / "third_party" / "TransReID"
    if str(tr) not in sys.path:
        sys.path.insert(0, str(tr))


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


def _cuda_works() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        a = torch.randn(4, 4, device="cuda")
        b = torch.randn(4, 4, device="cuda")
        _ = a @ b
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


@torch.no_grad()
def extract(model, loader, device):
    feats, pids, camids, paths = [], [], [], []
    model.eval()
    for x, pid, cam_label, view_label, path in loader:
        x = x.to(device)
        cam_label = cam_label.to(device)
        view_label = view_label.to(device)
        feat = model(x, cam_label=cam_label, view_label=view_label)
        feat = feat.detach().cpu().float().numpy()
        feats.append(feat)
        pids.append(np.array(pid))
        camids.append(np.array(cam_label.cpu().numpy()))
        paths += list(path)
    feats = np.concatenate(feats, axis=0)
    pids = np.concatenate(pids, axis=0)
    camids = np.concatenate(camids, axis=0)
    return feats, pids, camids, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="TransReID config used to build model")
    ap.add_argument("--weights", required=True, help="Teacher checkpoint path (*.pth)")
    ap.add_argument("--jsonl", required=True, help="Input jsonl list (path,pid,camid)")
    ap.add_argument("--out", required=True, help="Output npz cache")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_transreid_imports(repo_root)

    from config import cfg
    from datasets import make_dataloader
    from model import make_model

    cfg.merge_from_file(args.cfg)
    cfg.defrost()
    cfg.merge_from_list(
        [
            "MODEL.PRETRAIN_CHOICE",
            "none",
        ]
    )
    # Ensure dataset root points to our read-only mirror under output/
    reid_data = os.environ.get("REID_DATA_DIR", str(repo_root / "data"))
    reid_out = os.environ.get("REID_OUTPUT_DIR", str(repo_root / "output"))
    tmp_root = Path(reid_out) / "transreid_data"
    tmp_root.mkdir(parents=True, exist_ok=True)
    ds_name = cfg.DATASETS.NAMES
    if isinstance(ds_name, (list, tuple)):
        ds_name = ds_name[0]
    ds_name = str(ds_name)
    if ds_name.lower() == "market1501":
        src = Path(reid_data) / "market1501" / "Market-1501-v15.09.15"
        link = tmp_root / "market1501"
        if not link.exists():
            link.symlink_to(src, target_is_directory=True)
        cfg.DATASETS.ROOT_DIR = str(tmp_root)
    elif ds_name.lower() in ("dukemtmc", "dukemtmc-reid"):
        src = Path(reid_data) / "dukemtmc-reid" / "DukeMTMC-reID"
        link = tmp_root / "dukemtmc"
        if not link.exists():
            link.symlink_to(src, target_is_directory=True)
        cfg.DATASETS.ROOT_DIR = str(tmp_root)
    else:
        cfg.DATASETS.ROOT_DIR = str(tmp_root)
    cfg.freeze()
    cfg.freeze()

    device = "cuda" if _cuda_works() else "cpu"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Build model with correct num_classes/camera/view from dataset
    _, _, _, _num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(args.weights)
    model.to(device)

    ds = JsonlImageDataset(Path(args.jsonl), build_val_transform(cfg))
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    feats, pids, camids, paths = extract(model, loader, device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feats=feats, pids=pids, camids=camids, paths=np.array(paths, dtype=object))
    print("[OK] wrote:", out, "feats", feats.shape)


if __name__ == "__main__":
    main()


