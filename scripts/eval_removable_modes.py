from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from unlearning_reid.eval.reid_metrics import compute_cmc_map
from unlearning_reid.models.removable import TargetModule


def _ensure_transreid_imports(repo_root: Path) -> None:
    tr = repo_root / "third_party" / "TransReID"
    if str(tr) not in sys.path:
        sys.path.insert(0, str(tr))


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


def configure_data_root(cfg, repo_root: Path) -> None:
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
    elif ds_name in ("dukemtmc", "dukemtmc-reid"):
        src = Path(reid_data) / "dukemtmc-reid" / "DukeMTMC-reID"
        link = tmp_root / "dukemtmc"
        if not link.exists():
            link.symlink_to(src, target_is_directory=True)
        cfg.DATASETS.ROOT_DIR = str(tmp_root)
    else:
        cfg.DATASETS.ROOT_DIR = str(tmp_root)


def load_transreid_weights(model: torch.nn.Module, weights_path: str) -> None:
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_state = model.state_dict()
    cleaned = {}
    for k, v in state.items():
        key = k.replace("module.", "")
        if key in model_state and model_state[key].shape == v.shape:
            cleaned[key] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")


def _compute_dist(qf: np.ndarray, gf: np.ndarray) -> np.ndarray:
    if torch.cuda.is_available():
        with torch.no_grad():
            q = torch.from_numpy(qf).cuda(non_blocking=True)
            g = torch.from_numpy(gf).cuda(non_blocking=True)
            q = q / (q.norm(dim=1, keepdim=True) + 1e-12)
            g = g / (g.norm(dim=1, keepdim=True) + 1e-12)
            dist = 1.0 - torch.mm(q, g.t())
            return dist.cpu().numpy()
    qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.matmul(qn, gn.T)


@torch.no_grad()
def extract(model, target_module, mode: str, loader, device, target_pid: int, target_base_scale: float):
    feats, pids, camids, paths = [], [], [], []
    model.eval()
    target_module_used = False
    if target_module is not None:
        target_module.eval()
        if mode == "with":
            target_module_used = True

    for x, pid, cam_label, view_label, path in loader:
        x = x.to(device)
        cam_label = cam_label.to(device)
        view_label = view_label.to(device)
        pid_t = torch.as_tensor(pid, device=device)

        z = model(x, cam_label=cam_label, view_label=view_label)
        if mode == "with" and target_module is not None:
            delta = target_module(z)
            mask = (pid_t == int(target_pid)).float().unsqueeze(1)
            if target_base_scale != 0.0:
                z = z + delta * mask - target_base_scale * z * mask
            else:
                z = z + delta * mask
        # WITHOUT模式：明确不使用target_module（即使加载了也不调用）

        feats.append(z.detach().cpu().float().numpy())
        pids.append(np.array(pid))
        camids.append(np.array(cam_label.cpu().numpy()))
        paths += list(path)

    feats = np.concatenate(feats, axis=0)
    pids = np.concatenate(pids, axis=0)
    camids = np.concatenate(camids, axis=0)
    return feats, pids, camids, paths, target_module_used


def eval_npz(q_npz: Path, g_npz: Path) -> dict:
    q = np.load(q_npz, allow_pickle=True)
    g = np.load(g_npz, allow_pickle=True)
    qf, qp, qc = q["feats"].astype(np.float32), q["pids"].astype(np.int64), q["camids"].astype(np.int64)
    gf, gp, gc = g["feats"].astype(np.float32), g["pids"].astype(np.int64), g["camids"].astype(np.int64)
    dist = _compute_dist(qf, gf)
    cmc, mAP = compute_cmc_map(dist, qp, gp, qc, gc, max_rank=50)
    return {
        "mAP": mAP,
        "CMC@1": float(cmc[0]),
        "CMC@5": float(cmc[4]) if len(cmc) > 4 else None,
        "CMC@10": float(cmc[9]) if len(cmc) > 9 else None,
    }


def ensure_probe_dir(repo_root: Path, probe_dir: Path, split_dir: Path, feat_train_npz: Path, distractor_mode: str, distractor_ids: int, dataset: str) -> None:
    if probe_dir.exists():
        return
    cmd = [
        os.environ.get("PYTHON", "python"),
        str(repo_root / "scripts" / "make_probe_sets.py"),
        "--dataset",
        dataset,
        "--seed",
        "0",
        "--split_dir",
        str(split_dir),
        "--out_dir",
        str(probe_dir),
        "--distractor_mode",
        distractor_mode,
        "--forget_distractor_ids",
        str(distractor_ids),
        "--feat_train_npz",
        str(feat_train_npz),
    ]
    subprocess.check_call(cmd)
    subprocess.check_call(
        [
            os.environ.get("PYTHON", "python"),
            str(repo_root / "scripts" / "probe_stats.py"),
            "--probe_dir",
            str(probe_dir),
            "--split_dir",
            str(split_dir),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--target_module", required=True)
    ap.add_argument("--forget_id", type=int, required=True)
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--feat_train_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", default="market1501")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--distractor_mode", default="hard")
    ap.add_argument("--forget_distractor_ids", type=int, default=200)
    ap.add_argument("--neck_feat", default="before", choices=["before", "after"])
    ap.add_argument("--target_base_scale", type=float, default=0.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_transreid_imports(repo_root)

    from config import cfg  # type: ignore
    from datasets import make_dataloader  # type: ignore
    from model import make_model  # type: ignore

    cfg.merge_from_file(args.cfg)
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.TEST.NECK_FEAT = args.neck_feat
    configure_data_root(cfg, repo_root)
    cfg.freeze()

    device = "cuda" if _cuda_works() else "cpu"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe_dir)
    split_dir = Path(args.split_dir)
    feat_train_npz = Path(args.feat_train_npz)

    ensure_probe_dir(
        repo_root,
        probe_dir,
        split_dir,
        feat_train_npz,
        args.distractor_mode,
        args.forget_distractor_ids,
        args.dataset,
    )

    # Build model with correct num_classes/camera/view from dataset
    _, _, _, _num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    load_transreid_weights(model, args.base_ckpt)
    model.to(device)

    tm = None
    tm_path = Path(args.target_module)
    tm_ckpt = torch.load(tm_path, map_location="cpu")
    tm_dim = int(tm_ckpt["dim"])
    tm_cfg = tm_ckpt.get("config", {}) if isinstance(tm_ckpt.get("config", {}), dict) else {}
    tm = TargetModule(dim=tm_dim, hidden=tm_cfg.get("hidden"), dropout=tm_cfg.get("dropout", 0.0))
    tm.load_state_dict(tm_ckpt["state_dict"], strict=True)
    tm.to(device)

    transform = build_val_transform(cfg)
    sets = {
        "retain_query": probe_dir / "retain_query.jsonl",
        "retain_gallery": probe_dir / "retain_gallery.jsonl",
        "forget_query": probe_dir / "forget_query.jsonl",
        "forget_gallery": probe_dir / "forget_gallery.jsonl",
    }

    mode_verification = {}
    def _extract_mode(mode: str):
        features_dir = out_dir / f"features_{mode}"
        features_dir.mkdir(parents=True, exist_ok=True)
        for name, jsonl_path in sets.items():
            ds = JsonlImageDataset(jsonl_path, transform)
            loader = DataLoader(
                ds,
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device == "cuda"),
            )
            feats, pids, camids, paths, tm_used = extract(
                model,
                tm,
                mode,
                loader,
                device,
                args.forget_id,
                args.target_base_scale,
            )
            out_npz = features_dir / f"{name}.npz"
            np.savez_compressed(out_npz, feats=feats, pids=pids, camids=camids, paths=np.array(paths, dtype=object))
            if name == "forget_query":  # 记录一次即可
                mode_verification[mode] = {
                    "target_module_loaded": tm is not None,
                    "target_module_used_in_forward": tm_used,
                    "note": "WITHOUT mode should have target_module_used_in_forward=False",
                }

    _extract_mode("with")
    _extract_mode("without")

    metrics = {}
    for mode in ("with", "without"):
        features_dir = out_dir / f"features_{mode}"
        m_retain = eval_npz(features_dir / "retain_query.npz", features_dir / "retain_gallery.npz")
        m_forget = eval_npz(features_dir / "forget_query.npz", features_dir / "forget_gallery.npz")
        (out_dir / f"metrics_{mode}_retain.json").write_text(json.dumps(m_retain, indent=2) + "\n")
        (out_dir / f"metrics_{mode}_forget.json").write_text(json.dumps(m_forget, indent=2) + "\n")
        metrics[mode] = {"retain": m_retain, "forget": m_forget}

    report = {
        "target_id": int(args.forget_id),
        "metrics": metrics,
        "delta": {
            "retain_mAP_drop": metrics["with"]["retain"]["mAP"] - metrics["without"]["retain"]["mAP"],
            "forget_mAP_drop": metrics["with"]["forget"]["mAP"] - metrics["without"]["forget"]["mAP"],
            "retain_CMC1_drop": metrics["with"]["retain"]["CMC@1"] - metrics["without"]["retain"]["CMC@1"],
            "forget_CMC1_drop": metrics["with"]["forget"]["CMC@1"] - metrics["without"]["forget"]["CMC@1"],
        },
        "notes": {
            "CMC@1 equals NN hit@1 under cosine distance": True,
        },
        "diagnostic_10_without_mode_verification": mode_verification,
    }
    (out_dir / "sanity_report_removable.json").write_text(json.dumps(report, indent=2) + "\n")

    (out_dir / "inference_modes.md").write_text(
        "\n".join(
            [
                "# Removable Module Inference Modes",
                "",
                "WITH_MODULE:",
                "- Load `base_ckpt.pth` + `target_module.pth`.",
                "- Enable gate for target pid (FORGET_ID) only.",
                "",
                "WITHOUT_MODULE:",
                "- Load `base_ckpt.pth` only, or force gate=0 for all samples.",
                "",
                "Use `scripts/eval_removable_modes.sh` to generate both modes on retain/forget probes.",
                "",
            ]
        )
        + "\n"
    )

    print("[OK] removable eval done ->", out_dir)


if __name__ == "__main__":
    main()
