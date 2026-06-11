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

from unlearning_reid.models.removable import TargetModule

# Import TransReID utilities
repo_root = Path(__file__).resolve().parents[1]
tr = repo_root / "third_party" / "TransReID"
if str(tr) not in sys.path:
    sys.path.insert(0, str(tr))

from config import cfg  # type: ignore
from datasets import make_dataloader  # type: ignore
from model import make_model  # type: ignore


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


@torch.no_grad()
def compute_delta_stats(model, target_module, loader, device, target_pid: int, target_base_scale: float):
    """诊断1: TargetModule贡献幅度"""
    model.eval()
    target_module.eval()
    
    delta_norms = []
    base_norms = []
    ratios = []
    
    for x, pid, cam_label, view_label, _ in loader:
        x = x.to(device)
        cam_label = cam_label.to(device)
        view_label = view_label.to(device)
        pid_t = torch.as_tensor(pid, device=device)
        
        z_base = model(x, cam_label=cam_label, view_label=view_label)
        delta_raw = target_module(z_base)
        mask = (pid_t == int(target_pid)).float().unsqueeze(1)
        delta = delta_raw * mask
        
        target_mask = pid_t == int(target_pid)
        if torch.any(target_mask):
            delta_n = torch.norm(delta[target_mask], dim=1).cpu().numpy()
            base_n = torch.norm(z_base[target_mask], dim=1).cpu().numpy()
            ratio = delta_n / (base_n + 1e-12)
            
            delta_norms.extend(delta_n.tolist())
            base_norms.extend(base_n.tolist())
            ratios.extend(ratio.tolist())
    
    if not delta_norms:
        return None
    
    delta_norms = np.array(delta_norms)
    base_norms = np.array(base_norms)
    ratios = np.array(ratios)
    
    return {
        "delta_norm": {
            "mean": float(delta_norms.mean()),
            "p50": float(np.percentile(delta_norms, 50)),
            "p90": float(np.percentile(delta_norms, 90)),
        },
        "base_norm": {
            "mean": float(base_norms.mean()),
            "p50": float(np.percentile(base_norms, 50)),
            "p90": float(np.percentile(base_norms, 90)),
        },
        "delta_to_base_ratio": {
            "mean": float(ratios.mean()),
            "p50": float(np.percentile(ratios, 50)),
            "p90": float(np.percentile(ratios, 90)),
        },
    }


@torch.no_grad()
def compute_with_without_similarity(model, target_module, loader, device, target_pid: int, target_base_scale: float):
    """诊断2: WITH vs WITHOUT embedding差异"""
    model.eval()
    target_module.eval()
    
    cos_sims = []
    
    for x, pid, cam_label, view_label, _ in loader:
        x = x.to(device)
        cam_label = cam_label.to(device)
        view_label = view_label.to(device)
        pid_t = torch.as_tensor(pid, device=device)
        
        z_base = model(x, cam_label=cam_label, view_label=view_label)
        z_without = z_base.clone()
        
        delta = target_module(z_base)
        mask = (pid_t == int(target_pid)).float().unsqueeze(1)
        if target_base_scale != 0.0:
            z_with = z_base + delta * mask - target_base_scale * z_base * mask
        else:
            z_with = z_base + delta * mask
        
        target_mask = pid_t == int(target_pid)
        if torch.any(target_mask):
            z_with_n = F.normalize(z_with[target_mask], p=2, dim=1)
            z_without_n = F.normalize(z_without[target_mask], p=2, dim=1)
            cos = (z_with_n * z_without_n).sum(dim=1).cpu().numpy()
            cos_sims.extend(cos.tolist())
    
    if not cos_sims:
        return None
    
    cos_sims = np.array(cos_sims)
    return {
        "cos_sim_with_without": {
            "mean": float(cos_sims.mean()),
            "p50": float(np.percentile(cos_sims, 50)),
            "p90": float(np.percentile(cos_sims, 90)),
        },
    }


@torch.no_grad()
def compute_discriminator_metrics(model, discriminator, loader, device, target_pid: int):
    """诊断3: Base anti-target对抗判别器指标"""
    model.eval()
    discriminator.eval()
    
    all_logits = []
    all_labels = []
    
    for x, pid, cam_label, view_label, _ in loader:
        x = x.to(device)
        cam_label = cam_label.to(device)
        view_label = view_label.to(device)
        pid_t = torch.as_tensor(pid, device=device)
        
        z_base = model(x, cam_label=cam_label, view_label=view_label)
        logits = discriminator(z_base)
        labels = (pid_t == int(target_pid)).long()
        
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    probs = F.softmax(all_logits, dim=1)
    preds = probs.argmax(dim=1)
    acc = (preds == all_labels).float().mean()
    
    # AUC (simple version)
    pos_probs = probs[all_labels == 1, 1].numpy()
    neg_probs = probs[all_labels == 0, 1].numpy()
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        from sklearn.metrics import roc_auc_score  # type: ignore
        auc = roc_auc_score(all_labels.numpy(), probs[:, 1].numpy())
    else:
        auc = float("nan")
    
    return {
        "discriminator_acc": float(acc),
        "discriminator_auc": float(auc),
        "n_pos": int(all_labels.sum()),
        "n_neg": int((all_labels == 0).sum()),
    }


@torch.no_grad()
def compute_classification_margin(model, classifier, loader, device, target_pid: int, pid_to_label: dict):
    """诊断4: WITHOUT模式下ID2分类头logit margin"""
    model.eval()
    classifier.eval()
    
    target_label = pid_to_label.get(target_pid)
    if target_label is None:
        return None
    
    margins = []
    probs = []
    ranks = []
    
    for x, pid, cam_label, view_label, _ in loader:
        x = x.to(device)
        cam_label = cam_label.to(device)
        view_label = view_label.to(device)
        pid_t = torch.as_tensor(pid, device=device)
        
        z_base = model(x, cam_label=cam_label, view_label=view_label)
        logits = classifier(z_base)
        
        target_mask = pid_t == int(target_pid)
        if torch.any(target_mask):
            target_logits = logits[target_mask]
            target_probs = F.softmax(target_logits, dim=1)
            
            # Margin: logit[target] - max(logit[non-target])
            for i, tlog in enumerate(target_logits):
                tlog_np = tlog.cpu().numpy()
                margin = float(tlog_np[target_label] - tlog_np[tlog_np != tlog_np[target_label]].max())
                margins.append(margin)
                probs.append(float(target_probs[i, target_label].cpu()))
                
                # Rank of target label
                rank = int((tlog_np >= tlog_np[target_label]).sum())
                ranks.append(rank)
    
    if not margins:
        return None
    
    margins = np.array(margins)
    probs = np.array(probs)
    ranks = np.array(ranks)
    
    return {
        "logit_margin": {
            "mean": float(margins.mean()),
            "p50": float(np.percentile(margins, 50)),
            "p90": float(np.percentile(margins, 90)),
        },
        "target_prob": {
            "mean": float(probs.mean()),
            "p50": float(np.percentile(probs, 50)),
            "p90": float(np.percentile(probs, 90)),
        },
        "target_rank": {
            "mean": float(ranks.mean()),
            "p50": float(np.percentile(ranks, 50)),
            "p90": float(np.percentile(ranks, 90)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvp_dir", required=True, help="output/removable/<tag>")
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--forget_id", type=int, required=True)
    ap.add_argument("--target_base_scale", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--neck_feat", default="before", choices=["before", "after"])
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg.merge_from_file(args.cfg)
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.TEST.NECK_FEAT = args.neck_feat
    configure_data_root(cfg, repo_root)
    cfg.freeze()

    device = "cuda" if _cuda_works() else "cpu"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    mvp_dir = Path(args.mvp_dir)
    base_ckpt = mvp_dir / "base_ckpt.pth"
    target_module_ckpt = mvp_dir / "target_module.pth"
    run_args_path = mvp_dir / "run_args.json"

    if not base_ckpt.exists() or not target_module_ckpt.exists():
        raise RuntimeError(f"Missing checkpoints in {mvp_dir}")

    # Load run_args for pid_to_label
    run_args = json.loads(run_args_path.read_text()) if run_args_path.exists() else {}
    pid_to_label = {}
    label_to_pid = {}
    if "label_to_pid" in run_args:
        label_to_pid = {int(k): int(v) for k, v in run_args["label_to_pid"].items()}
        pid_to_label = {v: k for k, v in label_to_pid.items()}
    
    # If not found, try to build from dataset
    if not pid_to_label:
        print("[WARN] label_to_pid not found in run_args, building from dataset...", flush=True)
        from unlearning_reid.datasets.common import read_train_items
        data_root = Path(os.environ.get("REID_DATA_DIR", repo_root / "data"))
        items = read_train_items("market1501", data_root)
        pids = sorted({it.pid for it in items})
        pid_to_label = {pid: i for i, pid in enumerate(pids)}
        label_to_pid = {i: pid for pid, i in pid_to_label.items()}

    # Build model
    _, _, _, _num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    load_transreid_weights(model, str(base_ckpt))
    model.to(device)

    # Load target module
    tm_ckpt = torch.load(target_module_ckpt, map_location="cpu")
    tm_dim = int(tm_ckpt["dim"])
    tm_cfg = tm_ckpt.get("config", {}) if isinstance(tm_ckpt.get("config", {}), dict) else {}
    target_module = TargetModule(dim=tm_dim, hidden=tm_cfg.get("hidden"), dropout=tm_cfg.get("dropout", 0.0))
    target_module.load_state_dict(tm_ckpt["state_dict"], strict=True)
    target_module.to(device)

    # Build discriminator (for diagnostic 3)
    from unlearning_reid.models.removable import TargetDiscriminator

    discriminator = TargetDiscriminator(dim=tm_dim, hidden=256).to(device)
    # Try to load discriminator from checkpoint if exists
    disc_ckpt_path = mvp_dir / "discriminator.pth"
    disc_available = False
    if disc_ckpt_path.exists():
        try:
            discriminator.load_state_dict(torch.load(disc_ckpt_path, map_location="cpu"))
            disc_available = True
        except Exception as e:
            print(f"[WARN] Failed to load discriminator: {e}", flush=True)

    # Build classifier (for diagnostic 4)
    classifier = torch.nn.Linear(tm_dim, num_classes, bias=False).to(device)
    classifier_ckpt_path = mvp_dir / "classifier.pth"
    cls_available = False
    if classifier_ckpt_path.exists():
        try:
            classifier.load_state_dict(torch.load(classifier_ckpt_path, map_location="cpu"))
            cls_available = True
        except Exception as e:
            print(f"[WARN] Failed to load classifier: {e}", flush=True)
    
    if not cls_available:
        print("[WARN] Classifier checkpoint not found. Diagnostic 4 will be skipped.", flush=True)
    if not disc_available:
        print("[WARN] Discriminator checkpoint not found. Diagnostic 3 will be skipped.", flush=True)

    transform = build_val_transform(cfg)

    # Load forget query for diagnostics
    forget_query_jsonl = Path(args.probe_dir) / "forget_query.jsonl"
    if not forget_query_jsonl.exists():
        raise RuntimeError(f"Forget query not found: {forget_query_jsonl}")

    ds = JsonlImageDataset(forget_query_jsonl, transform)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    report = {
        "target_id": int(args.forget_id),
        "mvp_dir": str(mvp_dir),
        "probe_dir": str(args.probe_dir),
        "neck_feat": args.neck_feat,
        "target_base_scale": args.target_base_scale,
    }

    # 诊断1: Delta贡献幅度
    print("[DIAG] Computing delta stats...", flush=True)
    delta_stats = compute_delta_stats(model, target_module, loader, device, args.forget_id, args.target_base_scale)
    report["diagnostic_1_delta_contribution"] = delta_stats

    # 诊断2: WITH vs WITHOUT相似度
    print("[DIAG] Computing with/without similarity...", flush=True)
    sim_stats = compute_with_without_similarity(model, target_module, loader, device, args.forget_id, args.target_base_scale)
    report["diagnostic_2_with_without_similarity"] = sim_stats

    # 诊断3: 判别器指标
    if disc_available:
        print("[DIAG] Computing discriminator metrics...", flush=True)
        disc_stats = compute_discriminator_metrics(model, discriminator, loader, device, args.forget_id)
        report["diagnostic_3_discriminator"] = disc_stats
    else:
        report["diagnostic_3_discriminator"] = {
            "note": "Discriminator checkpoint not available. Re-run training with updated script to generate this diagnostic.",
        }

    # 诊断4: 分类头margin
    if pid_to_label and cls_available:
        print("[DIAG] Computing classification margin...", flush=True)
        cls_stats = compute_classification_margin(model, classifier, loader, device, args.forget_id, pid_to_label)
        report["diagnostic_4_classification_margin"] = cls_stats
    else:
        report["diagnostic_4_classification_margin"] = {
            "note": "Classifier checkpoint not available or pid_to_label missing. Re-run training with updated script to generate this diagnostic.",
        }

    # 诊断8: 特征来源
    report["diagnostic_8_feature_source"] = {
        "neck_feat": args.neck_feat,
        "jpm_enabled": cfg.MODEL.JPM if hasattr(cfg.MODEL, "JPM") else None,
        "transformer_type": cfg.MODEL.TRANSFORMER_TYPE if hasattr(cfg.MODEL, "TRANSFORMER_TYPE") else None,
    }

    # 诊断9: 评估协议
    probe_config_path = Path(args.probe_dir) / "probe_config.json"
    probe_config = {}
    if probe_config_path.exists():
        probe_config = json.loads(probe_config_path.read_text())
    report["diagnostic_9_eval_protocol"] = {
        "distractor_mode": probe_config.get("forget_distractor_mode", "unknown"),
        "distractor_ids": probe_config.get("forget_distractor_ids", "unknown"),
        "enforce_cross_cam": probe_config.get("enforce_cross_cam", "unknown"),
    }

    # 诊断10: WITHOUT模式确认
    report["diagnostic_10_without_mode_verification"] = {
        "target_module_loaded": True,
        "target_module_used_in_without": False,
        "note": "WITHOUT mode should not use target_module (gate=0 or module not loaded)",
    }

    # 读取训练日志以获取诊断5和6
    train_log_path = mvp_dir / "train_log.json"
    if train_log_path.exists():
        train_log = json.loads(train_log_path.read_text())
        # 诊断5: gate验证（从训练日志推断，实际需要在训练时记录）
        report["diagnostic_5_gate_verification"] = {
            "note": "Gate verification should be logged during training. Check train_log.json for batch-level stats.",
            "epochs_trained": len(train_log),
        }

        # 诊断6: 损失拆项（从训练日志提取）
        if train_log:
            last_epoch = train_log[-1]
            report["diagnostic_6_loss_breakdown"] = {
                "last_epoch": {
                    "loss_id": last_epoch.get("loss_id"),
                    "loss_tri": last_epoch.get("loss_tri"),
                    "loss_adv": last_epoch.get("loss_adv"),
                    "loss_off": last_epoch.get("loss_off"),
                    "loss_pull": last_epoch.get("loss_pull"),
                    "loss_off_norm": last_epoch.get("loss_off_norm"),
                    "loss_delta": last_epoch.get("loss_delta"),
                    "loss_distill": last_epoch.get("loss_distill"),
                    "loss_confuse": last_epoch.get("loss_confuse"),
                    "loss_spread": last_epoch.get("loss_spread"),
                },
                "note": "These are epoch-averaged losses. For ID2 vs non-ID2 breakdown, check training script logs.",
            }

        # 诊断7: ID2曝光率（需要从训练时记录）
        report["diagnostic_7_target_exposure"] = {
            "note": "Target exposure stats should be logged during training. Check run_args.json for sampling config.",
            "target_pid": int(args.forget_id),
            "sampler_type": "TargetPKSampler (ensures target in every batch)",
        }

    out_path = mvp_dir / "diagnostic_report_removable.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[OK] Diagnostic report -> {out_path}")


if __name__ == "__main__":
    main()
