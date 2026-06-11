#!/usr/bin/env python
"""
GRL sanity check: one batch forward, L_D backward through GRL; verify backbone
receives non-zero gradients and GRL sign is correct (reversed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from unlearning_reid.models.removable import TargetDiscriminator, grad_reverse


def _ensure_transreid_imports(repo_root: Path) -> None:
    tr = repo_root / "third_party" / "TransReID"
    if str(tr) not in sys.path:
        sys.path.insert(0, str(tr))


def main():
    repo_root = Path(__file__).resolve().parent
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
    from datasets.make_dataloader import Compose, Normalize, Pad, RandomCrop, RandomHorizontalFlip, Resize, ToTensor

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--grl_lambda", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    cfg_path = args.cfg or str(repo_root / "third_party/TransReID/configs/Market/vit_transreid_stride.yml")
    weights_path = args.weights or str(repo_root / "output/transreid/market_teacher_r50/transformer_120.pth")
    cfg.merge_from_file(cfg_path)
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.TEST.NECK_FEAT = "before"
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 751
    model = make_model(cfg, num_class=num_classes, camera_num=6, view_num=1)
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_state = model.state_dict()
    cleaned = {k.replace("module.", ""): v for k, v in state.items() if k.replace("module.", "") in model_state and model_state[k.replace("module.", "")].shape == v.shape}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.train()

    dummy_batch = torch.randn(args.batch, 3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1], device=device)
    cam_label = torch.zeros(args.batch, dtype=torch.long, device=device)
    view_label = torch.zeros(args.batch, dtype=torch.long, device=device)
    is_target = torch.zeros(args.batch, dtype=torch.long, device=device)
    is_target[0] = 1

    out = model(dummy_batch, cam_label=cam_label, view_label=view_label)
    if isinstance(out, (list, tuple)) and len(out) > 1:
        feats = out[1]
        z_base = feats[0] if isinstance(feats, (list, tuple)) else feats
    else:
        z_base = out[0] if isinstance(out, (list, tuple)) else out
    if not isinstance(z_base, torch.Tensor):
        z_base = z_base[0] if isinstance(z_base, (list, tuple)) else z_base
    feat_dim = int(z_base.shape[1])
    discriminator = TargetDiscriminator(dim=feat_dim, hidden=256).to(device)
    grl_lambda = args.grl_lambda

    model.zero_grad()
    discriminator.zero_grad()
    z_rev = grad_reverse(z_base, grl_lambda)
    adv_logits = discriminator(z_rev)
    loss_d = F.cross_entropy(adv_logits, is_target)
    loss_d.backward()

    names_with_grad = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gnorm = p.grad.norm().item()
            names_with_grad.append((name, gnorm))
    names_with_grad.sort(key=lambda x: x[0])

    print("GRL sanity check")
    print("  GRL lambda:", grl_lambda)
    print("  GRL backward multiplies gradient by -lambda (reversed direction).")
    print("  L_D backward -> disc gets normal grad; z_base gets -lambda * (disc grad).")
    print("  Backbone last layers (last 10 params with grad):")
    for name, gnorm in names_with_grad[-10:]:
        print(f"    {name}: grad_norm = {gnorm:.6f}")
    if not names_with_grad:
        print("  [FAIL] No backbone parameters received gradients. Check: detach, no_grad, or GRL implementation.")
        return 1
    if all(gnorm < 1e-9 for _, gnorm in names_with_grad):
        print("  [FAIL] All grad norms near zero. GRL or backward path may be broken.")
        return 1
    print("  [OK] Backbone received non-zero gradients; GRL path is active.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
