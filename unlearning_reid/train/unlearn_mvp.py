from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from unlearning_reid.models.scrubber import ForgetDiscriminator, LinearScrubber, MembershipDiscriminator, ResidualScrubber


class FeatDataset(Dataset):
    def __init__(
        self,
        npz_path: Path,
        id_set: set[int],
        pid_to_index: dict[int, int] | None = None,
        return_index: bool = False,
    ):
        d = np.load(npz_path, allow_pickle=True)
        feats = d["feats"].astype(np.float32)
        pids = d["pids"].astype(np.int64)
        paths = d["paths"] if "paths" in d.files else None
        mask = np.array([pid in id_set for pid in pids], dtype=bool)
        self.feats = feats[mask]
        self.pids = pids[mask]
        self.paths = paths[mask] if paths is not None else None
        self.indices = np.where(mask)[0]
        self.return_index = return_index
        if pid_to_index is not None:
            self.labels = np.array([pid_to_index[int(p)] for p in self.pids], dtype=np.int64)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.feats[idx])
        pid = int(self.pids[idx])
        if self.labels is None:
            if self.return_index:
                return x, pid, int(self.indices[idx])
            return x, pid
        if self.return_index:
            return x, int(self.labels[idx]), pid, int(self.indices[idx])
        return x, int(self.labels[idx]), pid


def save_env_snapshot(out_dir: Path) -> None:
    import subprocess
    import sys

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "python.txt").write_text(sys.version + "\n")
    try:
        pip = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (out_dir / "pip_freeze.txt").write_text(pip)
    except Exception:
        pass


def _ensure_nonempty(name: str, ids: Iterable[int]) -> None:
    if not list(ids):
        raise RuntimeError(f"{name} is empty; cannot train unlearning MVP.")


def train_mvp(
    feat_train_npz: Path,
    retain_ids: list[int],
    forget_ids: list[int],
    out_dir: Path,
    seed: int = 0,
    bottleneck: int | None = 256,
    batch: int = 512,
    lr: float = 3e-4,
    steps: int = 4000,
    epochs: int | None = None,
    scrubber_type: str = "linear",
    lambda_distill: float = 1.0,
    lambda_forget: float = 1.0,
    lambda_collapse: float = 0.1,
    lambda_moment: float = 0.0,
    lambda_id: float = 0.0,
    alpha_max: float = 0.1,
    lambda_centroid: float = 0.0,
    lambda_delta: float = 0.0,
    forget_target: str = "none",
    lambda_impostor: float = 0.0,
    impostor_topk: int = 200,
    neighbor_attract: bool = False,
    lambda_nbr: float = 0.0,
    nbr_k: int = 20,
    nbr_sample: str = "random_from_topK",
    neighbor_cache_dir: Path | None = None,
    disc_steps_per_iter: int = 1,
    forget_mode: str = "disc",
    forget_balance: str = "proportional",
    membership_adv: bool = False,
    lambda_mem: float = 0.5,
    mem_hidden: int = 512,
    mem_steps_per_iter: int = 1,
    mem_adv_mode: str = "uniform_forget",
    mem_warmup_frac: float = 0.2,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_env_snapshot(out_dir)

    _ensure_nonempty("retain_ids", retain_ids)
    _ensure_nonempty("forget_ids", forget_ids)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}", flush=True)

    forget_ids = sorted(forget_ids)
    pid_to_idx = {pid: i for i, pid in enumerate(forget_ids)}

    retain_set = set(retain_ids)
    forget_set = set(forget_ids)

    ds_retain = FeatDataset(feat_train_npz, retain_set)
    ds_forget = FeatDataset(
        feat_train_npz,
        forget_set,
        pid_to_idx,
        return_index=neighbor_attract or (forget_target == "impostor_hard"),
    )

    if len(ds_retain) == 0 or len(ds_forget) == 0:
        raise RuntimeError("No features found for retain/forget sets. Check split and train features.")

    dim = ds_retain[0][0].numel()
    n_retain = len(ds_retain)
    n_forget = len(ds_forget)
    if n_forget == 1 and lambda_collapse > 0:
        print("[WARN] single-ID forget: forcing lambda_collapse=0.0")
        lambda_collapse = 0.0
    print("[INFO] retain_samples", n_retain, "forget_samples", n_forget, "batch", batch)

    retain_centroids = None
    forget_topk = None
    if neighbor_attract:
        if neighbor_cache_dir is None:
            neighbor_cache_dir = out_dir / "id_map"
        centroids_path = Path(neighbor_cache_dir) / "retain_centroids.npz"
        topk_path = Path(neighbor_cache_dir) / "forget_to_retain_topk.json"
        if not centroids_path.exists() or not topk_path.exists():
            raise RuntimeError(
                f"Neighbor cache not found under {neighbor_cache_dir}. "
                "Run scripts/build_neighbor_cache.py first."
            )
        d = np.load(centroids_path, allow_pickle=True)
        retain_centroids = torch.from_numpy(d["centroids"].astype(np.float32)).to(device)
        retain_pid_to_idx = {int(pid): i for i, pid in enumerate(d["pids"].astype(np.int64).tolist())}
        records = json.loads(topk_path.read_text())
        forget_topk = {}
        for rec in records:
            gidx = int(rec["index"])
            topk_pids = rec["topk_pids"]
            topk_idx = [retain_pid_to_idx[int(pid)] for pid in topk_pids if int(pid) in retain_pid_to_idx]
            if len(topk_idx) == 0:
                continue
            forget_topk[gidx] = topk_idx[: max(1, min(len(topk_idx), nbr_k))]
        if not forget_topk:
            raise RuntimeError("Neighbor cache loaded but forget_topk is empty.")
        if nbr_sample not in ("hard", "random_from_topK"):
            raise ValueError(f"Unknown nbr_sample: {nbr_sample}")

    if forget_target not in ("none", "impostor_random", "impostor_hard"):
        raise ValueError(f"Unknown forget_target: {forget_target}")

    retain_pool = None
    retain_pool_norm = None
    forget_to_retain_topk = None
    if forget_target in ("impostor_random", "impostor_hard") and lambda_impostor > 0.0:
        retain_pool = torch.from_numpy(ds_retain.feats).to(device)
        retain_pool_norm = F.normalize(retain_pool, dim=1)
        if forget_target == "impostor_hard":
            if impostor_topk <= 0:
                raise ValueError("impostor_topk must be > 0 for impostor_hard")
            k = min(int(impostor_topk), int(retain_pool_norm.shape[0]))
            f_feats = torch.from_numpy(ds_forget.feats).to(device)
            f_norm = F.normalize(f_feats, dim=1)
            sim = torch.matmul(f_norm, retain_pool_norm.t())
            topk = torch.topk(sim, k=k, dim=1).indices.detach().cpu().tolist()
            forget_to_retain_topk = {
                int(gidx): topk_row for gidx, topk_row in zip(ds_forget.indices.tolist(), topk)
            }

    if scrubber_type not in ("linear", "residual"):
        raise ValueError(f"Unknown scrubber_type: {scrubber_type}")
    if scrubber_type == "residual":
        scrubber = ResidualScrubber(dim=dim, bottleneck=bottleneck, alpha_max=alpha_max).to(device)
    else:
        scrubber = LinearScrubber(dim=dim, bottleneck=bottleneck).to(device)
    disc = None
    opt_d = None
    if forget_mode == "disc":
        if len(forget_ids) < 2:
            print("[WARN] forget_mode=disc with single ID; switching to negrad_cos")
            forget_mode = "negrad_cos"
        else:
            disc = ForgetDiscriminator(dim=dim, n_forget_ids=len(forget_ids)).to(device)
            opt_d = torch.optim.AdamW(disc.parameters(), lr=lr, weight_decay=1e-4)

    opt_s = torch.optim.AdamW(scrubber.parameters(), lr=lr, weight_decay=1e-4)
    mem_disc = None
    opt_m = None
    if membership_adv:
        mem_disc = MembershipDiscriminator(dim=dim, hidden=mem_hidden).to(device)
        opt_m = torch.optim.AdamW(mem_disc.parameters(), lr=lr, weight_decay=1e-4)

    drop_last_r = n_retain >= batch
    drop_last_f = n_forget >= batch
    loader_r = DataLoader(
        ds_retain,
        batch_size=batch,
        shuffle=True,
        num_workers=2,
        drop_last=drop_last_r,
        pin_memory=True,
    )
    sampler_f = None
    shuffle_f = True
    if forget_balance == "uniform":
        counts = np.bincount(ds_forget.labels)
        weights = 1.0 / np.maximum(counts, 1)
        sample_weights = torch.tensor([weights[i] for i in ds_forget.labels], dtype=torch.float32)
        sampler_f = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(ds_forget), replacement=True
        )
        shuffle_f = False
    loader_f = DataLoader(
        ds_forget,
        batch_size=batch,
        shuffle=shuffle_f if sampler_f is None else False,
        sampler=sampler_f,
        num_workers=2,
        drop_last=drop_last_f,
        pin_memory=True,
    )

    def _unpack_retain(batch):
        if len(batch) == 2:
            return batch[0], batch[1], None
        return batch[0], batch[1], batch[2]

    def _unpack_forget(batch):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2], None
        return batch[0], batch[1], batch[2], batch[3]
    steps_per_epoch = max(len(loader_r), 1)
    if epochs is None:
        epochs = int(np.ceil(steps / steps_per_epoch))
    total_steps = epochs * steps_per_epoch
    print(f"[INFO] epochs={epochs} steps_per_epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    logs = []
    loss_d = torch.tensor(0.0)
    loss_mem_d = torch.tensor(0.0)
    global_step = 0
    for epoch_idx in range(1, epochs + 1):
        it_f = iter(loader_f)
        for batch_idx, batch_r in enumerate(loader_r, start=1):
            global_step += 1
            step = global_step
            scrubber.train()
            if disc is not None:
                disc.train()
            if mem_disc is not None:
                mem_disc.train()

            z_t_r, _pid_r, _idx_r = _unpack_retain(batch_r)
            z_t_r = z_t_r.to(device)

            if forget_mode == "disc":
                for _ in range(disc_steps_per_iter):
                    try:
                        z_t_f, y_f, _pid, _idx_f = _unpack_forget(next(it_f))
                    except StopIteration:
                        it_f = iter(loader_f)
                        z_t_f, y_f, _pid, _idx_f = _unpack_forget(next(it_f))

                    z_t_f = z_t_f.to(device)
                    y_f = y_f.to(device)

                    with torch.no_grad():
                        z_s_f = scrubber(z_t_f)
                    logits = disc(z_s_f.detach())
                    loss_d = F.cross_entropy(logits, y_f)

                    opt_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    opt_d.step()

            if mem_disc is not None:
                for _ in range(mem_steps_per_iter):
                    try:
                        z_t_f, y_f, _pid, _idx_f = _unpack_forget(next(it_f))
                    except StopIteration:
                        it_f = iter(loader_f)
                        z_t_f, y_f, _pid, _idx_f = _unpack_forget(next(it_f))

                    z_t_f = z_t_f.to(device)

                    with torch.no_grad():
                        z_s_r = scrubber(z_t_r)
                        z_s_f = scrubber(z_t_f)
                    z_s_r = F.normalize(z_s_r, dim=1)
                    z_s_f = F.normalize(z_s_f, dim=1)
                    z_s = torch.cat([z_s_r, z_s_f], dim=0)
                    labels = torch.cat(
                        [
                            torch.zeros(z_s_r.shape[0], dtype=torch.long),
                            torch.ones(z_s_f.shape[0], dtype=torch.long),
                        ],
                        dim=0,
                    ).to(device)
                    logits = mem_disc(z_s.detach())
                    loss_mem_d = F.cross_entropy(logits, labels)
                    opt_m.zero_grad(set_to_none=True)
                    loss_mem_d.backward()
                    opt_m.step()

            try:
                z_t_f, y_f, _pid, idx_f = _unpack_forget(next(it_f))
            except StopIteration:
                it_f = iter(loader_f)
                z_t_f, y_f, _pid, idx_f = _unpack_forget(next(it_f))

            z_t_f = z_t_f.to(device)
            y_f = y_f.to(device)

            z_s_r = scrubber(z_t_r)
            z_s_f = scrubber(z_t_f)

            delta_r = z_s_r - z_t_r
            delta_r_norms = delta_r.norm(dim=1)
            loss_distill = F.mse_loss(z_s_r, z_t_r)
            # identity regularization: only lock retain to teacher (avoid over-constraining forget)
            loss_id = F.mse_loss(z_s_r, z_t_r)
            # explicit delta suppression on retain
            loss_delta_retain = (delta_r_norms**2).mean()
        if forget_mode == "disc":
            logits_f = disc(z_s_f)
            loss_forget_adv = -F.cross_entropy(logits_f, y_f)
        elif forget_mode == "negrad_cos":
            cos = F.cosine_similarity(z_s_f, z_t_f, dim=1)
            loss_forget_adv = cos.mean()
        elif forget_mode == "spread":
            zf = F.normalize(z_s_f, dim=1)
            sim = zf @ zf.t()
            if sim.shape[0] > 1:
                off = (sim.sum() - sim.diag().sum()) / (sim.numel() - sim.shape[0])
                loss_forget_adv = off
            else:
                loss_forget_adv = torch.tensor(0.0, device=device)
        else:
            raise ValueError(f"Unknown forget_mode: {forget_mode}")

        mean_f = z_s_f.mean(dim=0, keepdim=True)
        loss_collapse = F.mse_loss(z_s_f, mean_f.expand_as(z_s_f))

        # moment matching (mean + CORAL)
        mean_r = z_s_r.mean(dim=0, keepdim=True)
        xc_r = z_s_r - mean_r
        xc_f = z_s_f - mean_f
        cov_r = xc_r.t().mm(xc_r) / max(z_s_r.shape[0] - 1, 1)
        cov_f = xc_f.t().mm(xc_f) / max(z_s_f.shape[0] - 1, 1)
        loss_mean = F.mse_loss(mean_r, mean_f)
        loss_coral = F.mse_loss(cov_r, cov_f)
        loss_moment = loss_mean + loss_coral
        mean_diff_norm = torch.norm(mean_r - mean_f, p=2)
        cov_diff_norm = torch.norm(cov_r - cov_f, p="fro")

        loss_impostor = torch.tensor(0.0, device=device)
        if retain_pool is not None and lambda_impostor > 0.0:
            zf = F.normalize(z_s_f, dim=1)
            if forget_target == "impostor_random":
                ridx = torch.randint(0, retain_pool.shape[0], (zf.shape[0],), device=device)
                tgt = retain_pool[ridx]
            elif forget_target == "impostor_hard":
                if idx_f is None:
                    raise RuntimeError("forget_target=impostor_hard requires forget batch indices.")
                topk_list = []
                for gidx in idx_f.tolist():
                    topk = forget_to_retain_topk.get(int(gidx)) if forget_to_retain_topk else None
                    if not topk:
                        raise RuntimeError(f"Missing impostor topK for forget index {gidx}")
                    topk_list.append(topk)
                topk_idx = torch.tensor(topk_list, device=device, dtype=torch.long)
                choice = torch.randint(0, topk_idx.shape[1], (topk_idx.shape[0],), device=device)
                ridx = topk_idx[torch.arange(topk_idx.shape[0], device=device), choice]
                tgt = retain_pool[ridx]
            else:
                raise RuntimeError(f"Unexpected forget_target: {forget_target}")
            tgt = F.normalize(tgt.detach(), dim=1)
            loss_impostor = ((zf - tgt) ** 2).sum(dim=1).mean()

        loss_centroid = torch.tensor(0.0, device=device)
        centroid_dist = None
        if lambda_centroid > 0.0:
            zf_centroid = F.normalize(z_s_f.mean(dim=0, keepdim=True), dim=1)
            pid_r = _pid_r if isinstance(_pid_r, torch.Tensor) else torch.tensor(_pid_r)
            pid_r = pid_r.to(device)
            uniq = torch.unique(pid_r)
            retain_centroids = []
            for pid in uniq.tolist():
                mask = pid_r == pid
                if mask.any():
                    retain_centroids.append(z_s_r[mask].mean(dim=0))
            if retain_centroids:
                rc = F.normalize(torch.stack(retain_centroids, dim=0), dim=1)
                sim = torch.matmul(zf_centroid, rc.t())
                max_sim = sim.max()
                loss_centroid = 1.0 - max_sim
                centroid_dist = float(1.0 - max_sim.detach().cpu())

        loss_nbr = torch.tensor(0.0, device=device)
        if neighbor_attract and lambda_nbr > 0.0:
            if idx_f is None:
                raise RuntimeError("neighbor_attract requires forget batch indices.")
            zf = F.normalize(z_s_f, dim=1)
            topk_list = []
            for gidx in idx_f.tolist():
                topk = forget_topk.get(int(gidx))
                if not topk:
                    raise RuntimeError(f"Missing topK for forget index {gidx}")
                if len(topk) < nbr_k:
                    topk = topk + [topk[0]] * (nbr_k - len(topk))
                topk_list.append(topk[:nbr_k])
            topk_idx = torch.tensor(topk_list, device=device, dtype=torch.long)
            centroids = retain_centroids[topk_idx]
            if nbr_sample == "random_from_topK":
                choice = torch.randint(0, centroids.shape[1], (centroids.shape[0],), device=device)
                c = centroids[torch.arange(centroids.shape[0], device=device), choice]
            else:
                sim = torch.einsum("bd,bkd->bk", zf, centroids)
                c = centroids[torch.arange(centroids.shape[0], device=device), sim.argmax(dim=1)]
            c = F.normalize(c, dim=1)
            loss_nbr = (1.0 - (zf * c).sum(dim=1)).mean()

        loss_mem_adv = torch.tensor(0.0, device=device)
        if mem_disc is not None:
            if mem_adv_mode == "grl":
                z_m_r = F.normalize(z_s_r, dim=1)
                z_m_f = F.normalize(z_s_f, dim=1)
                z_m = torch.cat([z_m_r, z_m_f], dim=0)
                labels = torch.cat(
                    [
                        torch.zeros(z_m_r.shape[0], dtype=torch.long),
                        torch.ones(z_m_f.shape[0], dtype=torch.long),
                    ],
                    dim=0,
                ).to(device)
                logits_m = mem_disc(z_m)
                loss_mem_adv = -F.cross_entropy(logits_m, labels)
            elif mem_adv_mode == "uniform_all":
                z_m_r = F.normalize(z_s_r, dim=1)
                z_m_f = F.normalize(z_s_f, dim=1)
                z_m = torch.cat([z_m_r, z_m_f], dim=0)
                logits_m = mem_disc(z_m)
                probs = F.softmax(logits_m, dim=1)
                uniform = torch.full_like(probs, 0.5)
                loss_mem_adv = F.kl_div(probs.log(), uniform, reduction="batchmean")
            else:
                # uniform_forget: encourage forget samples to be uncertain
                z_m_f = F.normalize(z_s_f, dim=1)
                logits_m = mem_disc(z_m_f)
                probs = F.softmax(logits_m, dim=1)
                uniform = torch.full_like(probs, 0.5)
                loss_mem_adv = F.kl_div(probs.log(), uniform, reduction="batchmean")

        warmup_steps = int(total_steps * mem_warmup_frac)
        if step <= warmup_steps:
            lambda_mem_eff = 0.0
            lambda_moment_eff = 0.0
        else:
            ramp = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            lambda_mem_eff = float(lambda_mem) * ramp
            lambda_moment_eff = float(lambda_moment) * ramp

        loss_s = (
            lambda_distill * loss_distill
            + lambda_forget * loss_forget_adv
            + lambda_collapse * loss_collapse
            + lambda_moment_eff * loss_moment
            + lambda_id * loss_id
            + lambda_centroid * loss_centroid
            + lambda_delta * loss_delta_retain
            + lambda_impostor * loss_impostor
            + lambda_nbr * loss_nbr
            + (lambda_mem_eff * loss_mem_adv if mem_disc is not None else 0.0)
        )

        opt_s.zero_grad(set_to_none=True)
        loss_s.backward()
        opt_s.step()

        if step % 50 == 0 or step == 1:
            epoch_retain = step * batch / max(n_retain, 1)
            epoch_forget = step * batch / max(n_forget, 1)
            epoch = epoch_idx - 1 + (batch_idx / max(steps_per_epoch, 1))
            norms = z_s_r.norm(dim=1)
            try:
                p90 = torch.quantile(norms, 0.9).item()
            except Exception:
                p90 = float(norms.kthvalue(max(1, int(0.9 * len(norms)))).values.item())
            rec = {
                "step": step,
                "forget_mode": forget_mode,
                "epoch": float(epoch),
                "epoch_idx": int(epoch_idx),
                "epoch_retain": float(epoch_retain),
                "epoch_forget": float(epoch_forget),
                "loss_distill": float(loss_distill.detach().cpu()),
                "loss_forget_adv": float(loss_forget_adv.detach().cpu()),
                "loss_collapse": float(loss_collapse.detach().cpu()),
                "loss_id": float(loss_id.detach().cpu()),
                "loss_delta_retain": float(loss_delta_retain.detach().cpu()),
                "loss_impostor": float(loss_impostor.detach().cpu()),
                "loss_d": float(loss_d.detach().cpu()),
                "loss_mem_d": float(loss_mem_d.detach().cpu()),
                "loss_mem_adv": float(loss_mem_adv.detach().cpu()),
                "loss_moment": float(loss_moment.detach().cpu()),
                "loss_nbr": float(loss_nbr.detach().cpu()),
                "loss_centroid": float(loss_centroid.detach().cpu()),
                "mean_diff_norm": float(mean_diff_norm.detach().cpu()),
                "cov_diff_norm": float(cov_diff_norm.detach().cpu()),
                "lambda_mem_eff": float(lambda_mem_eff),
                "lambda_moment_eff": float(lambda_moment_eff),
                "retain_norm_mean": float(norms.mean().item()),
                "retain_norm_p90": float(p90),
                "retain_delta_norm_mean": float(delta_r_norms.mean().item()),
                "retain_delta_norm_p90": float(torch.quantile(delta_r_norms, 0.9).item())
                if delta_r_norms.numel() > 1
                else float(delta_r_norms.mean().item()),
                "alpha": scrubber.alpha_value() if hasattr(scrubber, "alpha_value") else None,
            }
            if centroid_dist is not None:
                rec["centroid_dist"] = centroid_dist
            logs.append(rec)
            print("[LOG]", rec, flush=True)

        if step % 500 == 0:
            ckpt = {
                "scrubber": scrubber.state_dict(),
                "disc": disc.state_dict() if disc is not None else None,
                "dim": dim,
                "forget_ids": forget_ids,
                "seed": seed,
                "config": {
                    "bottleneck": bottleneck,
                    "batch": batch,
                    "lr": lr,
                    "steps": total_steps,
                    "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "scrubber_type": scrubber_type,
                    "lambda_distill": lambda_distill,
                    "lambda_forget": lambda_forget,
                    "lambda_collapse": lambda_collapse,
                    "lambda_moment": lambda_moment,
                    "lambda_id": lambda_id,
                    "alpha_max": alpha_max,
                    "lambda_centroid": lambda_centroid,
                    "lambda_delta": lambda_delta,
                    "forget_target": forget_target,
                    "lambda_impostor": lambda_impostor,
                    "impostor_topk": impostor_topk,
                    "neighbor_attract": neighbor_attract,
                    "lambda_nbr": lambda_nbr,
                    "nbr_k": nbr_k,
                    "nbr_sample": nbr_sample,
                    "neighbor_cache_dir": str(neighbor_cache_dir) if neighbor_cache_dir else None,
                    "forget_balance": forget_balance,
                    "membership_adv": membership_adv,
                    "lambda_mem": lambda_mem,
                    "mem_hidden": mem_hidden,
                    "mem_steps_per_iter": mem_steps_per_iter,
                    "mem_adv_mode": mem_adv_mode,
                    "mem_warmup_frac": mem_warmup_frac,
                },
            }
            torch.save(ckpt, out_dir / f"ckpt_step{step}.pt")


    torch.save(
        {
            "scrubber": scrubber.state_dict(),
            "disc": disc.state_dict() if disc is not None else None,
            "dim": dim,
            "forget_ids": forget_ids,
            "seed": seed,
            "config": {
                "bottleneck": bottleneck,
                "batch": batch,
                "lr": lr,
                "steps": total_steps,
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "scrubber_type": scrubber_type,
                "lambda_distill": lambda_distill,
                "lambda_forget": lambda_forget,
                "lambda_collapse": lambda_collapse,
                "lambda_moment": lambda_moment,
                "lambda_id": lambda_id,
                "alpha_max": alpha_max,
                "lambda_centroid": lambda_centroid,
                "lambda_delta": lambda_delta,
                "forget_target": forget_target,
                "lambda_impostor": lambda_impostor,
                "impostor_topk": impostor_topk,
                "neighbor_attract": neighbor_attract,
                "lambda_nbr": lambda_nbr,
                "nbr_k": nbr_k,
                "nbr_sample": nbr_sample,
                "neighbor_cache_dir": str(neighbor_cache_dir) if neighbor_cache_dir else None,
                "forget_balance": forget_balance,
                "membership_adv": membership_adv,
                "lambda_mem": lambda_mem,
                "mem_hidden": mem_hidden,
                "mem_steps_per_iter": mem_steps_per_iter,
                "mem_adv_mode": mem_adv_mode,
                "mem_warmup_frac": mem_warmup_frac,
            },
        },
        out_dir / "ckpt_final.pt",
    )
    (out_dir / "train_log.json").write_text(json.dumps(logs, indent=2) + "\n")


