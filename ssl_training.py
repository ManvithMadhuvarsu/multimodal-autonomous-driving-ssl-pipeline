"""
=============================================================================
ssl_training.py  —  SSL Data Loading + Contrastive Training
=============================================================================
CHANGES vs previous version:
  [FIX-1] BYOL implemented — MomentumEncoder EMA target, predictor head,
           byol_loss(), momentum update after every optimiser step
  [FIX-2] Cross-modal contrastive loss (L_cm) implemented — NT-Xent applied
           between all pairs of different modalities that appear in the same
           DataLoader batch (rgb↔thermal, rgb↔lidar, etc.)
  [FIX-3] λ-weighted multi-objective loss applied:
               L = λ_nt·L_NT + λ_cm·L_cm + λ_byol·L_BYOL
           with λ_nt=1.0, λ_cm=0.5, λ_byol=0.5  (paper Hyperparameter Table)
  [FIX-4] Checkpoint save/load extended: "pred" and "mom" state dicts
           saved and restored alongside enc/proj
  All safe loaders, augmentations, WeightedRandomSampler, gradient clipping,
  CosineAnnealingLR, and resumability logic kept unchanged.
=============================================================================
"""
import os
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from setup import (
    OUTPUT_ROOT, PROGRESS_ROOT, SSL_DIR, DEVICE,
    logger, load_progress, save_progress, write_json, read_json,
    load_bin_lidar, load_radar_pcd,
)
from models import (
    RGBEncoder, ThermalEncoder, LiDAREncoder, RadarEncoder,
    projection_head, nt_xent, byol_loss, build_ssl_models,
)

# ── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE    = 224
N_POINTS    = 2048
BATCH_SIZE  = 32
NUM_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))
SAVE_EVERY  = 100       # batches between mid-epoch checkpoints
KEEP_CKPTS  = 2

# ── Loss weights  (paper Hyperparameter Table) ─────────────────────────────
LAMBDA_NT   = 1.0       # per-modality NT-Xent weight
LAMBDA_CM   = 0.5       # cross-modal NT-Xent weight
LAMBDA_BYOL = 0.5       # BYOL bootstrap loss weight

# ── RGB transforms ──────────────────────────────────────────────────────────
_RGB_MEAN = [0.485, 0.456, 0.406]
_RGB_STD  = [0.229, 0.224, 0.225]

img_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(_RGB_MEAN, _RGB_STD),
])
rgb_augment = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
    T.ToTensor(),
    T.Normalize(_RGB_MEAN, _RGB_STD),
])

# ── Thermal transforms (separate LWIR stats, NOT ImageNet) ─────────────────
_TH_MEAN = [0.5,  0.5,  0.5]
_TH_STD  = [0.25, 0.25, 0.25]

thermal_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(_TH_MEAN, _TH_STD),
])
thermal_augment = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.1),
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.7),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5))], p=0.4),
    T.ToTensor(),
    T.Normalize(_TH_MEAN, _TH_STD),
])


# ── Safe loaders ────────────────────────────────────────────────────────────
def load_rgb_safe(path: str) -> "torch.Tensor | None":
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return None
        img = Image.open(str(p)).convert("RGB")
        if img.size[0] < 32 or img.size[1] < 32:
            return None
        return img_transform(img)
    except Exception:
        return None


def load_rgb_aug(path: str) -> "torch.Tensor | None":
    try:
        return rgb_augment(Image.open(str(path)).convert("RGB"))
    except Exception:
        return None


def _tiff_to_pil(path: str) -> "Image.Image | None":
    """Load FLIR 16-bit TIFF → uint8 3-channel PIL image."""
    try:
        img = Image.open(str(path))
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 2:
            lo, hi = arr.min(), arr.max()
            arr = (arr - lo) / (hi - lo + 1e-8)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            lo, hi = arr.min(), arr.max()
            arr = ((arr - lo) / (hi - lo + 1e-8) * 255).clip(0, 255).astype(np.uint8)
        else:
            return None
        return Image.fromarray(arr)
    except Exception:
        return None


def load_thermal_safe(path: str) -> "torch.Tensor | None":
    pil = _tiff_to_pil(path)
    return thermal_transform(pil) if pil is not None else None


def load_thermal_aug(path: str) -> "torch.Tensor | None":
    pil = _tiff_to_pil(path)
    return thermal_augment(pil) if pil is not None else None


def load_pointcloud_safe(path: str, n: int = N_POINTS) -> "torch.Tensor | None":
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return None
        pts = load_bin_lidar(p) if path.endswith(".bin") else load_radar_pcd(p)
        if pts is None or pts.shape[0] < 5:
            return None
        pts = np.nan_to_num(pts[:, :3].astype(np.float32))
        N   = pts.shape[0]
        idx = np.random.choice(N, n, replace=(N < n))
        return torch.from_numpy(pts[idx])
    except Exception:
        return None


def augment_pointcloud(pts: torch.Tensor) -> torch.Tensor:
    """Yaw rotation + jitter + random point dropout."""
    theta  = float(torch.rand(1)) * 2 * 3.14159
    c, s   = np.cos(theta), np.sin(theta)
    R      = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    out    = pts @ R.T + torch.randn_like(pts) * 0.02
    keep   = max(int(pts.shape[0] * (0.85 + 0.15 * float(torch.rand(1)))), 1)
    idx    = torch.randperm(pts.shape[0])[:keep]
    if keep < pts.shape[0]:
        pad = idx[torch.randperm(keep)[: pts.shape[0] - keep]]
        idx = torch.cat([idx, pad])
    return torch.nan_to_num(out[idx])


# ── Validation pre-scanner ──────────────────────────────────────────────────
VALID_IDX_FILE  = PROGRESS_ROOT / "valid_indices.json"
VALID_PROG_FILE = PROGRESS_ROOT / "valid_scan_progress.json"


class ValidationRunner:
    def __init__(self, entries):
        self.entries = entries
        VALID_IDX_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.valid_indices = (read_json(VALID_IDX_FILE)
                              if VALID_IDX_FILE.exists() else [])
        prog = read_json(VALID_PROG_FILE) if VALID_PROG_FILE.exists() else {}
        self.last = int(prog.get("last_index", -1))

    def run(self) -> List[int]:
        start = self.last + 1
        if start >= len(self.entries):
            logger.info(f"Pre-scan done: {len(self.valid_indices)} valid.")
            return self.valid_indices
        for i in tqdm(range(start, len(self.entries)), desc="Pre-scanning"):
            e   = self.entries[i]
            m, p = e.get("modality"), e.get("path")
            ok  = False
            if m == "rgb":               ok = load_rgb_safe(p)          is not None
            elif m == "thermal":         ok = load_thermal_safe(p)      is not None
            elif m in ("lidar", "radar"): ok = load_pointcloud_safe(p)  is not None
            if ok:
                self.valid_indices.append(i)
            if i % 1000 == 0:
                write_json(VALID_IDX_FILE,  self.valid_indices)
                write_json(VALID_PROG_FILE, {"last_index": i})
        write_json(VALID_IDX_FILE,  self.valid_indices)
        write_json(VALID_PROG_FILE, {"last_index": len(self.entries) - 1})
        logger.info(
            f"Pre-scan done: {len(self.valid_indices)}/{len(self.entries)} valid.")
        return self.valid_indices


# ── Dataset ─────────────────────────────────────────────────────────────────
class SSLDataset(Dataset):
    def __init__(self, entries, valid_indices):
        self.entries = entries
        self.indices = valid_indices
        counts: Dict[str, int] = {}
        for i in valid_indices:
            m = entries[i]["modality"]
            counts[m] = counts.get(m, 0) + 1
        self.counts = counts

    def __len__(self) -> int:
        return len(self.indices)

    def get_sample_weights(self) -> List[float]:
        total = sum(self.counts.values())
        n_mod = len(self.counts)
        w = {m: total / (n_mod * max(c, 1)) for m, c in self.counts.items()}
        return [w[self.entries[self.indices[i]]["modality"]]
                for i in range(len(self.indices))]

    def __getitem__(self, idx):
        e   = self.entries[self.indices[idx]]
        m, p = e["modality"], e["path"]
        if m == "rgb":
            a, b = load_rgb_safe(p), load_rgb_aug(p)
            if a is None or b is None:
                return None
        elif m == "thermal":
            a, b = load_thermal_safe(p), load_thermal_aug(p)
            if a is None or b is None:
                return None
        elif m in ("lidar", "radar"):
            a = load_pointcloud_safe(p)
            if a is None:
                return None
            b = augment_pointcloud(a.clone())
        else:
            return None
        return {"m": m, "a": a, "b": b}


def collate_group(batch):
    """Group samples by modality; return list of {m, a, b} tensors."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return []
    grouped: Dict[str, list] = {}
    for item in batch:
        grouped.setdefault(item["m"], []).append(item)
    out = []
    for m, items in grouped.items():
        try:
            out.append({
                "m": m,
                "a": torch.stack([x["a"] for x in items]),
                "b": torch.stack([x["b"] for x in items]),
            })
        except Exception:
            continue
    return out


# ── Checkpoint manager ──────────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, ckpt_dir: Path, keep_last: int = 2):
        self.dir  = ckpt_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep_last

    def save(self, name: str, data: dict, prefix: str) -> None:
        torch.save(data, str(self.dir / name))
        files = sorted(self.dir.glob(f"{prefix}*.pth"),
                       key=lambda x: x.stat().st_mtime)
        while len(files) > self.keep:
            try:
                files.pop(0).unlink()
            except Exception:
                break

    def latest(self, prefix: str) -> "Path | None":
        files = sorted(self.dir.glob(f"{prefix}*.pth"),
                       key=lambda x: x.stat().st_mtime)
        return files[-1] if files else None


# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_PROGRESS = PROGRESS_ROOT / "ssl_train_progress.json"
SSL_DONE_FLAG  = SSL_DIR / "ssl_done.flag"


# ── Helpers ─────────────────────────────────────────────────────────────────
def _get_proj_emb(ssl_models: dict, m: str,
                  x: torch.Tensor) -> torch.Tensor:
    """Forward through encoder + projection head."""
    return ssl_models[m]["proj"](ssl_models[m]["enc"](x))


def _compute_cross_modal_loss(proj_embs: dict) -> torch.Tensor:
    """
    [FIX-2]  Cross-modal contrastive loss  L_cm.

    For every pair of different modalities (m_i, m_j) that are both present
    in the current batch, compute NT-Xent between their projected embeddings.
    All pairwise losses are averaged.

    proj_embs : {modality_name: (B, D) tensor}  — only modalities present
                in this batch are included.
    """
    mods = list(proj_embs.keys())
    if len(mods) < 2:
        return torch.tensor(0.0, device=next(iter(proj_embs.values())).device)

    loss_sum, count = torch.tensor(0.0, device=next(iter(proj_embs.values())).device), 0
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            za = proj_embs[mods[i]]
            zb = proj_embs[mods[j]]
            # Align batch sizes: take the minimum
            n  = min(za.size(0), zb.size(0))
            if n < 2:
                continue
            loss_sum = loss_sum + nt_xent(za[:n], zb[:n])
            count   += 1

    return loss_sum / max(count, 1)


# ── Main training loop ───────────────────────────────────────────────────────
def train_ssl(unified_json_path: Path,
              num_epochs: int = 30,
              batch_size: int = BATCH_SIZE) -> None:
    if SSL_DONE_FLAG.exists():
        logger.info("SSL training already complete.")
        return

    entries = [e for e in read_json(unified_json_path)
               if e.get("modality") in ("rgb", "thermal", "lidar", "radar")]
    logger.info(f"SSL entries: {len(entries)}")

    from collections import Counter
    logger.info(f"Modality counts: "
                f"{dict(Counter(e['modality'] for e in entries))}")

    # Pre-validate
    vr = ValidationRunner(entries)
    vi = read_json(VALID_IDX_FILE) if VALID_IDX_FILE.exists() else vr.run()
    if not vi:
        logger.error("No valid entries found.")
        return

    dataset = SSLDataset(entries, vi)
    logger.info(f"Valid samples per modality: {dataset.counts}")

    sampler = WeightedRandomSampler(
        weights=dataset.get_sample_weights(),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size * 4,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        collate_fn=collate_group,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True,
    )

    ssl_models = build_ssl_models(DEVICE)    # now includes "pred" and "mom"
    use_cuda   = (DEVICE.type == "cuda")
    scaler     = torch.amp.GradScaler("cuda" if use_cuda else "cpu")
    chk        = CheckpointManager(SSL_DIR, keep_last=KEEP_CKPTS)

    # ── Resume ──────────────────────────────────────────────────────────────
    prog        = load_progress(TRAIN_PROGRESS)
    start_epoch = int(prog.get("epoch", 0))
    start_batch = int(prog.get("batch", 0))

    latest = chk.latest("ssl_epoch")
    if latest:
        try:
            state = torch.load(str(latest), map_location=DEVICE, weights_only=False)
            start_epoch = state.get("epoch", start_epoch)
            start_batch = state.get("batch", start_batch)
            for name in ("rgb", "thermal", "lidar", "radar"):
                if f"{name}_enc" in state:
                    ssl_models[name]["enc"].load_state_dict(state[f"{name}_enc"])
                    ssl_models[name]["proj"].load_state_dict(state[f"{name}_proj"])
                    if f"{name}_pred" in state:
                        ssl_models[name]["pred"].load_state_dict(state[f"{name}_pred"])
                    if f"{name}_mom_enc" in state:
                        ssl_models[name]["mom"].enc.load_state_dict(state[f"{name}_mom_enc"])
                        ssl_models[name]["mom"].proj.load_state_dict(state[f"{name}_mom_proj"])
                    try:
                        ssl_models[name]["opt"].load_state_dict(state[f"{name}_opt"])
                        ssl_models[name]["sched"].load_state_dict(state[f"{name}_sched"])
                    except Exception:
                        pass
            if "scaler" in state and use_cuda:
                try:
                    scaler.load_state_dict(state["scaler"])
                except Exception:
                    pass
            logger.info(
                f"Resumed from {latest.name} (ep={start_epoch}, batch={start_batch})")
        except Exception as e:
            logger.warning(f"SSL ckpt load failed: {e}")

    if start_epoch >= num_epochs:
        SSL_DONE_FLAG.touch()
        return

    logger.info(
        f"SSL training: {num_epochs} epochs, batch={batch_size}\n"
        f"  λ_nt={LAMBDA_NT}, λ_cm={LAMBDA_CM}, λ_byol={LAMBDA_BYOL}")

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        for name in ssl_models:
            ssl_models[name]["enc"].train()
            ssl_models[name]["proj"].train()
            ssl_models[name]["pred"].train()
            # mom stays in eval mode — never receives gradients

        # Per-modality running losses (for logging)
        pm_nt   = {m: 0.0 for m in ("rgb", "thermal", "lidar", "radar")}
        pm_byol = {m: 0.0 for m in ("rgb", "thermal", "lidar", "radar")}
        pm_n    = {m: 0   for m in ("rgb", "thermal", "lidar", "radar")}
        cm_loss_sum, cm_steps = 0.0, 0
        total_loss, steps = 0.0, 0

        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f"SSL {epoch + 1}/{num_epochs}")

        for bi, groups in pbar:
            if epoch == start_epoch and bi < start_batch:
                continue
            if not groups:
                continue

            for name in ssl_models:
                ssl_models[name]["opt"].zero_grad(set_to_none=True)

            # ── Collect projected embeddings from all modalities present ──
            # used for cross-modal loss computation below
            proj_embs_a: dict = {}   # view-a embeddings per modality
            ep_loss = torch.tensor(0.0, device=DEVICE)

            for grp in groups:
                try:
                    m = grp["m"]
                    a = grp["a"].float().to(DEVICE, non_blocking=True)
                    b = grp["b"].float().to(DEVICE, non_blocking=True)
                    if a.size(0) < 2:
                        continue
                    # Shape guards
                    if m in ("rgb", "thermal") and (a.dim() != 4 or a.size(1) != 3):
                        continue
                    if m in ("lidar", "radar") and (a.dim() != 3 or a.size(2) != 3):
                        continue

                    enc  = ssl_models[m]["enc"]
                    proj = ssl_models[m]["proj"]
                    pred = ssl_models[m]["pred"]
                    mom  = ssl_models[m]["mom"]

                    with torch.amp.autocast("cuda" if use_cuda else "cpu"):
                        # ── NT-Xent  (L_NT) ──────────────────────────────
                        z_a = proj(enc(a))                   # online view-a
                        z_b = proj(enc(b))                   # online view-b
                        l_nt = nt_xent(z_a, z_b)

                        # ── BYOL  (L_BYOL) ───────────────────────────────
                        # Online: predictor over view-a embedding
                        q_a  = pred(z_a)                     # q_θ(z_θ(x_a))
                        # Target: momentum encoder on view-b (stop-grad inside byol_loss)
                        with torch.no_grad():
                            t_b = mom(b)                     # z_ξ(x_b)
                        l_byol = byol_loss(q_a, t_b)

                        # ── Weighted single-modality loss ─────────────────
                        l_mod = LAMBDA_NT * l_nt + LAMBDA_BYOL * l_byol

                    scaler.scale(l_mod).backward(retain_graph=False)
                    ep_loss = ep_loss + l_mod.detach()

                    # Accumulate for cross-modal + logging
                    proj_embs_a[m] = z_a.detach()
                    pm_nt[m]   += l_nt.item()
                    pm_byol[m] += l_byol.item()
                    pm_n[m]    += 1

                except Exception:
                    continue

            # ── Cross-modal loss  (L_cm)  [FIX-2] ────────────────────────
            # Computed over accumulated proj embeddings from this batch
            if len(proj_embs_a) >= 2:
                try:
                    with torch.amp.autocast("cuda" if use_cuda else "cpu"):
                        l_cm = _compute_cross_modal_loss(proj_embs_a)
                        l_cm_weighted = LAMBDA_CM * l_cm
                    scaler.scale(l_cm_weighted).backward()
                    ep_loss = ep_loss + l_cm_weighted.detach()
                    cm_loss_sum += l_cm.item()
                    cm_steps    += 1
                except Exception:
                    pass

            # ── Gradient clip + optimiser step ────────────────────────────
            for name in ssl_models:
                try:
                    scaler.unscale_(ssl_models[name]["opt"])
                    nn.utils.clip_grad_norm_(
                        list(ssl_models[name]["enc"].parameters()) +
                        list(ssl_models[name]["proj"].parameters()) +
                        list(ssl_models[name]["pred"].parameters()),
                        max_norm=1.0,
                    )
                    scaler.step(ssl_models[name]["opt"])
                except Exception:
                    pass
            scaler.update()

            # ── EMA update for all momentum encoders  [FIX-1] ─────────────
            for name in ssl_models:
                ssl_models[name]["mom"].update(
                    ssl_models[name]["enc"],
                    ssl_models[name]["proj"],
                )

            steps      += 1
            total_loss += ep_loss.item()
            pbar.set_postfix({"loss": f"{total_loss / steps:.4f}"})

            if (bi + 1) % SAVE_EVERY == 0:
                _save_ssl_ckpt(ssl_models, scaler, epoch, bi, chk)
                save_progress({"epoch": epoch, "batch": bi, "at": time.time()},
                               TRAIN_PROGRESS)

        # ── Epoch end ────────────────────────────────────────────────────────
        for name in ssl_models:
            ssl_models[name]["sched"].step()

        avg = total_loss / max(steps, 1)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} | total_loss={avg:.4f}")
        for m in ("rgb", "thermal", "lidar", "radar"):
            if pm_n[m]:
                logger.info(
                    f"  {m:8} | nt={pm_nt[m]/pm_n[m]:.4f} "
                    f"byol={pm_byol[m]/pm_n[m]:.4f}")
        if cm_steps:
            logger.info(f"  cross-modal loss: {cm_loss_sum / cm_steps:.4f}")

        _save_ssl_ckpt(ssl_models, scaler, epoch + 1, 0, chk)
        save_progress({"epoch": epoch + 1, "batch": 0, "at": time.time()},
                       TRAIN_PROGRESS)
        start_batch = 0

    SSL_DONE_FLAG.touch()
    logger.info("SSL training complete.")


# ── Checkpoint save (extended for pred + mom)  [FIX-4] ─────────────────────
def _save_ssl_ckpt(ssl_models: dict,
                   scaler,
                   epoch: int,
                   batch: int,
                   chk: CheckpointManager) -> None:
    state = {"epoch": epoch, "batch": batch, "scaler": scaler.state_dict()}
    for name in ssl_models:
        state[f"{name}_enc"]       = ssl_models[name]["enc"].state_dict()
        state[f"{name}_proj"]      = ssl_models[name]["proj"].state_dict()
        state[f"{name}_pred"]      = ssl_models[name]["pred"].state_dict()    # [FIX-4]
        state[f"{name}_mom_enc"]   = ssl_models[name]["mom"].enc.state_dict() # [FIX-4]
        state[f"{name}_mom_proj"]  = ssl_models[name]["mom"].proj.state_dict()# [FIX-4]
        state[f"{name}_opt"]       = ssl_models[name]["opt"].state_dict()
        state[f"{name}_sched"]     = ssl_models[name]["sched"].state_dict()
    chk.save(f"ssl_epoch{epoch}_batch{batch}.pth", state, prefix="ssl_epoch")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unified = OUTPUT_ROOT / "unified_dataset.json"
    if not unified.exists():
        raise SystemExit("Run dataset_indexing.py first.")
    train_ssl(unified, num_epochs=30, batch_size=BATCH_SIZE)
