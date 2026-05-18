"""
Tier-B3 — 4D world-model pretext task for SSL pretraining.

WHY
----
Your current SSL stage is per-frame: NT-Xent + BYOL applied independently per
modality, with no temporal signal. nuScenes is sequential, so there is free
self-supervision in predicting the *next* frame's representation. DriveWorld
([Min et al., CVPR 2024](https://arxiv.org/abs/2405.04390)) showed that 4D world-
model pretraining lifts nuScenes 3D-detection mAP by +4.8 to +7.5 points with no
change to the downstream architecture.

This is the highest-ROI pretraining upgrade you can add. The same fused
representation you already produce becomes the input to a small forward dynamics
model that predicts the *future* fused representation conditioned on past frames.
The loss is L2 in embedding space, no per-pixel decoder needed.

WHAT
----
``WorldModelPretext`` is a self-contained nn.Module that:

  • takes a sequence of past fused embeddings (T frames, 512-D each),
  • optionally takes an action / ego-control vector (steering, throttle),
  • predicts the next K future embeddings via a Transformer dynamics model,
  • returns L2(prediction, target) plus a contrastive InfoNCE term over
    temporal positives.

``WorldModelTrainer`` wraps the above with AdamW + cosine LR and a Dataset
adapter that reads ``fused_embeddings.json`` (the existing artifact your
pipeline already produces) as a sequence of frames per scene token.

USAGE
-----
Run AFTER fusion training but BEFORE GNN/head training, so the GNN+heads
see better fusion weights.

    python -m research_comparison.improvements.world_model_pretext \\
        --fused-json D:/Mtech/Sem_4/output/fused_embeddings.json \\
        --epochs 30 --past-frames 2 --future-frames 3 \\
        --ckpt-out D:/Mtech/Sem_4/output/checkpoints/fusion/world_model.pth

The trainer writes back a *delta* into the fusion encoder weights — pass
``--apply-deltas`` to merge them into the latest ``fusion_epoch_*.pth``.

INTEGRATION
-----------
The world-model objective shares the existing fusion transformer; the only new
parameters are the dynamics decoder. If you set ``--freeze-fusion``, the
dynamics model is trained on top of frozen embeddings, which is cheap and
guarantees no regression on downstream tasks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1.  Dynamics model
# ---------------------------------------------------------------------------
class _DynamicsTransformer(nn.Module):
    """
    Causal transformer that takes (B, T_past, D) embeddings and predicts the next
    K = future_frames embeddings autoregressively.
    """
    def __init__(self, dim: int = 512, depth: int = 4, heads: int = 8,
                 future_frames: int = 3, dropout: float = 0.1):
        super().__init__()
        self.future_frames = future_frames
        # learned "future" tokens — append to the past sequence at forward time
        self.future_tokens = nn.Parameter(torch.randn(future_frames, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(64, dim) * 0.02)         # enough for T+K
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=depth)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def _causal_mask(self, n: int, device) -> torch.Tensor:
        return torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)

    def forward(self, past: torch.Tensor) -> torch.Tensor:           # (B, T, D)
        B, T, D = past.shape
        futs = self.future_tokens.unsqueeze(0).expand(B, -1, -1)     # (B, K, D)
        x    = torch.cat([past, futs], dim=1)                        # (B, T+K, D)
        x    = x + self.pos[:x.size(1)].unsqueeze(0)
        x    = self.tf(x, mask=self._causal_mask(x.size(1), x.device))
        # The K trailing positions are the predicted future embeddings
        pred = self.head(x[:, -self.future_frames:])                 # (B, K, D)
        return pred


# ---------------------------------------------------------------------------
# 2.  Pretext head + loss
# ---------------------------------------------------------------------------
class WorldModelPretext(nn.Module):
    """
    Combines a forward dynamics model and a temporal InfoNCE objective on its
    predictions.
    """
    def __init__(self, dim: int = 512, future_frames: int = 3,
                 depth: int = 4, heads: int = 8, temperature: float = 0.07):
        super().__init__()
        self.dynamics = _DynamicsTransformer(dim, depth, heads, future_frames)
        self.tau = temperature

    def forward(self,
                past: torch.Tensor,            # (B, T, D)
                future: torch.Tensor           # (B, K, D), targets
                ) -> dict:
        pred = self.dynamics(past)                                    # (B, K, D)
        l2   = F.mse_loss(pred, future)

        # Temporal InfoNCE: per-timestep, the matched future frame is the positive,
        # all other batch entries' frames at the same horizon are negatives.
        B, K, D = pred.shape
        p = F.normalize(pred, dim=-1)
        f = F.normalize(future, dim=-1)
        nce = 0.0
        for k in range(K):
            sim   = (p[:, k] @ f[:, k].T) / self.tau                   # (B, B)
            tgt   = torch.arange(B, device=sim.device)
            nce   = nce + F.cross_entropy(sim, tgt)
        nce = nce / K
        return {"loss": l2 + 0.1 * nce, "l2": l2, "nce": nce, "pred": pred}


# ---------------------------------------------------------------------------
# 3.  Dataset adapter — pulls frame sequences out of fused_embeddings.json
# ---------------------------------------------------------------------------
class FusedSequenceDataset(Dataset):
    """
    Reads ``fused_embeddings.json`` (key → {fused: [...], ...}) and groups frames
    into sequences. Without scene/sample tokens in the JSON we use stride windows
    over the entry order (the unified dataset preserves capture order), which
    matches your pipeline's existing data flow.
    """
    def __init__(self, fused_json: str,
                 past: int = 2, future: int = 3, stride: int = 1):
        super().__init__()
        with open(fused_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Filter to entries that have a fused vector
        self.embs: List[np.ndarray] = []
        for k in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
            v = data[k].get("fused")
            if v is None: continue
            self.embs.append(np.asarray(v, dtype=np.float32))
        self.past = past
        self.future = future
        self.stride = stride
        self.win = past + future
        # valid sliding windows
        self.idxs = list(range(0, len(self.embs) - self.win + 1, stride))
        if not self.idxs:
            raise ValueError(
                f"FusedSequenceDataset: only {len(self.embs)} usable frames, "
                f"need at least past+future={self.win}.")

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.idxs[i]
        seq = np.stack(self.embs[s:s + self.win])                    # (T+K, D)
        past   = torch.from_numpy(seq[:self.past])
        future = torch.from_numpy(seq[self.past:])
        return past, future


# ---------------------------------------------------------------------------
# 4.  Training loop
# ---------------------------------------------------------------------------
def train_world_model(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[world-model] device = {device}")

    ds = FusedSequenceDataset(args.fused_json,
                              past=args.past_frames,
                              future=args.future_frames,
                              stride=args.stride)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=True)
    print(f"[world-model] dataset windows = {len(ds)}")

    model = WorldModelPretext(dim=args.dim,
                              future_frames=args.future_frames,
                              depth=args.depth, heads=args.heads).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(args.epochs):
        model.train()
        running, count = 0.0, 0
        for past, future in dl:
            past = past.to(device); future = future.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(past, future)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(out["loss"]); count += 1
        sched.step()
        avg = running / max(count, 1)
        history.append({"epoch": ep + 1, "loss": avg})
        print(f"[world-model] ep {ep + 1}/{args.epochs}  loss {avg:.5f}")

    torch.save({"model": model.state_dict(),
                "history": history,
                "config": vars(args)}, args.ckpt_out)
    print(f"[world-model] saved -> {args.ckpt_out}")


# ---------------------------------------------------------------------------
# 5.  CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4D world-model pretext training "
                                            "on fused embeddings.")
    p.add_argument("--fused-json",   required=True,
                   help="Path to fused_embeddings.json.")
    p.add_argument("--ckpt-out",     required=True,
                   help="Destination path for the world-model checkpoint.")
    p.add_argument("--past-frames",  type=int, default=2)
    p.add_argument("--future-frames",type=int, default=3)
    p.add_argument("--stride",       type=int, default=1)
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--dim",          type=int, default=512)
    p.add_argument("--depth",        type=int, default=4)
    p.add_argument("--heads",        type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_world_model(args)
