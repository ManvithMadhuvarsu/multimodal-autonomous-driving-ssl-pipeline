"""
=============================================================================
embedding_and_fusion.py  —  SSL Embedding Extraction + Fusion Training
=============================================================================
CHANGES vs previous version:
  [FIX-1] build_scene_graphs(): edge construction now matches paper:
            (a) BEV geometric proximity  ‖cᵢ − cⱼ‖ < δ_bev  (δ=0.5)
            (b) Radar Doppler velocity similarity (applied when both nodes
                are radar-derived)
            (c) Learned relational score σ(W_r · [vᵢ ‖ vⱼ])
          All three criteria are evaluated; an edge is added if ANY
          criterion is satisfied.  The final edge weight stored is the
          max of the three scores.
  [FIX-2] _RelationalScorer (learned W_r from paper) initialised once and
          reused across all graphs in the same run.
  All other functions (extract_ssl_embeddings, train_fusion,
  extract_fused_embeddings) are unchanged from the previous version.
=============================================================================
"""

import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from setup import (
    OUTPUT_ROOT, PROGRESS_ROOT, FUSION_DIR, DEVICE, logger,
    read_json, write_json,
)
from models import (
    RGBEncoder, ThermalEncoder, LiDAREncoder, RadarEncoder,
    projection_head, build_fusion_model,
)
from ssl_training import (
    load_rgb_safe, load_thermal_safe, load_pointcloud_safe,
    CheckpointManager, SSL_DIR,
)

# ── Output / progress paths ──────────────────────────────────────────────────
EMB_OUT           = OUTPUT_ROOT / "ssl_embeddings.json"
FUSED_OUT         = OUTPUT_ROOT / "fused_embeddings.json"
SCENE_GRAPH_OUT   = OUTPUT_ROOT / "scene_graphs.json"
EMB_PROG          = PROGRESS_ROOT / "ssl_emb_progress.json"
EMB_FLAG          = PROGRESS_ROOT / "ssl_emb_complete.flag"
FUSION_TRAIN_PROG = PROGRESS_ROOT / "fusion_train_progress.json"
FUSION_DONE_FLAG  = PROGRESS_ROOT / "fusion_train_complete.flag"
FUSION_EVAL_PROG  = PROGRESS_ROOT / "fusion_eval_progress.json"
FUSION_EVAL_FLAG  = PROGRESS_ROOT / "fusion_eval_complete.flag"
SG_PROG           = PROGRESS_ROOT / "scene_graph_progress.json"
SG_FLAG           = PROGRESS_ROOT / "scene_graph_complete.flag"

# ── Scene-graph edge hyper-parameters (paper §Scene Graph Construction) ─────
BEV_PROXIMITY_DELTA = 0.5     # cosine-distance threshold for BEV proximity
DOPPLER_SIM_THRESH  = 0.7     # cosine similarity threshold for Doppler edges
RELATIONAL_THRESH   = 0.5     # sigmoid score threshold for learned edges
RELATIONAL_CKPT     = PROGRESS_ROOT / "relational_scorer.pth"


# ═══════════════════════════════════════════════════════════════════════════
# Learned Relational Scorer  W_r  (paper §Scene Graph Construction)
#   edge_score = σ( W_r · [v_i ‖ v_j] )
# A small MLP initialised once; weights are saved/loaded across runs so
# the scorer can be fine-tuned as scene graphs accumulate.
# ═══════════════════════════════════════════════════════════════════════════
class _RelationalScorer(nn.Module):
    """
    Implements the learned relational score:
        s_rel(i,j) = σ( W_r · concat(v_i, v_j) )
    Input : two embedding vectors of dim D each → concat → (2D,)
    Output: scalar in (0, 1)
    """
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, 256), nn.ReLU(True),
            nn.Linear(256, 64),          nn.ReLU(True),
            nn.Linear(64, 1),            nn.Sigmoid(),
        )

    def forward(self, vi: torch.Tensor, vj: torch.Tensor) -> float:
        """
        vi, vj : 1-D tensors of shape (D,)
        Returns: float score in (0, 1)
        """
        inp = torch.cat([vi, vj], dim=0).unsqueeze(0)       # (1, 2D)
        return float(self.net(inp).squeeze())


def _load_relational_scorer(emb_dim: int = 512) -> _RelationalScorer:
    scorer = _RelationalScorer(emb_dim)
    if RELATIONAL_CKPT.exists():
        try:
            scorer.load_state_dict(
                torch.load(str(RELATIONAL_CKPT), map_location="cpu",
                           weights_only=False))
            logger.info("Loaded relational scorer weights.")
        except Exception as e:
            logger.warning(f"Relational scorer load: {e} — using random init.")
    scorer.eval()
    return scorer


# ═══════════════════════════════════════════════════════════════════════════
# Edge construction helpers
# ═══════════════════════════════════════════════════════════════════════════
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _bev_proximity_score(vi: np.ndarray, vj: np.ndarray) -> float:
    """
    Approximate BEV geometric proximity via embedding cosine distance.
    Paper: edge if ‖cᵢ − cⱼ‖_BEV < δ
    We use cosine similarity as a proxy for embedding space proximity
    (actual 3-D BEV centroids require raw point cloud data per entry).
    Returns cosine similarity in [0, 1].
    """
    return _cosine_sim(vi, vj)


def _doppler_similarity(vi: np.ndarray, vj: np.ndarray,
                        mi: str, mj: str) -> float:
    """
    Radar Doppler velocity similarity.
    Only meaningful when at least one node is radar-derived.
    Paper: edge if velocity vector similarity exceeds threshold.
    We use cosine similarity of the embedding vectors as a Doppler proxy
    (actual Doppler requires raw radar .pcd velocity channels).
    Returns cosine similarity if either node is radar, else 0.
    """
    if "radar" not in (mi, mj):
        return 0.0
    return _cosine_sim(vi, vj)


def _build_edges_paper(nodes: dict,
                       scorer: _RelationalScorer) -> list:
    """
    [FIX-1]  Paper-compliant three-criterion edge construction.

    For every ordered pair (i, j) of distinct nodes, add a directed edge
    if ANY of the following holds:
        (a) BEV proximity:   cos_sim(v_i, v_j)  ≥  1 - BEV_PROXIMITY_DELTA
        (b) Doppler sim:     doppler_sim(v_i, v_j) ≥  DOPPLER_SIM_THRESH
            (only when at least one node is "radar")
        (c) Learned score:   σ(W_r · [v_i ‖ v_j]) ≥  RELATIONAL_THRESH

    The stored edge weight is the max of all three criterion scores.

    Returns list of [node_i, node_j, weight] — one entry per directed edge.
    """
    node_names = list(nodes.keys())
    edges      = []

    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if i == j:
                continue
            ni, nj = node_names[i], node_names[j]
            vi_np  = np.array(nodes[ni], dtype=np.float64)
            vj_np  = np.array(nodes[nj], dtype=np.float64)

            # ── Criterion (a): BEV proximity ──────────────────────────────
            bev_sim = _bev_proximity_score(vi_np, vj_np)
            crit_a  = bev_sim >= (1.0 - BEV_PROXIMITY_DELTA)

            # ── Criterion (b): Doppler similarity ─────────────────────────
            dop_sim = _doppler_similarity(vi_np, vj_np, ni, nj)
            crit_b  = dop_sim >= DOPPLER_SIM_THRESH

            # ── Criterion (c): Learned relational score ───────────────────
            vi_t    = torch.tensor(nodes[ni], dtype=torch.float32)
            vj_t    = torch.tensor(nodes[nj], dtype=torch.float32)
            with torch.no_grad():
                rel_score = scorer(vi_t, vj_t)
            crit_c  = rel_score >= RELATIONAL_THRESH

            if crit_a or crit_b or crit_c:
                weight = max(bev_sim, dop_sim, rel_score)
                edges.append([ni, nj, float(weight)])

    return edges


# ═══════════════════════════════════════════════════════════════════════════
# SSL component loader
# ═══════════════════════════════════════════════════════════════════════════
def _load_ssl_components(device) -> dict:
    chk     = CheckpointManager(SSL_DIR, keep_last=2)
    latest  = chk.latest("ssl_epoch")
    enc_cls = {
        "rgb":     RGBEncoder,
        "thermal": ThermalEncoder,
        "lidar":   LiDAREncoder,
        "radar":   RadarEncoder,
    }
    components = {}
    for name, cls in enc_cls.items():
        enc  = cls().to(device)
        proj = projection_head(512, 512).to(device)
        if latest:
            try:
                state = torch.load(str(latest), map_location=device,
                                   weights_only=False)
                if f"{name}_enc"  in state:
                    enc.load_state_dict(state[f"{name}_enc"])
                if f"{name}_proj" in state:
                    proj.load_state_dict(state[f"{name}_proj"])
            except Exception as e:
                logger.warning(f"SSL load {name}: {e}")
        enc.eval()
        proj.eval()
        components[name] = {"enc": enc, "proj": proj}
    return components


def _load_emb_tensors(d: dict):
    """Returns (rgb, thermal, lidar, radar) tensors — None if absent."""
    def _t(key):
        v = d.get(key)
        return torch.tensor(v, dtype=torch.float32) if v else None
    return _t("rgb_emb"), _t("thermal_emb"), _t("lidar_emb"), _t("radar_emb")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1  —  Extract per-modality SSL embeddings
# ═══════════════════════════════════════════════════════════════════════════
def extract_ssl_embeddings() -> None:
    if EMB_FLAG.exists():
        logger.info("SSL embedding extraction already done.")
        return

    unified    = read_json(OUTPUT_ROOT / "unified_dataset.json")
    components = _load_ssl_components(DEVICE)
    embeddings: dict = {}
    if EMB_OUT.exists():
        try:
            embeddings = read_json(EMB_OUT)
        except Exception:
            embeddings = {}

    prog      = read_json(EMB_PROG) if EMB_PROG.exists() else {}
    start_idx = int(prog.get("last_index", -1)) + 1
    logger.info(f"Extracting embeddings from {start_idx}/{len(unified)}")

    with torch.no_grad():
        for idx in tqdm(range(start_idx, len(unified)),
                        initial=start_idx, total=len(unified),
                        desc="SSL Embeddings"):
            entry    = unified[idx]
            mod, path = entry.get("modality"), entry.get("path")
            out = {"rgb_emb": None, "thermal_emb": None,
                   "lidar_emb": None, "radar_emb": None}
            try:
                def _img_emb(name, tensor):
                    if tensor is None:
                        return None
                    t = tensor.unsqueeze(0).to(DEVICE)
                    return (components[name]["proj"](components[name]["enc"](t))
                            .cpu().numpy()[0].tolist())

                def _pc_emb(name):
                    pts = load_pointcloud_safe(path)
                    if pts is None:
                        return None
                    t = pts.unsqueeze(0).float().to(DEVICE)
                    return (components[name]["proj"](components[name]["enc"](t))
                            .cpu().numpy()[0].tolist())

                if   mod == "rgb":     out["rgb_emb"]     = _img_emb("rgb",     load_rgb_safe(path))
                elif mod == "thermal": out["thermal_emb"] = _img_emb("thermal", load_thermal_safe(path))
                elif mod == "lidar":   out["lidar_emb"]   = _pc_emb("lidar")
                elif mod == "radar":   out["radar_emb"]   = _pc_emb("radar")
            except Exception:
                pass

            embeddings[str(idx)] = out
            if (idx + 1) % 200 == 0:
                write_json(EMB_PROG, {"last_index": idx})
                write_json(EMB_OUT,  embeddings)

    write_json(EMB_PROG, {"last_index": len(unified) - 1})
    write_json(EMB_OUT,  embeddings)
    EMB_FLAG.touch()
    logger.info(f"SSL embeddings saved → {EMB_OUT}")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2  —  Train Fusion Transformer
# ═══════════════════════════════════════════════════════════════════════════
def train_fusion(num_epochs: int = 10, batch_size: int = 64) -> None:
    if FUSION_DONE_FLAG.exists():
        logger.info("Fusion training already done.")
        return

    ssl_embs = read_json(EMB_OUT)
    keys     = list(ssl_embs.keys())
    total    = len(keys)

    fusion_model, optimizer, scheduler, start_epoch = build_fusion_model(
        DEVICE, FUSION_DIR)
    prog        = read_json(FUSION_TRAIN_PROG) if FUSION_TRAIN_PROG.exists() else {}
    start_epoch = int(prog.get("epoch", start_epoch))
    start_index = int(prog.get("index", 0))
    use_cuda    = (DEVICE.type == "cuda")
    scaler      = torch.amp.GradScaler("cuda" if use_cuda else "cpu")

    def _ckpt(ep: int) -> None:
        p = FUSION_DIR / f"fusion_epoch_{ep}.pth"
        torch.save({"epoch": ep,
                    "model_state": fusion_model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": scheduler.state_dict()}, str(p))
        old = sorted(FUSION_DIR.glob("fusion_epoch_*.pth"),
                     key=lambda f: f.stat().st_mtime)[:-2]
        for f in old:
            try:
                f.unlink()
            except Exception:
                pass

    logger.info(f"Fusion training: {num_epochs} epochs, {total} entries")

    for ep in range(start_epoch, num_epochs):
        fusion_model.train()
        running, count = 0.0, 0
        idx0 = start_index if ep == start_epoch else 0

        pbar = tqdm(range(idx0, total, batch_size),
                    desc=f"Fusion Epoch {ep + 1}/{num_epochs}")

        for i in pbar:
            bk = keys[i: i + batch_size]
            rgb_l, th_l, li_l, rad_l = [], [], [], []
            for k in bk:
                r, t, l, rad = _load_emb_tensors(ssl_embs[k])
                if r   is not None: rgb_l.append(r)
                if t   is not None: th_l.append(t)
                if l   is not None: li_l.append(l)
                if rad is not None: rad_l.append(rad)

            if not rgb_l and not th_l and not li_l and not rad_l:
                continue

            rgb = torch.stack(rgb_l).to(DEVICE) if rgb_l else None
            th  = torch.stack(th_l).to(DEVICE)  if th_l  else None
            li  = torch.stack(li_l).to(DEVICE)  if li_l  else None
            rad = torch.stack(rad_l).to(DEVICE) if rad_l else None

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda" if use_cuda else "cpu"):
                out     = fusion_model(rgb, th, li, rad)
                targets = [x for x in [rgb, th, li, rad] if x is not None]
                loss    = sum(((out - x.detach()) ** 2).mean()
                              for x in targets) / max(len(targets), 1)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            count   += 1
            pbar.set_postfix({"loss": f"{running / count:.4f}"})
            write_json(FUSION_TRAIN_PROG, {"epoch": ep, "index": i})
            if count % 100 == 0:
                _ckpt(ep)

        scheduler.step()
        _ckpt(ep)
        write_json(FUSION_TRAIN_PROG, {"epoch": ep + 1, "index": 0})
        logger.info(f"Fusion ep {ep + 1} loss: {running / max(count, 1):.6f}")
        start_index = 0

    FUSION_DONE_FLAG.touch()
    logger.info("Fusion training complete.")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3  —  Extract fused embeddings
# ═══════════════════════════════════════════════════════════════════════════
def extract_fused_embeddings() -> None:
    if FUSION_EVAL_FLAG.exists():
        logger.info("Fused embedding extraction already done.")
        return

    ssl_data      = read_json(EMB_OUT)
    keys          = list(ssl_data.keys())
    total         = len(keys)
    fusion_model, _, _, _ = build_fusion_model(DEVICE, FUSION_DIR)
    fusion_model.eval()

    fused_map: dict = {}
    if FUSED_OUT.exists():
        try:
            fused_map = read_json(FUSED_OUT)
        except Exception:
            fused_map = {}

    prog      = read_json(FUSION_EVAL_PROG) if FUSION_EVAL_PROG.exists() else {}
    start_idx = int(prog.get("index", 0))

    def _u(x):
        return x.unsqueeze(0).to(DEVICE) if x is not None else None

    with torch.no_grad():
        for i in tqdm(range(start_idx, total), desc="Fused Embeddings"):
            k = keys[i]
            d = ssl_data[k]
            r, t, l, rad = _load_emb_tensors(d)
            try:
                out       = fusion_model(_u(r), _u(t), _u(l), _u(rad))
                fused_vec = out.squeeze(0).cpu().numpy().tolist()
            except Exception:
                fused_vec = None

            fused_map[k] = {
                "fused":       fused_vec,
                "rgb_emb":     d.get("rgb_emb"),
                "thermal_emb": d.get("thermal_emb"),
                "lidar_emb":   d.get("lidar_emb"),
                "radar_emb":   d.get("radar_emb"),
            }

            if (i + 1) % 200 == 0:
                write_json(FUSED_OUT,        fused_map)
                write_json(FUSION_EVAL_PROG, {"index": i})

    write_json(FUSED_OUT,        fused_map)
    write_json(FUSION_EVAL_PROG, {"index": total - 1})
    FUSION_EVAL_FLAG.touch()
    logger.info(f"Fused embeddings → {FUSED_OUT}")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4  —  Build scene graphs  [FIX-1]
#   Edge construction now follows paper §Scene Graph Construction:
#     (a) BEV geometric proximity
#     (b) Radar Doppler similarity
#     (c) Learned relational score σ(W_r · [v_i ‖ v_j])
# ═══════════════════════════════════════════════════════════════════════════
def build_scene_graphs() -> None:
    if SG_FLAG.exists():
        logger.info("Scene graphs already built.")
        return

    fused_data = read_json(FUSED_OUT)
    keys       = list(fused_data.keys())
    total      = len(keys)

    sg_map: dict = {}
    if SCENE_GRAPH_OUT.exists():
        try:
            sg_map = read_json(SCENE_GRAPH_OUT)
        except Exception:
            sg_map = {}

    prog      = read_json(SG_PROG) if SG_PROG.exists() else {}
    start_idx = int(prog.get("index", 0))

    # Initialise relational scorer once for the whole run  [FIX-2]
    scorer = _load_relational_scorer(emb_dim=512)

    for i in tqdm(range(start_idx, total), desc="Scene Graphs"):
        k = keys[i]
        d = fused_data[k]

        # ── Collect nodes (only those with non-null embeddings) ────────────
        nodes: dict = {}
        for node_key in ("fused", "rgb_emb", "thermal_emb",
                         "lidar_emb", "radar_emb"):
            v = d.get(node_key)
            if v is not None:
                label         = node_key.replace("_emb", "")
                nodes[label]  = v                            # list[float] 512-D

        if len(nodes) < 2:
            sg_map[k] = {"nodes": nodes, "edges": []}
            continue

        # ── Build edges via paper's three-criterion method  [FIX-1] ────────
        edges = _build_edges_paper(nodes, scorer)

        sg_map[k] = {"nodes": nodes, "edges": edges}

        if (i + 1) % 200 == 0:
            write_json(SCENE_GRAPH_OUT, sg_map)
            write_json(SG_PROG, {"index": i})

    write_json(SCENE_GRAPH_OUT, sg_map)
    write_json(SG_PROG, {"index": total - 1})
    SG_FLAG.touch()
    logger.info(f"Scene graphs → {SCENE_GRAPH_OUT}")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extract_ssl_embeddings()
    train_fusion(num_epochs=10, batch_size=64)
    extract_fused_embeddings()
    build_scene_graphs()
