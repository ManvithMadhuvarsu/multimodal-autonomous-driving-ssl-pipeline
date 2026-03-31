"""
=============================================================================
06_gnn_training.py — GNN Training + Evaluation (Scene Graph Embeddings)
=============================================================================
FIXES vs original:
  1. radar node correctly handled in _prep_graph (5 possible node types now)
  2. Checkpoint save/load uses save_ckpt/load_ckpt properly
  3. LR scheduler added
  4. Gradient clipping added
=============================================================================
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from setup import (
    OUTPUT_ROOT, PROGRESS_ROOT, GNN_DIR, DEVICE, logger,
    read_json, write_json, save_ckpt, load_ckpt, find_latest
)
from models import GNNEncoder

SCENE_GRAPH_PATH = OUTPUT_ROOT / "scene_graphs.json"
GNN_EMB_OUT      = OUTPUT_ROOT / "gnn_embeddings.json"
GNN_TRAIN_PROG   = PROGRESS_ROOT / "gnn_train_progress.json"
GNN_TRAIN_FLAG   = PROGRESS_ROOT / "gnn_train_complete.flag"
GNN_EVAL_PROG    = PROGRESS_ROOT / "gnn_eval_progress.json"
GNN_EVAL_FLAG    = PROGRESS_ROOT / "gnn_eval_complete.flag"

# All possible node names (5 including radar)
ALL_NODES = ["fused", "rgb", "thermal", "lidar", "radar"]


def _prep_graph(entry: dict, device):
    nodes_dict = entry.get("nodes", {})
    names      = [n for n in ALL_NODES if n in nodes_dict]

    if not names:
        return (torch.zeros(1, 512, dtype=torch.float32, device=device),
                torch.zeros(2, 0, dtype=torch.long, device=device))

    feats = []
    for n in names:
        v = nodes_dict[n]
        feats.append(
            np.array(v, dtype=np.float32) if v is not None
            else np.zeros(512, dtype=np.float32)
        )
    node_tensor = torch.tensor(np.stack(feats), dtype=torch.float32, device=device)

    # Build fully-connected undirected edge index
    edges = entry.get("edges", [])
    pairs = []
    for edge in edges:
        if len(edge) >= 2:
            u, v = edge[0], edge[1]
            if u in names and v in names:
                ui, vi = names.index(u), names.index(v)
                pairs += [[ui, vi], [vi, ui]]

    edge_idx = (torch.zeros(2, 0, dtype=torch.long, device=device)
                if not pairs else
                torch.tensor(pairs, dtype=torch.long, device=device).T)

    return node_tensor, edge_idx


def train_gnn(num_epochs: int = 5):
    if GNN_TRAIN_FLAG.exists():
        logger.info("GNN training already done.")
        return

    if not SCENE_GRAPH_PATH.exists():
        raise FileNotFoundError(f"scene_graphs.json missing. Run 05 first.")

    sg_data = read_json(SCENE_GRAPH_PATH)
    keys    = list(sg_data.keys())
    total   = len(keys)

    gnn = GNNEncoder(in_dim=512, hid=512).to(DEVICE)
    opt = torch.optim.AdamW(gnn.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    start_epoch, start_batch = 0, 0
    latest = find_latest(GNN_DIR, "gnn")
    if latest:
        try:
            ep, extra = load_ckpt(latest, gnn, opt)
            start_epoch = ep + 1
            start_batch = extra.get("batch", 0) if extra else 0
            logger.info(f"Resumed GNN from {latest.name} (ep {ep})")
        except Exception as e:
            logger.warning(f"GNN ckpt load failed: {e}")

    if start_epoch >= num_epochs:
        GNN_TRAIN_FLAG.touch()
        return

    logger.info(f"GNN training: {num_epochs} epochs, {total} graphs")

    for ep in range(start_epoch, num_epochs):
        gnn.train()
        running, n_proc = 0.0, 0
        pbar = tqdm(range(total), desc=f"GNN Epoch {ep+1}/{num_epochs}")

        for bi in pbar:
            if ep == start_epoch and bi < start_batch:
                continue
            k = keys[bi]
            feats, edge_idx = _prep_graph(sg_data[k], DEVICE)

            opt.zero_grad(set_to_none=True)
            out  = gnn(feats, edge_idx)
            # Contrastive energy: minimise L2 norm for compact embeddings
            loss = (out ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            opt.step()

            running += loss.item(); n_proc += 1
            pbar.set_postfix({"loss": f"{running/n_proc:.4f}"})

            if (bi + 1) % 500 == 0:
                save_ckpt(GNN_DIR / f"gnn_epoch_{ep}.pth",
                          model=gnn, optim=opt, epoch=ep,
                          extra={"batch": bi})
                write_json(GNN_TRAIN_PROG, {"epoch": ep, "batch": bi})

        save_ckpt(GNN_DIR / f"gnn_epoch_{ep}.pth",
                  model=gnn, optim=opt, epoch=ep, extra={"batch": -1})
        sched.step()
        write_json(GNN_TRAIN_PROG, {"epoch": ep, "batch": -1})
        logger.info(f"GNN ep {ep+1} avg loss: {running/max(n_proc,1):.4f}")
        start_batch = 0

    GNN_TRAIN_FLAG.touch()
    logger.info("GNN training complete.")


def eval_gnn():
    if GNN_EVAL_FLAG.exists():
        logger.info("GNN evaluation already done.")
        return

    if not SCENE_GRAPH_PATH.exists():
        raise FileNotFoundError("scene_graphs.json missing.")

    sg_data = read_json(SCENE_GRAPH_PATH)
    keys    = list(sg_data.keys())
    total   = len(keys)

    gnn = GNNEncoder(in_dim=512, hid=512).to(DEVICE)
    opt = torch.optim.AdamW(gnn.parameters(), lr=1e-4)
    latest = find_latest(GNN_DIR, "gnn")
    if latest:
        try:
            ep, _ = load_ckpt(latest, gnn, opt)
            logger.info(f"Loaded GNN from {latest.name} (ep {ep})")
        except Exception as e:
            logger.warning(f"GNN load: {e}")
    gnn.eval()

    existing: dict = {}
    if GNN_EMB_OUT.exists():
        try: existing = read_json(GNN_EMB_OUT)
        except: existing = {}

    prog      = read_json(GNN_EVAL_PROG) if GNN_EVAL_PROG.exists() else {}
    start_idx = int(prog.get("last_index", -1)) + 1
    logger.info(f"GNN eval: {total} graphs, resuming from {start_idx}")

    with torch.no_grad():
        for idx in tqdm(range(start_idx, total), desc="GNN Eval"):
            k = keys[idx]
            feats, edge_idx = _prep_graph(sg_data[k], DEVICE)
            out = gnn(feats, edge_idx)
            existing[str(k)] = out.cpu().numpy().tolist()

            if (idx + 1) % 200 == 0:
                write_json(GNN_EVAL_PROG, {"last_index": idx})
                write_json(GNN_EMB_OUT,   existing)

    write_json(GNN_EVAL_PROG, {"last_index": total - 1})
    write_json(GNN_EMB_OUT,   existing)
    GNN_EVAL_FLAG.touch()
    logger.info(f"GNN embeddings -> {GNN_EMB_OUT}")


if __name__ == "__main__":
    train_gnn(num_epochs=5)
    eval_gnn()
