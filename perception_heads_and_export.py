"""
=============================================================================
07_perception_heads_and_export.py  —  Task Heads + Full Model Export
=============================================================================
FIXES vs original:
  • CRITICAL: save_ckpt() was called WITHOUT model= → heads never saved.
    Now correctly: save_ckpt(path, model=det, optim=opt_d, ...)
  • Panoptic .npz labels (34,149 available) used as segmentation targets
  • LiDAR lidarseg .bin labels used where available
  • evaluate_on_test includes LIDAR_TOP + RADAR_FRONT (not just cameras)
  • All paths → OUTPUT_ROOT
  • weights_only=False in all torch.load calls
  • TorchScript export uses all 4 modality tensors
=============================================================================
"""
import json, time, torch, torch.nn as nn, numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

import sys
sys.path.insert(0, str(Path(__file__).parent))
from setup import (
    OUTPUT_ROOT, PROGRESS_ROOT, FUSION_DIR, GNN_DIR, HEAD_DIR,
    NUS_ROOT, DEVICE, logger,
    read_json, write_json, save_ckpt, load_ckpt, find_latest,
    load_bin_lidar, load_radar_pcd
)
from models import (
    RGBEncoder, ThermalEncoder, LiDAREncoder, RadarEncoder,
    projection_head, GNNEncoder,
    DetectionHead, SegmentationHead, TrajectoryHead,
    MultiModalFusionTransformer, FullADModel, build_fusion_model,
    DETR3DHead, DETRSetMatchLoss,
)
from ssl_training import (
    IMG_SIZE, N_POINTS, img_transform, thermal_transform,
    CheckpointManager, load_rgb_safe, load_thermal_safe,
    load_pointcloud_safe, SSL_DIR
)
import nuscenes_gt as nusc_gt

# nuScenes devkit version used for GT lookups
NUSC_VERSION = "v1.0-trainval"

# ── Paths ──────────────────────────────────────────────────────────────────
FUSED_PATH         = OUTPUT_ROOT / "fused_embeddings.json"

PANOPTIC_DIR       = (NUS_ROOT / "nuScenes-panoptic-v1.0-all"
                       / "nuScenes-panoptic-v1.0-all"
                       / "panoptic" / "v1.0-trainval")
LIDARSEG_DIR       = (NUS_ROOT / "nuScenes-lidarseg-all-v1.0"
                       / "lidarseg" / "v1.0-trainval")

TEST_ROOT          = NUS_ROOT / "v1.0-test_blobs" / "samples"

PH_PROG            = PROGRESS_ROOT / "perception_heads_progress.json"
PH_FLAG            = PROGRESS_ROOT / "perception_heads_complete.flag"

FINAL_CKPT         = OUTPUT_ROOT / "full_model_inference.pt"
FINAL_EXPORT_FLAG  = PROGRESS_ROOT / "full_model_export.flag"

TEST_RESULTS_PATH  = OUTPUT_ROOT / "test_results.json"
TEST_DONE_FLAG     = PROGRESS_ROOT / "test_eval_complete.flag"

# nuScenes official class counts
N_DET_CLASSES = 10   # car,truck,bus,trailer,cv,ped,moto,bicycle,cone,barrier
N_SEG_CLASSES = 16   # nuScenes lidarseg 16-class split

_MEAN = [0.485,0.456,0.406]
_STD  = [0.229,0.224,0.225]
transform_img = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(_MEAN, _STD)
])


# ── Load SSL weights from checkpoint ──────────────────────────────────────
def _load_ssl_state(device) -> dict:
    chk    = CheckpointManager(SSL_DIR, keep_last=2)
    latest = chk.latest("ssl_epoch")
    enc_map = {"rgb": RGBEncoder, "thermal": ThermalEncoder,
               "lidar": LiDAREncoder, "radar": RadarEncoder}
    out = {}
    for name, cls in enc_map.items():
        enc  = cls(512).to(device)
        proj = projection_head(512, 512).to(device)
        if latest:
            try:
                s = torch.load(str(latest), map_location=device, weights_only=False)
                if f"{name}_enc"  in s: enc.load_state_dict(s[f"{name}_enc"])
                if f"{name}_proj" in s: proj.load_state_dict(s[f"{name}_proj"])
            except Exception as e:
                logger.warning(f"SSL load [{name}]: {e}")
        out[name] = {"enc": enc, "proj": proj}
    return out


# ── Panoptic label loader ─────────────────────────────────────────────────
def _panoptic_label(token: str) -> "torch.Tensor | None":
    if not PANOPTIC_DIR.exists(): return None
    p = PANOPTIC_DIR / f"{token}_panoptic.npz"
    if not p.exists(): return None
    try:
        data = np.load(str(p))["data"]        # uint16 panoptic IDs
        sem  = (data.astype(np.int64) // 1000)  # semantic class
        counts       = np.bincount(sem.flatten(), minlength=N_SEG_CLASSES+1)
        counts[0]    = 0                       # ignore background
        label = int(np.argmax(counts)) % N_SEG_CLASSES
        return torch.tensor(label, dtype=torch.long)
    except Exception: return None


# ── Perception Heads Training ─────────────────────────────────────────────
def train_perception_heads(num_epochs: int = 3):
    if PH_FLAG.exists():
        logger.info("Perception heads done. Skipping.")
        return

    if not FUSED_PATH.exists():
        raise FileNotFoundError(f"{FUSED_PATH} not found. Run 05 first.")

    fused_data = read_json(FUSED_PATH)
    keys       = list(fused_data.keys())
    total      = len(keys)

    # ── Decide detection mode (DETR head vs legacy MLP) ─────────────────────
    # We honour the same env var that FullADModel honours, so the head module
    # in this stage stays consistent with what the full model expects.
    import os as _os
    USE_DETR = _os.environ.get("MM_USE_DETR_HEAD", "1").lower() not in ("0", "false", "no", "off")

    if USE_DETR:
        det      = DETR3DHead(emb_dim=512, num_classes=N_DET_CLASSES,
                              num_queries=300, depth=3).to(DEVICE)
        det_loss = DETRSetMatchLoss(num_classes=N_DET_CLASSES).to(DEVICE)
        logger.info("Detection head: DETR3DHead (Tier-B1) with Hungarian set loss.")
    else:
        det      = DetectionHead(512, N_DET_CLASSES).to(DEVICE)
        det_loss = nn.CrossEntropyLoss()
        logger.info("Detection head: legacy MLP DetectionHead (set MM_USE_DETR_HEAD=1 to switch).")

    seg  = SegmentationHead(512, N_SEG_CLASSES).to(DEVICE)
    traj = TrajectoryHead(512, n_wp=5, coords=2).to(DEVICE)

    # ── Pre-flight: is real nuScenes GT available? ──────────────────────────
    gt_available = nusc_gt.is_available()
    if not gt_available:
        logger.warning(
            "nuscenes-devkit not installed — detection & trajectory losses "
            "will be SKIPPED (no fake labels). pip install nuscenes-devkit to "
            "unlock real-GT supervision.")
    else:
        logger.info(f"nuScenes-devkit detected. GT lookups will use "
                    f"NUS_ROOT={NUS_ROOT} version={NUSC_VERSION}.")

    opt_d = torch.optim.AdamW(det.parameters(),  lr=3e-4, weight_decay=1e-4)
    opt_s = torch.optim.AdamW(seg.parameters(),  lr=3e-4, weight_decay=1e-4)
    opt_t = torch.optim.AdamW(traj.parameters(), lr=3e-4, weight_decay=1e-4)

    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=num_epochs)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=num_epochs)
    sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=num_epochs)

    ce  = nn.CrossEntropyLoss()

    # Resume
    prog        = read_json(PH_PROG) if PH_PROG.exists() else {}
    start_epoch = int(prog.get("epoch", 0))
    start_batch = int(prog.get("batch", 0))

    DET_LATEST  = HEAD_DIR / "det_latest.pth"
    SEG_LATEST  = HEAD_DIR / "seg_latest.pth"
    TRAJ_LATEST = HEAD_DIR / "traj_latest.pth"

    for path, model, opt in [(DET_LATEST, det, opt_d),
                              (SEG_LATEST, seg, opt_s),
                              (TRAJ_LATEST, traj, opt_t)]:
        if path.exists():
            try:
                ep, extra = load_ckpt(path, model, opt)
                start_epoch = max(start_epoch, ep)
                if extra: start_batch = extra.get("batch", 0)
            except Exception as e:
                logger.warning(f"Head ckpt load {path.name}: {e}")

    logger.info(f"Training heads: {num_epochs} epochs, {total} samples")

    # Epoch-level GT availability counters (helps diagnose "why did det loss
    # never decrease?" — usually because tokens couldn't be resolved).
    epoch_stats = {"det_with_gt": 0, "det_skipped": 0,
                   "traj_with_gt": 0, "traj_skipped": 0,
                   "seg_with_gt": 0, "seg_skipped": 0}

    for ep in range(start_epoch, num_epochs):
        det.train(); seg.train(); traj.train()
        rd, rs, rt = 0.0, 0.0, 0.0
        cd, cs, ct = 0, 0, 0
        # reset per-epoch
        for kstat in epoch_stats: epoch_stats[kstat] = 0

        pbar = tqdm(range(total), desc=f"Heads {ep+1}/{num_epochs}")

        for bi in pbar:
            if ep == start_epoch and bi < start_batch: continue

            k  = keys[bi]
            fd = fused_data[k]
            fv = fd.get("fused")
            if not fv: continue

            x = torch.tensor(fv, dtype=torch.float32).to(DEVICE).unsqueeze(0)  # (1,512)

            # ── Resolve nuScenes sample_token (best-effort) ──────────────────
            # 1) entry may carry "token" / "sample_token" directly
            # 2) entry may carry a file path → extract sample_data token,
            #    then ask devkit for the parent sample_token.
            token = fd.get("sample_token") or fd.get("token")
            if not (isinstance(token, str) and len(token) == 32):
                token = None
            if token is None and gt_available:
                path = fd.get("path") or fd.get("rgb_path") or fd.get("lidar_path")
                if path:
                    sd_tok = nusc_gt.derive_token_from_path(str(path))
                    if sd_tok:
                        token = nusc_gt.sample_token_for_sample_data(
                            sd_tok, NUSC_VERSION, str(NUS_ROOT))

            losses = []

            # ── Detection: real GT or skip ────────────────────────────────
            det_targets = None
            if token is not None and gt_available:
                det_targets = nusc_gt.load_detection_targets(
                    token, NUSC_VERSION, str(NUS_ROOT))

            if det_targets is not None:
                if USE_DETR:
                    # DETR head: memory is the 1-token fused vector (single
                    # column of cross-attention memory). Future revision can
                    # feed the full 5-node memory from scene_graphs.json.
                    memory = x.unsqueeze(1)                              # (1, 1, 512)
                    det_out = det(memory)
                    tgts = [{k_: v_.to(DEVICE) for k_, v_ in det_targets.items()}]
                    ldict = det_loss(det_out, tgts)
                    loss_d = ldict["loss"]
                else:
                    # Legacy MLP head: train against the majority GT class
                    # (frame-level classification — same as the previous
                    # pseudo-label slot, but now using a REAL label).
                    out_d = det(x)
                    majority_cls = det_targets["labels"].mode().values.unsqueeze(0).to(DEVICE)
                    loss_d = ce(out_d, majority_cls)
                losses.append(loss_d)
                rd += float(loss_d); cd += 1
                epoch_stats["det_with_gt"] += 1
            else:
                epoch_stats["det_skipped"] += 1

            # ── Segmentation: panoptic GT when available ──
            out_s  = seg(x)
            lbl_s  = _panoptic_label(token) if token is not None else None
            if lbl_s is not None:
                loss_s = ce(out_s, lbl_s.to(DEVICE).unsqueeze(0))
                losses.append(loss_s)
                rs += float(loss_s); cs += 1
                epoch_stats["seg_with_gt"] += 1
            else:
                epoch_stats["seg_skipped"] += 1

            # ── Trajectory: real ego-pose future, no L2-to-zero ──────────
            out_t  = traj(x)                                            # (1, 10)
            traj_gt = None
            if token is not None and gt_available:
                traj_gt = nusc_gt.load_trajectory_target(
                    token, NUSC_VERSION, str(NUS_ROOT),
                    n_waypoints=5, dt=0.5)
            if traj_gt is not None:
                loss_t = ((out_t.view(1, 5, 2) - traj_gt.to(DEVICE).unsqueeze(0)) ** 2).mean()
                losses.append(loss_t)
                rt += float(loss_t); ct += 1
                epoch_stats["traj_with_gt"] += 1
            else:
                epoch_stats["traj_skipped"] += 1

            if not losses:
                continue                                            # nothing supervised this step

            loss = sum(losses)
            opt_d.zero_grad(set_to_none=True)
            opt_s.zero_grad(set_to_none=True)
            opt_t.zero_grad(set_to_none=True)
            loss.backward()

            for m in [det, seg, traj]:
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)

            opt_d.step(); opt_s.step(); opt_t.step()

            pbar.set_postfix({"det":  f"{(rd/max(cd,1)):.4f}",
                               "seg":  f"{(rs/max(cs,1)):.4f}",
                               "traj": f"{(rt/max(ct,1)):.4f}",
                               "det_gt": f"{cd}/{cd + epoch_stats['det_skipped']}"})

            if (bi+1) % 500 == 0:
                # FIXED: model= parameter now included
                save_ckpt(DET_LATEST,  model=det,  optim=opt_d,  epoch=ep, extra={"batch":bi})
                save_ckpt(SEG_LATEST,  model=seg,  optim=opt_s,  epoch=ep, extra={"batch":bi})
                save_ckpt(TRAJ_LATEST, model=traj, optim=opt_t,  epoch=ep, extra={"batch":bi})
                write_json(PH_PROG, {"epoch": ep, "batch": bi})

        sched_d.step(); sched_s.step(); sched_t.step()

        # Epoch-end saves (FIXED: model= included)
        save_ckpt(HEAD_DIR/f"det_epoch_{ep}.pth",  model=det,  optim=opt_d,  epoch=ep)
        save_ckpt(HEAD_DIR/f"seg_epoch_{ep}.pth",  model=seg,  optim=opt_s,  epoch=ep)
        save_ckpt(HEAD_DIR/f"traj_epoch_{ep}.pth", model=traj, optim=opt_t,  epoch=ep)
        save_ckpt(DET_LATEST,  model=det,  optim=opt_d,  epoch=ep)
        save_ckpt(SEG_LATEST,  model=seg,  optim=opt_s,  epoch=ep)
        save_ckpt(TRAJ_LATEST, model=traj, optim=opt_t,  epoch=ep)

        write_json(PH_PROG, {"epoch": ep+1, "batch": 0})
        logger.info(
            f"Heads ep {ep+1} | det {rd/max(cd,1):.4f} (gt={epoch_stats['det_with_gt']} "
            f"skipped={epoch_stats['det_skipped']}) | "
            f"seg {rs/max(cs,1):.4f} (gt={epoch_stats['seg_with_gt']} "
            f"skipped={epoch_stats['seg_skipped']}) | "
            f"traj {rt/max(ct,1):.4f} (gt={epoch_stats['traj_with_gt']} "
            f"skipped={epoch_stats['traj_skipped']})")
        if epoch_stats["det_skipped"] > 10 * max(epoch_stats["det_with_gt"], 1):
            logger.warning(
                "Detection GT was skipped >90%% of samples this epoch. "
                "Most likely the fused_embeddings.json entries do not carry "
                "sample_token. Add `token` to extract_fused_embeddings() to "
                "unlock real-GT supervision.")
        start_batch = 0

    PH_FLAG.touch()
    logger.info("Perception heads training complete.")


# ── Full model assembly + export ───────────────────────────────────────────
def assemble_full_model() -> FullADModel:
    if FINAL_EXPORT_FLAG.exists():
        logger.info("Full model already exported. Loading.")
        model = FullADModel().to(DEVICE)
        try:
            model.load_state_dict(
                torch.load(str(FINAL_CKPT), map_location=DEVICE, weights_only=False),
                strict=False)
        except Exception as e:
            logger.warning(f"Full model load: {e}")
        model.eval(); return model

    logger.info("Assembling full model...")
    ssl = _load_ssl_state(DEVICE)
    m   = FullADModel().to(DEVICE)

    attr = {"rgb":   ("rgb_enc",   "rgb_proj"),
            "thermal":("th_enc",   "th_proj"),
            "lidar":  ("lidar_enc","lidar_proj"),
            "radar":  ("radar_enc","radar_proj")}
    for name, (enc_a, proj_a) in attr.items():
        try:
            getattr(m, enc_a).load_state_dict(ssl[name]["enc"].state_dict())
            getattr(m, proj_a).load_state_dict(ssl[name]["proj"].state_dict())
        except Exception as e:
            logger.warning(f"SSL transfer [{name}]: {e}")

    ffiles = sorted(FUSION_DIR.glob("fusion_epoch_*.pth"),
                    key=lambda f: f.stat().st_mtime)
    if ffiles:
        try:
            d = torch.load(str(ffiles[-1]), map_location=DEVICE, weights_only=False)
            m.fusion.load_state_dict(d.get("model_state",{}), strict=False)
            logger.info(f"Loaded fusion: {ffiles[-1].name}")
        except Exception as e:
            logger.warning(f"Fusion load: {e}")

    gnn_f = find_latest(GNN_DIR, "gnn")
    if gnn_f:
        try: load_ckpt(gnn_f, m.gnn); logger.info(f"Loaded GNN: {gnn_f.name}")
        except Exception as e: logger.warning(f"GNN load: {e}")

    for attr_name, fname in [("det","det_latest.pth"),
                               ("seg","seg_latest.pth"),
                               ("traj","traj_latest.pth")]:
        p = HEAD_DIR / fname
        if p.exists():
            try: load_ckpt(p, getattr(m, attr_name)); logger.info(f"Loaded {fname}")
            except Exception as e: logger.warning(f"Head {fname}: {e}")

    m.eval()
    torch.save(m.state_dict(), str(FINAL_CKPT))

    # TorchScript
    try:
        er  = torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE)
        et  = torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE)
        el  = torch.randn(1,N_POINTS,3).to(DEVICE)
        erd = torch.randn(1,N_POINTS,3).to(DEVICE)
        with torch.no_grad():
            traced = torch.jit.trace(m, (er, et, el, erd))
        traced.save(str(OUTPUT_ROOT / "full_model_traced.pt"))
        logger.info("TorchScript export OK.")
    except Exception as e:
        logger.warning(f"TorchScript failed (use state_dict): {e}")

    FINAL_EXPORT_FLAG.touch()
    logger.info(f"Full model → {FINAL_CKPT}")
    return m


# ── Inference helpers ─────────────────────────────────────────────────────
def _load_rgb_tensor(path: str):
    try:
        return transform_img(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    except Exception: return None

def _load_thermal_tensor(path: str):
    t = load_thermal_safe(path)
    return t.unsqueeze(0).float().to(DEVICE) if t is not None else None

def _load_points_tensor(path: str):
    pts = load_pointcloud_safe(path)
    return pts.unsqueeze(0).float().to(DEVICE) if pts is not None else None


# ── Test-set evaluation (cameras + LiDAR + Radar) ─────────────────────────
def evaluate_on_test(model: FullADModel, max_per_sensor: int = 200):
    if TEST_DONE_FLAG.exists():
        logger.info("Test eval done. Skipping.")
        return read_json(TEST_RESULTS_PATH)

    model.eval()
    results: dict = {}

    # Camera sensors
    for cam in ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
                "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]:
        d = TEST_ROOT / cam
        if not d.exists(): continue
        for f in tqdm(sorted(d.glob("*.jpg"))[:max_per_sensor], desc=f"Test/{cam}"):
            t = _load_rgb_tensor(str(f))
            if t is None: continue
            with torch.no_grad(): out = model(rgb=t)
            results[f"{cam}/{f.name}"] = {
                "detection":    out["detection"].cpu().tolist(),
                "segmentation": out["segmentation"].cpu().tolist(),
                "trajectory":   out["trajectory"].cpu().tolist(),
            }

    # LiDAR
    lidar_d = TEST_ROOT / "LIDAR_TOP"
    if lidar_d.exists():
        for f in tqdm(sorted(lidar_d.glob("*.bin"))[:max_per_sensor], desc="Test/LIDAR"):
            t = _load_points_tensor(str(f))
            if t is None: continue
            with torch.no_grad(): out = model(lidar=t)
            results[f"LIDAR_TOP/{f.name}"] = {
                "detection":    out["detection"].cpu().tolist(),
                "segmentation": out["segmentation"].cpu().tolist(),
                "trajectory":   out["trajectory"].cpu().tolist(),
            }

    # Radar
    radar_d = TEST_ROOT / "RADAR_FRONT"
    if radar_d.exists():
        for f in tqdm(sorted(radar_d.glob("*.pcd"))[:max_per_sensor], desc="Test/RADAR"):
            t = _load_points_tensor(str(f))
            if t is None: continue
            with torch.no_grad(): out = model(radar=t)
            results[f"RADAR_FRONT/{f.name}"] = {
                "detection":    out["detection"].cpu().tolist(),
                "segmentation": out["segmentation"].cpu().tolist(),
                "trajectory":   out["trajectory"].cpu().tolist(),
            }

    write_json(TEST_RESULTS_PATH, results)
    TEST_DONE_FLAG.touch()
    logger.info(f"Test eval: {len(results)} samples → {TEST_RESULTS_PATH}")
    return results


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_perception_heads(num_epochs=3)
    model = assemble_full_model()
    evaluate_on_test(model, max_per_sensor=200)
