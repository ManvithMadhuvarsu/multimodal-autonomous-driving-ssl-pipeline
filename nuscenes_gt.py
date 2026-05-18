"""
nuscenes_gt.py — real-GT loaders for nuScenes annotations.

Used by perception_heads_and_export.py to replace the previous
hash(k) %% N_CLASSES pseudo-label with actual annotation-based supervision.

If nuscenes-devkit is not installed, every loader returns ``None`` and the
caller must skip the corresponding loss contribution. The pipeline keeps
running, but the heads are not trained against fake labels.

Design notes
------------
* The devkit handle is cached at module load time so we do not pay the
  ~30 s indexing cost per sample.
* All boxes are returned in the **global frame** for downstream simplicity
  (DETR3DHead works in this frame). For per-sample / ego-relative frame,
  apply the inverse ego pose externally.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import torch

try:                                       # pragma: no cover — optional dep
    from nuscenes.nuscenes import NuScenes
    _DEVKIT_AVAILABLE = True
except Exception:
    _DEVKIT_AVAILABLE = False


# nuScenes 10-class detection set (must match DETR3DHead.NUSCENES_CLASS_NAMES).
NUSCENES_CLASS_NAMES: Tuple[str, ...] = (
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
)
_NAME_TO_IDX = {n: i for i, n in enumerate(NUSCENES_CLASS_NAMES)}

# Category prefix map (devkit categories use "vehicle.car", "human.pedestrian", ...)
_CATEGORY_PREFIX = {
    "vehicle.car":               "car",
    "vehicle.truck":             "truck",
    "vehicle.bus":               "bus",
    "vehicle.trailer":           "trailer",
    "vehicle.construction":      "construction_vehicle",
    "human.pedestrian":          "pedestrian",
    "vehicle.motorcycle":        "motorcycle",
    "vehicle.bicycle":           "bicycle",
    "movable_object.trafficcone":"traffic_cone",
    "movable_object.barrier":    "barrier",
}


def _category_to_class_idx(cat: str) -> Optional[int]:
    for prefix, name in _CATEGORY_PREFIX.items():
        if cat.startswith(prefix):
            return _NAME_TO_IDX[name]
    return None


@lru_cache(maxsize=1)
def _get_nuscenes(version: str, dataroot: str):
    if not _DEVKIT_AVAILABLE:
        return None
    try:
        return NuScenes(version=version, dataroot=dataroot, verbose=False)
    except Exception:
        return None


def is_available() -> bool:
    """True if nuscenes-devkit is importable."""
    return _DEVKIT_AVAILABLE


def load_detection_targets(token: str,
                           version: str,
                           dataroot: str
                           ) -> Optional[dict]:
    """
    Return a DETR-style targets dict for the given sample_token, or None if
    the token is unknown / devkit unavailable / no annotations.

        {"labels":   LongTensor(N,),
         "boxes":    FloatTensor(N, 7),   # cx,cy,cz,w,l,h,yaw   (global frame)
         "velocity": FloatTensor(N, 2)}   # vx, vy               (global frame)
    """
    nusc = _get_nuscenes(version, dataroot)
    if nusc is None: return None
    try:
        sample = nusc.get("sample", token)
    except KeyError:
        return None

    labels: List[int]  = []
    boxes:  List[list] = []
    vels:   List[list] = []
    for ann_tok in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        cls_idx = _category_to_class_idx(ann["category_name"])
        if cls_idx is None: continue
        tx, ty, tz = ann["translation"]
        w, l, h    = ann["size"]
        qw, qx, qy, qz = ann["rotation"]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        try:
            vx, vy = nusc.box_velocity(ann_tok)[:2]
            if math.isnan(vx) or math.isnan(vy):
                vx, vy = 0.0, 0.0
        except Exception:
            vx, vy = 0.0, 0.0
        labels.append(cls_idx)
        boxes.append([tx, ty, tz, w, l, h, yaw])
        vels.append([float(vx), float(vy)])

    if not labels:
        return None

    return {
        "labels":   torch.tensor(labels, dtype=torch.long),
        "boxes":    torch.tensor(boxes,  dtype=torch.float32),
        "velocity": torch.tensor(vels,   dtype=torch.float32),
    }


def load_trajectory_target(token: str,
                           version: str,
                           dataroot: str,
                           n_waypoints: int = 5,
                           dt: float = 0.5
                           ) -> Optional[torch.Tensor]:
    """
    Return the ego trajectory for the n_waypoints frames after ``token`` at
    ``dt``-second intervals, expressed as relative XY offsets from the current
    ego pose. Shape (n_waypoints, 2). Returns None if devkit / token / future
    frames are unavailable.
    """
    nusc = _get_nuscenes(version, dataroot)
    if nusc is None: return None
    try:
        sample = nusc.get("sample", token)
    except KeyError:
        return None

    def _ego_xy(samp):
        lidar_sd_tok = samp["data"].get("LIDAR_TOP")
        if not lidar_sd_tok: return None
        sd = nusc.get("sample_data", lidar_sd_tok)
        ego = nusc.get("ego_pose", sd["ego_pose_token"])
        return ego["translation"][0], ego["translation"][1]

    p0 = _ego_xy(sample)
    if p0 is None: return None

    pts = []
    cur = sample
    for _ in range(n_waypoints):
        nxt_tok = cur.get("next", "")
        if not nxt_tok: return None
        cur = nusc.get("sample", nxt_tok)
        p = _ego_xy(cur)
        if p is None: return None
        pts.append([p[0] - p0[0], p[1] - p0[1]])
    return torch.tensor(pts, dtype=torch.float32)              # (n_wp, 2)


def derive_token_from_path(path: str) -> Optional[str]:
    """
    nuScenes sample_data filenames embed the sensor + utime + the
    sample-data token (the 32-char hex blob just before the extension).
    Example:
        /samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg
    The sample_token isn't directly in the filename, but the sample_data
    token IS. This helper extracts that and lets the caller use nusc.get
    ("sample_data", ..)["sample_token"] to get the sample token.
    """
    if not path: return None
    # The token is the 32-char hex string in the filename component before "__"
    # Devkit-formatted filenames may not contain it directly — use as best-effort
    p = path.replace("\\", "/").rsplit("/", 1)[-1]
    p = p.rsplit(".", 1)[0]
    parts = [seg for seg in p.split("__") if len(seg) == 32 and all(c in "0123456789abcdef" for c in seg.lower())]
    return parts[0] if parts else None


def sample_token_for_sample_data(sd_token: str,
                                 version: str, dataroot: str) -> Optional[str]:
    """Look up the sample_token via sample_data_token."""
    nusc = _get_nuscenes(version, dataroot)
    if nusc is None: return None
    try:
        return nusc.get("sample_data", sd_token)["sample_token"]
    except (KeyError, Exception):
        return None
