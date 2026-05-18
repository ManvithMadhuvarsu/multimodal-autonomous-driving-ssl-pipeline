"""
Tier-A1 — Planning evaluator in the UniAD open-loop protocol.

WHY
----
Every 2023–2026 planning paper (UniAD CVPR 2023, VAD ICCV 2023, FusionAD 2023,
GraphAD 2024, DriveWorld CVPR 2024, UAD 2024, GenAD ECCV 2024, FSDrive NeurIPS
2025, VLA-World arXiv 2026) reports the *same* nuScenes open-loop planning
table. Without these numbers, your trajectory head's "−34.2 % MSE" line is
unverifiable against the field.

WHAT
----
Computes the standard table:

    L2 @ 1 s    (m)   ↓
    L2 @ 2 s    (m)   ↓
    L2 @ 3 s    (m)   ↓
    L2 avg      (m)   ↓
    Collision %       ↓     — % of predicted trajectories that intersect any GT
                              box at any horizon step

INPUTS
------
Two files (both produced by the existing pipeline plus a small GT wiring):

  --pred-json    JSON: { sample_token → { trajectory: [x1,y1,x2,y2,..] } }
                 The user's existing test_results.json already has a
                 `trajectory` key; pass it directly.
  --gt-json      JSON: { sample_token → {
                       gt_trajectory: [[x0,y0],[x1,y1],...,[x5,y5]],   # 6 wp, 0.5 s each
                       gt_boxes:      [{translation:[x,y,z], size:[w,l,h],
                                        rotation:[q0,q1,q2,q3]}, ...]
                  }}
                 Build this once from nuScenes — helper ``build_gt_planning_json``
                 below uses the official devkit to do it.

OUTPUTS
-------
A JSON written to ``--out`` plus a printed Markdown-style table.

USAGE
-----
    python -m research_comparison.evaluators.eval_planning_uniad \\
        --pred-json D:/Mtech/Sem_4/output/test_results.json \\
        --gt-json   D:/Mtech/Sem_4/output/gt_planning.json \\
        --out       D:/Mtech/Sem_4/output/planning_uniad.json

If ``--gt-json`` does not exist, the script can build it for you:

    python -m research_comparison.evaluators.eval_planning_uniad build-gt \\
        --nuscenes-root D:/Mtech/Sem_3/Case\\ Study/Data/NuScences/v1.0-trainval \\
        --out D:/Mtech/Sem_4/output/gt_planning.json

    (Requires `pip install nuscenes-devkit`.)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# UniAD samples predictions at 0.5 s intervals → 6 waypoints in 3 s.
PLANNING_HORIZON_SEC = (1.0, 2.0, 3.0)
WP_PER_SEC = 2.0
N_WP_GT = int(3.0 * WP_PER_SEC)   # 6


# ---------------------------------------------------------------------------
# 1.  Trajectory resampling
# ---------------------------------------------------------------------------
def _resample_to_gt_horizon(pred_flat: List[float]) -> np.ndarray:
    """
    Take the user's flat trajectory output ([x0,y0,x1,y1,...]) of arbitrary length
    and produce N_WP_GT 2-D points uniformly spaced over the planning horizon.

    Linear interpolation in the index space. If fewer than 2 waypoints in the
    prediction, returns NaNs (sample excluded from L2 stats).
    """
    pts = np.asarray(pred_flat, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return np.full((N_WP_GT, 2), np.nan)
    src_x = np.linspace(0.0, 1.0, pts.shape[0])
    dst_x = np.linspace(0.0, 1.0, N_WP_GT)
    out = np.stack([np.interp(dst_x, src_x, pts[:, 0]),
                    np.interp(dst_x, src_x, pts[:, 1])], axis=1)
    return out


# ---------------------------------------------------------------------------
# 2.  Collision check (axis-aligned proxy)
# ---------------------------------------------------------------------------
def _box_corners_topdown(translation, size, rotation_quat) -> np.ndarray:
    """
    Project the 3-D box to its 2-D top-down footprint. Returns (4, 2) corner array
    in global frame.

    rotation_quat = [w, x, y, z] (nuScenes convention).
    """
    x, y, _ = translation
    w, l, _ = size
    qw, qx, qy, qz = rotation_quat
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    c, s = math.cos(yaw), math.sin(yaw)
    dxdy = np.array([[+l / 2, +w / 2],
                     [+l / 2, -w / 2],
                     [-l / 2, -w / 2],
                     [-l / 2, +w / 2]])
    rot = np.array([[c, -s], [s, c]])
    return (dxdy @ rot.T) + np.array([x, y])


def _point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    # Ray casting; poly is (4, 2)
    n = poly.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > p[1]) != (yj > p[1])) and \
           (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _trajectory_collides(traj_xy: np.ndarray, gt_boxes: List[dict],
                         ego_size=(1.85, 4.08)) -> bool:
    """
    Returns True if the ego footprint at ANY predicted waypoint overlaps
    ANY GT box footprint. This is the UniAD collision-% definition.
    ``ego_size`` defaults to the nuScenes mean (1.85 m × 4.08 m).
    """
    ew, el = ego_size
    for wp in traj_xy:
        if not np.all(np.isfinite(wp)):
            continue
        ego_corners = np.array([[wp[0] + el / 2, wp[1] + ew / 2],
                                [wp[0] + el / 2, wp[1] - ew / 2],
                                [wp[0] - el / 2, wp[1] - ew / 2],
                                [wp[0] - el / 2, wp[1] + ew / 2]])
        for b in gt_boxes:
            corners = _box_corners_topdown(b["translation"], b["size"], b["rotation"])
            # Cheap test: any ego corner inside box, or any box corner inside ego
            if any(_point_in_polygon(c, corners) for c in ego_corners): return True
            ego_poly = ego_corners
            if any(_point_in_polygon(c, ego_poly) for c in corners): return True
    return False


# ---------------------------------------------------------------------------
# 3.  Main evaluation
# ---------------------------------------------------------------------------
def evaluate(pred_json: str, gt_json: str, out_json: str) -> dict:
    with open(pred_json, "r", encoding="utf-8") as f: preds = json.load(f)
    with open(gt_json,   "r", encoding="utf-8") as f: gts   = json.load(f)

    horizon_indices = [int(h * WP_PER_SEC) - 1 for h in PLANNING_HORIZON_SEC]
    l2_at = {h: [] for h in PLANNING_HORIZON_SEC}
    l2_avg = []
    collisions = 0
    matched   = 0

    for token, pd in preds.items():
        if token not in gts:        continue
        if "trajectory" not in pd:  continue
        gt = gts[token]
        gt_traj = np.asarray(gt.get("gt_trajectory", []), dtype=np.float64)
        if gt_traj.shape[0] < N_WP_GT: continue

        pred_traj = _resample_to_gt_horizon(pd["trajectory"])
        if not np.all(np.isfinite(pred_traj)): continue

        d = np.linalg.norm(pred_traj - gt_traj[:N_WP_GT], axis=1)
        for k, h in enumerate(PLANNING_HORIZON_SEC):
            l2_at[h].append(float(d[horizon_indices[k]]))
        l2_avg.append(float(d.mean()))

        if _trajectory_collides(pred_traj, gt.get("gt_boxes", [])):
            collisions += 1
        matched += 1

    summary = {
        "matched_samples":     matched,
        "L2_1s":               float(np.mean(l2_at[1.0])) if l2_at[1.0] else None,
        "L2_2s":               float(np.mean(l2_at[2.0])) if l2_at[2.0] else None,
        "L2_3s":               float(np.mean(l2_at[3.0])) if l2_at[3.0] else None,
        "L2_avg":              float(np.mean(l2_avg))     if l2_avg    else None,
        "collision_pct":       100.0 * collisions / max(matched, 1),
        "horizon_indices":     horizon_indices,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _print_table(summary: dict) -> None:
    def _fmt(v):
        return "-" if v is None else f"{v:.2f} m"
    print()
    print(" Planning (UniAD protocol, nuScenes open-loop)")
    print(" ---------------------------------------------")
    print(f"   matched samples : {summary['matched_samples']}")
    print(f"   L2 @ 1 s        : {_fmt(summary['L2_1s'])}")
    print(f"   L2 @ 2 s        : {_fmt(summary['L2_2s'])}")
    print(f"   L2 @ 3 s        : {_fmt(summary['L2_3s'])}")
    print(f"   L2 avg          : {_fmt(summary['L2_avg'])}")
    print(f"   collision %     : {summary['collision_pct']:.2f}")
    print()


# ---------------------------------------------------------------------------
# 4.  Optional helper — build the GT JSON from nuScenes devkit
# ---------------------------------------------------------------------------
def build_gt_planning_json(nuscenes_root: str, out: str,
                           split: str = "val") -> None:
    """
    Iterate every sample in the nuScenes split, gather:
      • next 6 ego positions at 0.5 s intervals,
      • the GT 3-D boxes visible in that sample (used for collision %).
    """
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as e:
        raise SystemExit("This helper requires nuscenes-devkit. "
                         "pip install nuscenes-devkit") from e

    nusc = NuScenes(version="v1.0-trainval", dataroot=nuscenes_root, verbose=False)
    splits = create_splits_scenes()
    target_scenes = set(splits.get(split, []))
    gt: Dict[str, dict] = {}

    for scene in nusc.scene:
        if scene["name"] not in target_scenes: continue
        sample_token = scene["first_sample_token"]
        # Pre-collect ego poses for the scene
        ego_path: List[Tuple[float, np.ndarray]] = []
        while sample_token:
            sample = nusc.get("sample", sample_token)
            lidar_top = sample["data"].get("LIDAR_TOP")
            if lidar_top:
                sd = nusc.get("sample_data", lidar_top)
                ego = nusc.get("ego_pose", sd["ego_pose_token"])
                ego_path.append((sample["timestamp"] / 1e6,
                                 np.asarray(ego["translation"][:2])))
            sample_token = sample["next"]
        if len(ego_path) < N_WP_GT + 1: continue

        # Sample with stride 1; planning horizon 3 s at 0.5 s = 6 waypoints
        for i in range(len(ego_path) - N_WP_GT):
            t0, p0 = ego_path[i]
            # 6 future positions at 0.5 s
            future_pts = []
            for k in range(1, N_WP_GT + 1):
                tk, pk = ego_path[i + k]
                rel = pk - p0
                future_pts.append([float(rel[0]), float(rel[1])])

            sample_token_i = nusc.scene[
                [s["name"] for s in nusc.scene].index(scene["name"])
            ]["first_sample_token"]
            sample_i = nusc.get("sample", sample_token_i)
            for _ in range(i):
                sample_i = nusc.get("sample", sample_i["next"])

            ann_boxes = []
            for ann_token in sample_i["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                ann_boxes.append({
                    "translation": ann["translation"],
                    "size":        ann["size"],
                    "rotation":    ann["rotation"],
                    "category":    ann["category_name"],
                })

            gt[sample_i["token"]] = {
                "gt_trajectory": future_pts,
                "gt_boxes":      ann_boxes,
            }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    print(f"[gt] wrote {len(gt)} samples -> {out}")


# ---------------------------------------------------------------------------
# 5.  CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="UniAD-protocol planning evaluator.")
    sub = p.add_subparsers(dest="cmd")

    e = sub.add_parser("eval", help="Evaluate predictions against GT.")
    e.add_argument("--pred-json", required=True)
    e.add_argument("--gt-json",   required=True)
    e.add_argument("--out",       required=True)

    b = sub.add_parser("build-gt", help="Build the GT planning JSON from nuScenes.")
    b.add_argument("--nuscenes-root", required=True)
    b.add_argument("--out",           required=True)
    b.add_argument("--split",         default="val")

    args = p.parse_args()

    if args.cmd == "build-gt":
        build_gt_planning_json(args.nuscenes_root, args.out, args.split)
        return

    if args.cmd is None or args.cmd == "eval":
        # default mode: positional eval args via short form
        if not getattr(args, "pred_json", None):
            p.print_help(); return
        summary = evaluate(args.pred_json, args.gt_json, args.out)
        _print_table(summary)


if __name__ == "__main__":
    main()
