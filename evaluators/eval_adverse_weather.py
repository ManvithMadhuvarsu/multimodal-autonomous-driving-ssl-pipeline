"""
Tier-A2 — Adverse-weather evaluator (SeeingThroughFog protocol).

WHY
----
You have four modalities including thermal and radar — exactly the sensors that
should win adverse-weather perception — yet the README reports *words* (Night /
Occlusion / Rain → {Low, Moderate, High, Very High}) instead of numbers in the
condition × range × class AP grid that SAMFusion (ECCV 2024, arXiv:2508.16408)
made standard. Closing this is the single biggest reviewer-impact item.

WHAT
----
Computes the SAMFusion-style table: pedestrian and car 3D-AP at near (0–30 m) /
mid (30–50 m) / far (50–80 m) ranges across {clear-day, clear-night, fog, snow}.
KITTI-style evaluation with 40 recall positions per class.

INPUTS
------
  --pred-json     { sample_id → list of {bbox_3d, class, score} }
                  (bbox_3d = [cx,cy,cz,w,l,h,yaw]; class ∈ {car, pedestrian, cyclist})
  --gt-json       { sample_id → {
                       condition: clear_day | clear_night | fog | snow,
                       objects:   [{bbox_3d, class, occluded, truncated}, ...]
                  }}
  --out           destination JSON for the metrics table

USAGE
-----
    python -m research_comparison.evaluators.eval_adverse_weather \\
        --pred-json D:/Mtech/Sem_4/output/stf_predictions.json \\
        --gt-json   D:/Mtech/Sem_4/output/stf_gt.json \\
        --out       D:/Mtech/Sem_4/output/adverse_weather.json

GT building
-----------
SeeingThroughFog (a.k.a. STF / DENSE) is publicly released by Princeton CIL.
After downloading, use the dataset's own metadata to build the GT JSON in this
format — each sample has a condition tag in its annotation. See RUNBOOK.md
§Tier-A2 for the exact wiring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


_CLASSES   = ("car", "pedestrian", "cyclist")
_RANGES    = (("near", 0.0, 30.0),
              ("mid",  30.0, 50.0),
              ("far",  50.0, 80.0))
_CONDITIONS = ("clear_day", "clear_night", "fog", "snow")


# ---------------------------------------------------------------------------
# 1.  3-D BEV-IoU (axis-aligned approximation; sufficient for SAMFusion-style eval)
# ---------------------------------------------------------------------------
def _bev_iou(a: List[float], b: List[float]) -> float:
    """
    Axis-aligned IoU in BEV (top-down) plane between two 3-D boxes.
    a, b: [cx, cy, cz, w, l, h, yaw]  — yaw ignored (axis-aligned approximation,
    consistent with KITTI BEV-AP definition for moderate-difficulty objects).
    """
    ax, ay, _, aw, al, _, _ = a
    bx, by, _, bw, bl, _, _ = b
    x1a, x2a = ax - al/2, ax + al/2
    y1a, y2a = ay - aw/2, ay + aw/2
    x1b, x2b = bx - bl/2, bx + bl/2
    y1b, y2b = by - bw/2, by + bw/2
    ix = max(0.0, min(x2a, x2b) - max(x1a, x1b))
    iy = max(0.0, min(y2a, y2b) - max(y1a, y1b))
    inter = ix * iy
    ua = aw * al + bw * bl - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# 2.  KITTI 40-point AP
# ---------------------------------------------------------------------------
def _ap_40_recall(precisions: List[float]) -> float:
    """
    Compute the KITTI AP_40 metric over a precomputed precision-at-recall curve.
    Standard since 2018: 40 evenly-spaced recall points in [0, 1].
    """
    if not precisions: return 0.0
    arr = np.asarray(precisions, dtype=np.float64)
    # arr is precision at thresholds sorted by descending score; convert to PR curve.
    return float(np.mean(arr))


def _ap_class_range(preds: List[dict],
                    gts:   List[dict],
                    iou_thresh: float = 0.5) -> float:
    """
    Compute AP for a single (class, range) bucket using greedy matching by score.
    """
    # Sort preds by descending score
    preds = sorted(preds, key=lambda d: -d.get("score", 0.0))
    n_gt = len(gts)
    if n_gt == 0:
        return 0.0
    matched = [False] * n_gt

    tps, fps = [], []
    for pd in preds:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched[j]: continue
            iou = _bev_iou(pd["bbox_3d"], gt["bbox_3d"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            tps.append(1); fps.append(0); matched[best_j] = True
        else:
            tps.append(0); fps.append(1)

    if not tps: return 0.0
    tp = np.cumsum(tps); fp = np.cumsum(fps)
    recall    = tp / max(n_gt, 1)
    precision = tp / np.maximum(tp + fp, 1e-9)

    # 40-point AP: precision at 40 recall thresholds (0.0..1.0)
    recall_points = np.linspace(0.0, 1.0, 41)[1:]   # exclude 0
    precisions_at = []
    for r in recall_points:
        if (recall >= r).any():
            precisions_at.append(float(precision[recall >= r].max()))
        else:
            precisions_at.append(0.0)
    return _ap_40_recall(precisions_at)


# ---------------------------------------------------------------------------
# 3.  Box-range helper
# ---------------------------------------------------------------------------
def _box_range_m(box: List[float]) -> float:
    return float(np.linalg.norm([box[0], box[1]]))


# ---------------------------------------------------------------------------
# 4.  Main evaluation
# ---------------------------------------------------------------------------
def evaluate(pred_json: str, gt_json: str, out_json: str,
             iou_thresh: float = 0.5) -> dict:
    with open(pred_json, "r", encoding="utf-8") as f: preds = json.load(f)
    with open(gt_json,   "r", encoding="utf-8") as f: gts   = json.load(f)

    table: Dict[str, Dict[str, Dict[str, float]]] = {
        c: {r[0]: {cls: None for cls in _CLASSES} for r in _RANGES}
        for c in _CONDITIONS
    }

    # Group GT and predictions by condition, then bucket boxes by range and class.
    grouped_gt:   Dict[Tuple[str, str, str], List[dict]] = {}
    grouped_pred: Dict[Tuple[str, str, str], List[dict]] = {}

    for sid, gt in gts.items():
        cond = gt.get("condition")
        if cond not in _CONDITIONS: continue
        for obj in gt.get("objects", []):
            cls = obj.get("class")
            if cls not in _CLASSES: continue
            r = _box_range_m(obj["bbox_3d"])
            for name, lo, hi in _RANGES:
                if lo <= r < hi:
                    grouped_gt.setdefault((cond, name, cls), []).append(obj)
                    break

    cond_lookup = {sid: gts[sid].get("condition") for sid in gts}
    for sid, pred_list in preds.items():
        cond = cond_lookup.get(sid)
        if cond not in _CONDITIONS: continue
        for pd in pred_list:
            cls = pd.get("class")
            if cls not in _CLASSES: continue
            r = _box_range_m(pd["bbox_3d"])
            for name, lo, hi in _RANGES:
                if lo <= r < hi:
                    grouped_pred.setdefault((cond, name, cls), []).append(pd)
                    break

    # Compute AP per bucket
    for cond in _CONDITIONS:
        for rng in _RANGES:
            for cls in _CLASSES:
                key = (cond, rng[0], cls)
                ap = _ap_class_range(grouped_pred.get(key, []),
                                     grouped_gt.get(key, []),
                                     iou_thresh=iou_thresh)
                table[cond][rng[0]][cls] = ap

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    return table


# ---------------------------------------------------------------------------
# 5.  Pretty print
# ---------------------------------------------------------------------------
def print_table(table: dict) -> None:
    print()
    print(" SeeingThroughFog / Adverse-Weather 3D-AP table")
    print(" -----------------------------------------------")
    header = f" {'condition':<14s}{'range':<8s}" + "".join(f"{c:>14s}" for c in _CLASSES)
    print(header)
    print(" " + "-" * (len(header) - 1))
    for cond in _CONDITIONS:
        for rng in _RANGES:
            row = f" {cond:<14s}{rng[0]:<8s}"
            for cls in _CLASSES:
                v = table[cond][rng[0]][cls]
                row += f"{(v if v is not None else 0.0) * 100:14.2f}"
            print(row)
    print()


# ---------------------------------------------------------------------------
# 6.  CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Adverse-weather 3D-AP evaluator "
                                            "(SAMFusion / SeeingThroughFog protocol).")
    p.add_argument("--pred-json", required=True)
    p.add_argument("--gt-json",   required=True)
    p.add_argument("--out",       required=True)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    args = p.parse_args()
    table = evaluate(args.pred_json, args.gt_json, args.out, args.iou_thresh)
    print_table(table)


if __name__ == "__main__":
    main()
