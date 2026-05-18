"""
Tier-A3 — nuScenes official 3D detection evaluator (NDS, per-class AP, error metrics).

WHY
----
mAP alone is the bare minimum a reviewer accepts. The nuScenes leaderboard format
is NDS + per-class AP across 10 classes + (mATE, mASE, mAOE, mAVE, mAAE). Without
this table, your detection contribution is unverifiable against BEVFusion (0.696 mAP /
0.721 NDS), IS-Fusion (0.723 / 0.737), SparseLIF (0.730 / 0.746), etc.

WHAT
----
A thin wrapper around the official ``nuscenes.eval.detection.evaluate.DetectionEval``
class. It:

  1. converts your ``DETR3DHead.predict_nuscenes_format`` output (a list of dicts
     per sample) into the official submission JSON format,
  2. invokes the devkit evaluator on the val split,
  3. prints the standard table and writes the result JSON.

USAGE
-----
    python -m research_comparison.evaluators.eval_nuscenes_official \\
        --pred-json    D:/Mtech/Sem_4/output/detr_predictions.json \\
        --nuscenes-root D:/Mtech/Sem_3/Case\\ Study/Data/NuScences/v1.0-trainval \\
        --version       v1.0-trainval \\
        --eval-split    val \\
        --out-dir       D:/Mtech/Sem_4/output/nuscenes_eval

The expected ``pred-json`` is a flat dict::

    { sample_token: [ {translation, size, rotation, velocity,
                       detection_name, detection_score, attribute_name}, ... ] }

Use ``research_comparison/improvements/detr_head.py::DETR3DHead.predict_nuscenes_format``
to build it.

REQUIREMENTS
------------
    pip install nuscenes-devkit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1.  Wrap raw predictions into the official submission file
# ---------------------------------------------------------------------------
def wrap_official_format(raw_pred_json: str, out_path: str) -> None:
    with open(raw_pred_json, "r", encoding="utf-8") as f:
        flat: dict = json.load(f)

    # nuScenes submission schema:
    # { "meta": {use_camera,use_lidar,use_radar,use_map,use_external},
    #   "results": { sample_token: [box,box,...] } }
    submission = {
        "meta": {
            "use_camera":  True,
            "use_lidar":   True,
            "use_radar":   True,
            "use_map":     False,
            "use_external": False,
        },
        "results": flat,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f)
    print(f"[wrap] wrote official submission -> {out_path}")


# ---------------------------------------------------------------------------
# 2.  Run the devkit evaluator
# ---------------------------------------------------------------------------
def run_devkit_eval(submission_json: str,
                    nuscenes_root: str,
                    version: str,
                    eval_split: str,
                    out_dir: str) -> dict:
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import DetectionEval
        from nuscenes.eval.detection.config import config_factory
    except ImportError as e:
        raise SystemExit("This evaluator requires `nuscenes-devkit`. "
                         "pip install nuscenes-devkit") from e

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    nusc   = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
    config = config_factory("detection_cvpr_2019")

    evaluator = DetectionEval(
        nusc,
        config=config,
        result_path=submission_json,
        eval_set=eval_split,
        output_dir=out_dir,
        verbose=True,
    )
    metrics_summary = evaluator.main(plot_examples=0, render_curves=False)
    # The devkit writes metrics_summary.json and metrics_details.json in out_dir.
    return metrics_summary


# ---------------------------------------------------------------------------
# 3.  Pretty-print the leaderboard-style table
# ---------------------------------------------------------------------------
_CLASSES = ("car", "truck", "bus", "trailer", "construction_vehicle",
            "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier")


def print_official_table(metrics: dict) -> None:
    print()
    print(" nuScenes 3D Detection — Official Eval (CVPR-2019 config)")
    print(" ---------------------------------------------------------")
    print(f"   mAP : {metrics.get('mean_ap', 0):.4f}")
    print(f"   NDS : {metrics.get('nd_score', 0):.4f}")
    print()
    print("   Class                       AP")
    label_aps = metrics.get("label_aps", {})
    for cls in _CLASSES:
        aps = label_aps.get(cls, {})
        if not aps:
            print(f"   {cls:<24s}    -")
            continue
        avg = sum(aps.values()) / len(aps)
        print(f"   {cls:<24s}  {avg:.4f}")
    print()
    print("   Error metrics (lower = better)")
    for k, name in [("mean_dist_aps",      "mAP@dist [0.5,1,2,4]"),
                    ("tp_errors",          "")]:
        if k == "mean_dist_aps":
            pass
    for k, name in [("trans_err", "mATE"),
                    ("scale_err", "mASE"),
                    ("orient_err","mAOE"),
                    ("vel_err",   "mAVE"),
                    ("attr_err",  "mAAE")]:
        v = metrics.get("tp_errors", {}).get(k)
        if v is not None:
            print(f"   {name}  {v:.4f}")
    print()


# ---------------------------------------------------------------------------
# 4.  CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="nuScenes official 3D detection evaluator.")
    p.add_argument("--pred-json",     required=True,
                   help="Raw per-sample DETR3DHead output dict.")
    p.add_argument("--nuscenes-root", required=True)
    p.add_argument("--version",       default="v1.0-trainval")
    p.add_argument("--eval-split",    default="val")
    p.add_argument("--out-dir",       required=True)
    args = p.parse_args()

    submission_json = str(Path(args.out_dir) / "submission.json")
    wrap_official_format(args.pred_json, submission_json)
    metrics = run_devkit_eval(submission_json,
                              args.nuscenes_root,
                              args.version, args.eval_split,
                              args.out_dir)
    print_official_table(metrics)
    print(f"[nuscenes-eval] full result in {args.out_dir}/metrics_summary.json")


if __name__ == "__main__":
    main()
