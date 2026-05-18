"""
Smoke tests for the integrated pipeline.

After the integration commits, the Tier-B modules (DETR head, hetero GNN,
depth-weighted fusion, world-model pretext) live in the main pipeline:

    models.py                                  -> DETR3DHead, HeteroEdgeGNN,
                                                  DepthWeightedFusionTransformer
                                                  + DETRSetMatchLoss, compute_depth_stats
    world_model_pretext.py                     -> WorldModelPretext + Dataset adapter
    quantize_int8.py                           -> quantize_dynamic / static helpers
    nuscenes_gt.py                             -> real-GT loaders (devkit optional)
    evaluators/                                -> the four Tier-A evaluators

This script imports each via its *new* path and verifies a tiny forward pass /
helper call. Run it after any model edit:

    python research_comparison/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

# Make the repo root importable from anywhere
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def ok(msg): print(f"  ok   {msg}")
def fail(msg, exc):
    print(f"  FAIL {msg}\n         {exc}")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Tier-B1: DETR3DHead — now in models.py
# ---------------------------------------------------------------------------
def test_detr_head():
    try:
        from models import DETR3DHead, DETRSetMatchLoss
        head = DETR3DHead(emb_dim=512, num_classes=10, num_queries=32, depth=2)
        mem  = torch.randn(2, 5, 512)
        out  = head(mem)
        assert out["cls_logits"].shape == (2, 32, 11)
        assert out["box_3d"].shape     == (2, 32, 7)
        targets = [
            {"labels":  torch.tensor([0, 5], dtype=torch.long),
             "boxes":   torch.tensor([[1, 2, 0, 4, 2, 1.5, 0.1],
                                      [3, 4, 0, 0.5, 0.5, 1.8, -0.2]]),
             "velocity": torch.tensor([[1.0, 0.2], [0.0, 0.0]])},
            {"labels":  torch.tensor([], dtype=torch.long),
             "boxes":   torch.zeros(0, 7),
             "velocity": torch.zeros(0, 2)},
        ]
        loss = DETRSetMatchLoss(num_classes=10)(out, targets)
        assert torch.isfinite(loss["loss"])
        ok(f"models.DETR3DHead + DETRSetMatchLoss: loss = {float(loss['loss']):.4f}")
        preds = head.predict_nuscenes_format(mem, sample_token="tok123")
        ok(f"DETR3DHead.predict_nuscenes_format: {len(preds)} entries")
    except Exception:
        fail("DETR3DHead", traceback.format_exc())


# ---------------------------------------------------------------------------
# Tier-B2: HeteroEdgeGNN — now in models.py + embedding_and_fusion.py
# ---------------------------------------------------------------------------
def test_hetero_gnn():
    try:
        from models import HeteroEdgeGNN
        from embedding_and_fusion import build_typed_edges, typed_edges_to_tensor
        x = torch.randn(5, 512)
        typed = {
            "bev":        torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
            "doppler":    torch.tensor([[3], [4]], dtype=torch.long),
            "relational": torch.tensor([[0, 4], [3, 0]], dtype=torch.long),
        }
        gnn = HeteroEdgeGNN(in_dim=512, hid=256, num_layers=3,
                            edge_types=("bev", "doppler", "relational"))
        out = gnn(x, typed)
        assert out.shape == (256,) and torch.isfinite(out).all()
        ok(f"models.HeteroEdgeGNN: graph_emb {tuple(out.shape)}")

        class _DummyScorer:
            def __call__(self, vi, vj): return torch.tensor(0.6)
        nodes = {f"n{i}": list(np.random.randn(512)) for i in range(4)}
        typed_dict = build_typed_edges(nodes, _DummyScorer())
        typed_t = typed_edges_to_tensor(typed_dict)
        ok(f"embedding_and_fusion.build_typed_edges: "
           f"bev={typed_t['bev'].shape[1]} relational={typed_t['relational'].shape[1]}")
    except Exception:
        fail("HeteroEdgeGNN", traceback.format_exc())


# ---------------------------------------------------------------------------
# Tier-B4: DepthWeightedFusionTransformer — now in models.py
# ---------------------------------------------------------------------------
def test_depth_weighted_fusion():
    try:
        from models import DepthWeightedFusionTransformer, compute_depth_stats
        fusion = DepthWeightedFusionTransformer(emb_dim=512, depth=2, heads=8)
        rgb, th = torch.randn(2, 512), torch.randn(2, 512)
        lidar, radar = torch.randn(2, 512), torch.randn(2, 512)
        stats = torch.tensor([[1.0, 10.0, 30.0, 80.0], [0.5, 5.0, 12.0, 30.0]])
        out = fusion(rgb, th, lidar, radar, depth_stats=stats)
        assert out.shape == (2, 512) and torch.isfinite(out).all()
        ok(f"models.DepthWeightedFusionTransformer (with stats): {tuple(out.shape)}")
        out2 = fusion(rgb, th, lidar, radar)
        assert torch.isfinite(out2).all()
        ok("models.DepthWeightedFusionTransformer (no-stats fallback) OK")
        pts = torch.randn(2, 1024, 3) * 10
        assert compute_depth_stats(pts).shape == (2, 4)
        ok("models.compute_depth_stats OK")
    except Exception:
        fail("DepthWeightedFusionTransformer", traceback.format_exc())


# ---------------------------------------------------------------------------
# Tier-B6: FullADModel — verifies all three modules are wired together
# ---------------------------------------------------------------------------
def test_full_ad_model():
    try:
        from models import FullADModel, DETR3DHead, HeteroEdgeGNN, DepthWeightedFusionTransformer
        fad = FullADModel(use_detr_head=True, use_hetero_gnn=True, use_depth_fusion=True)
        fad.eval()
        with torch.no_grad():
            out = fad(rgb=torch.randn(1, 3, 224, 224))
        assert isinstance(out["detection"], dict), "detection must be DETR dict"
        assert out["detection"]["cls_logits"].shape == (1, 300, 11)
        assert out["fused"].shape == (512,)
        assert out["gnn"].shape == (256,)
        assert isinstance(fad.fusion, DepthWeightedFusionTransformer)
        assert isinstance(fad.gnn,    HeteroEdgeGNN)
        assert isinstance(fad.det,    DETR3DHead)
        ok("FullADModel wires DepthWeightedFusion + HeteroEdgeGNN + DETR3DHead")
    except Exception:
        fail("FullADModel", traceback.format_exc())


# ---------------------------------------------------------------------------
# Tier-B3: world_model_pretext — top-level module
# ---------------------------------------------------------------------------
def test_world_model():
    try:
        from world_model_pretext import WorldModelPretext, FusedSequenceDataset
        m = WorldModelPretext(dim=512, future_frames=3, depth=2, heads=8)
        past   = torch.randn(4, 2, 512)
        future = torch.randn(4, 3, 512)
        out = m(past, future)
        assert torch.isfinite(out["loss"])
        ok(f"world_model_pretext.forward: loss = {float(out['loss']):.4f}")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "fused.json"
            data = {str(i): {"fused": np.random.randn(512).tolist()} for i in range(20)}
            json.dump(data, open(tmp, "w"))
            ds = FusedSequenceDataset(str(tmp), past=2, future=3)
            ok(f"FusedSequenceDataset windows = {len(ds)}")
    except Exception:
        fail("world_model_pretext", traceback.format_exc())


# ---------------------------------------------------------------------------
# Tier-B5: quantize_int8 — top-level module
# ---------------------------------------------------------------------------
def test_quantize():
    try:
        from quantize_int8 import quantize_dynamic
        mlp = torch.nn.Sequential(torch.nn.Linear(64, 64),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(64, 10))
        q = quantize_dynamic(mlp)
        y = q(torch.randn(2, 64))
        assert y.shape == (2, 10)
        ok(f"quantize_int8.quantize_dynamic: out {tuple(y.shape)}")
    except Exception:
        fail("quantize_int8", traceback.format_exc())


# ---------------------------------------------------------------------------
# Tier-A1..A4 evaluators — top-level evaluators/
# ---------------------------------------------------------------------------
def test_planning():
    try:
        from evaluators.eval_planning_uniad import evaluate
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pred = {f"tok{i}": {"trajectory": list(np.random.randn(10))} for i in range(50)}
            gt = {}
            for i in range(50):
                gt[f"tok{i}"] = {
                    "gt_trajectory": np.random.randn(6, 2).tolist(),
                    "gt_boxes": [{"translation":[100, 100, 0], "size":[2, 4, 1.5],
                                  "rotation":[1, 0, 0, 0]}],
                }
            json.dump(pred, open(td / "p.json", "w"))
            json.dump(gt,   open(td / "g.json", "w"))
            summary = evaluate(str(td / "p.json"), str(td / "g.json"), str(td / "out.json"))
            assert summary["matched_samples"] > 0
            ok(f"evaluators.eval_planning_uniad: L2_avg = {summary['L2_avg']:.3f}, "
               f"collision = {summary['collision_pct']:.2f}%")
    except Exception:
        fail("eval_planning_uniad", traceback.format_exc())


def test_adverse_weather():
    try:
        from evaluators.eval_adverse_weather import evaluate
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            preds, gts = {}, {}
            conds = ("clear_day", "clear_night", "fog", "snow")
            np.random.seed(0)
            for i, cond in enumerate(conds * 5):
                sid = f"s{i}"
                objs = []
                for cls in ("car", "pedestrian", "cyclist"):
                    for r in (15, 40, 65):
                        objs.append({"class": cls, "bbox_3d": [r, 0, 0, 2, 4, 1.5, 0.0]})
                gts[sid] = {"condition": cond, "objects": objs}
                pl = []
                for cls in ("car", "pedestrian", "cyclist"):
                    for r in (15, 40, 65):
                        pl.append({"class": cls,
                                   "bbox_3d": [r + np.random.randn() * 0.1,
                                               np.random.randn() * 0.1, 0, 2, 4, 1.5, 0.0],
                                   "score": float(0.6 + 0.3 * np.random.rand())})
                preds[sid] = pl
            json.dump(preds, open(td / "p.json", "w"))
            json.dump(gts,   open(td / "g.json", "w"))
            tbl = evaluate(str(td / "p.json"), str(td / "g.json"), str(td / "out.json"))
            assert tbl["fog"]["far"]["pedestrian"] is not None
            ok("evaluators.eval_adverse_weather: table populated")
    except Exception:
        fail("eval_adverse_weather", traceback.format_exc())


def test_nuscenes_official():
    try:
        from evaluators.eval_nuscenes_official import wrap_official_format
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            flat = {"toka": [
                {"sample_token": "toka", "translation": [0, 0, 0], "size": [2, 4, 1.5],
                 "rotation": [1, 0, 0, 0], "velocity": [0, 0], "detection_name": "car",
                 "detection_score": 0.9, "attribute_name": ""}
            ]}
            json.dump(flat, open(td / "flat.json", "w"))
            wrap_official_format(str(td / "flat.json"), str(td / "sub.json"))
            sub = json.load(open(td / "sub.json"))
            assert "meta" in sub and "results" in sub
            ok("evaluators.eval_nuscenes_official.wrap_official_format OK")
    except Exception:
        fail("eval_nuscenes_official", traceback.format_exc())


def test_multi_seed():
    try:
        from evaluators.run_multi_seed import collect_metrics, aggregate, print_table
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            seeds = [42, 123]
            for s in seeds:
                root = td / f"seed_{s}"; root.mkdir()
                json.dump({"L2_avg": 0.71 + 0.01 * (s % 5),
                           "collision_pct": 0.18 + 0.005 * (s % 7)},
                          open(root / "planning_uniad.json", "w"))
                json.dump({"mAP": 0.59 + 0.001 * s, "mIoU": 52.5},
                          open(root / "aggregated_metrics.json", "w"))
            seed_metrics = {s: collect_metrics(td / f"seed_{s}") for s in seeds}
            agg = aggregate(seed_metrics)
            print_table(agg, seeds)
            assert any(k.endswith("L2_avg") for k in agg)
            ok("evaluators.run_multi_seed.aggregate OK")
    except Exception:
        fail("run_multi_seed", traceback.format_exc())


# ---------------------------------------------------------------------------
# nuscenes_gt — module presence (devkit is optional)
# ---------------------------------------------------------------------------
def test_nuscenes_gt():
    try:
        import nuscenes_gt
        avail = nuscenes_gt.is_available()
        ok(f"nuscenes_gt importable (devkit_available={avail})")
    except Exception:
        fail("nuscenes_gt", traceback.format_exc())


def main():
    print("Smoke-testing integrated pipeline...")
    test_detr_head()
    test_hetero_gnn()
    test_depth_weighted_fusion()
    test_full_ad_model()
    test_world_model()
    test_quantize()
    test_planning()
    test_adverse_weather()
    test_nuscenes_official()
    test_multi_seed()
    test_nuscenes_gt()
    print()
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
