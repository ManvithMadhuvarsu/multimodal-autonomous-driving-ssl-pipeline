# RUNBOOK — integrated state (post-merge)

The Tier-A evaluators and Tier-B architectural improvements are **now active
in the main pipeline**. This file documents what changed, where to find each
piece, and the exact commands to produce the IEEE-grade table from the
repository as it currently stands.

> Smoke-test confirmation: `python research_comparison/smoke_test.py` →
> all 14 modules green, including the integrated FullADModel.

---

## 1. What changed in the main pipeline

| Component | Before | After |
|---|---|---|
| Detection head | `DetectionHead` MLP → 10-class logits per frame | `DETR3DHead` → per-query (class, 7-DoF box, velocity) in nuScenes format. Hungarian set loss. Toggle via `MM_USE_DETR_HEAD=0`. |
| GNN | `GNNEncoder` (homogeneous GraphSAGE, hidden 256) | `HeteroEdgeGNN` (per-edge-type attention + PairNorm). Edge types: bev / doppler / relational. Toggle via `MM_USE_HETERO_GNN=0` or `GNN_TYPE=sage`. |
| Fusion | `MultiModalFusionTransformer` (uniform attention) | `DepthWeightedFusionTransformer` (per-bucket modality-attention bias from depth statistics). Toggle via `MM_USE_DEPTH_FUSION=0`. |
| Detection loss | `hash(k) % 10` pseudo-label + `CrossEntropyLoss` | Hungarian set loss against real nuScenes GT (devkit). Skipped (no fake label) when GT unresolvable. |
| Trajectory loss | `(out_t ** 2).mean() * 0.01` (L2-to-zero "regulariser") | Smooth-L2 against ego-pose deltas from devkit. Skipped when GT unresolvable. |
| Pretraining | NT-Xent + BYOL only | + 4D world-model pretext stage (`world_model_pretext.py`, stage 6). |
| Deployment | FP32 export only | + optional INT8 quantization stage (`quantize_int8.py`, stage 14). |
| Seeding | `set_seed(42)` hard-coded | Honours `MULTIMODAL_SSL_SEED` env var (used by `run_multi_seed.py`). |
| Evaluators | none | `evaluators/` with planning / adverse-weather / nuScenes-official / multi-seed CLIs. |

The legacy modules (`DetectionHead`, `GNNEncoder`, `MultiModalFusionTransformer`)
are preserved for ablations; flip the env vars to switch back.

---

## 2. New repo layout

```
multimodal-ssl-ad/
├── README.md
├── requirements.txt
├── run_pipeline.py          # 15 stages now (was 12)
├── setup.py                 # honours MULTIMODAL_SSL_SEED
├── dataset_indexing.py
├── ssl_training.py
├── embedding_and_fusion.py  # writes edges_by_type in scene_graphs.json
├── gnn_training.py          # GNN_TYPE env var: sage | hetero
├── perception_heads_and_export.py  # DETR head + real GT, no pseudo-labels
├── models.py                # DETR3DHead + HeteroEdgeGNN + DepthWeightedFusion
├── nuscenes_gt.py           # real-GT loaders (devkit optional)
├── world_model_pretext.py   # NEW Tier-B3 stage
├── quantize_int8.py         # NEW Tier-B5 stage
├── rl_agent.py
├── inference_pipeline.py
├── evaluators/              # NEW — Tier-A
│   ├── eval_planning_uniad.py
│   ├── eval_adverse_weather.py
│   ├── eval_nuscenes_official.py
│   └── run_multi_seed.py
└── research_comparison/     # 2026 SOTA comparison + analysis (this dir)
    ├── README.md
    ├── COMPARISON.md
    ├── IMPROVEMENTS.md
    ├── RUNBOOK.md           # ← you are here
    ├── data.py
    ├── generate_gifs.py
    ├── smoke_test.py        # tests the *integrated* pipeline
    └── gifs/                # 7 animated comparison artifacts
```

---

## 3. Command sequence to produce the IEEE-grade table

Assumes nuScenes is on disk under `D:/Mtech/Sem_3/Case Study/Data/NuScences/`
and `pip install nuscenes-devkit` has been run.

### 3a. Train the upgraded pipeline

```bash
# defaults: Tier-B1, B2, B4 all on; world-model + INT8 enabled via flags
python run_pipeline.py --quantize-int8

# to run a specific seed for the multi-seed table
MULTIMODAL_SSL_SEED=123 python run_pipeline.py
```

### 3b. Produce planning numbers (Tier-A1)

```bash
# build GT once
python -m evaluators.eval_planning_uniad build-gt \
    --nuscenes-root "D:/Mtech/Sem_3/Case Study/Data/NuScences/v1.0-trainval" \
    --out D:/Mtech/Sem_4/output/gt_planning.json

# evaluate
python -m evaluators.eval_planning_uniad eval \
    --pred-json D:/Mtech/Sem_4/output/test_results.json \
    --gt-json   D:/Mtech/Sem_4/output/gt_planning.json \
    --out       D:/Mtech/Sem_4/output/planning_uniad.json
```

### 3c. Produce official nuScenes detection numbers (Tier-A3)

```bash
# Generate per-sample DETR predictions:
#   for token in val_split:
#       preds[token] = full_model.det.predict_nuscenes_format(memory, token)
# Then:
python -m evaluators.eval_nuscenes_official \
    --pred-json     D:/Mtech/Sem_4/output/detr_predictions.json \
    --nuscenes-root "D:/Mtech/Sem_3/Case Study/Data/NuScences/v1.0-trainval" \
    --version        v1.0-trainval \
    --eval-split     val \
    --out-dir        D:/Mtech/Sem_4/output/nuscenes_eval
```

### 3d. Produce adverse-weather numbers (Tier-A2)

Download SeeingThroughFog from Princeton CIL; build `stf_gt.json` from STF
metadata; then:

```bash
python -m evaluators.eval_adverse_weather \
    --pred-json D:/Mtech/Sem_4/output/stf_predictions.json \
    --gt-json   D:/Mtech/Sem_4/output/stf_gt.json \
    --out       D:/Mtech/Sem_4/output/adverse_weather.json
```

### 3e. Multi-seed mean ± std (Tier-A4)

```bash
python -m evaluators.run_multi_seed \
    --seeds 42 123 7 \
    --pipeline-cmd "python run_pipeline.py --skip-rl --quantize-int8" \
    --output-root  D:/Mtech/Sem_4/output \
    --results-root D:/Mtech/Sem_4/output/multi_seed
```

### 3f. Refresh the comparison GIFs with measured numbers

After 3b/3c/3d/3e have produced real JSON outputs:

```python
# In research_comparison/data.py, in the OURS record:
nds          = <metrics_summary.json["nd_score"]>
l2_avg       = <planning_uniad.json["L2_avg"]>
collision_pct= <planning_uniad.json["collision_pct"]>
# fill weather_ap from adverse_weather.json

# Then:
python research_comparison/generate_gifs.py
```

The "estimated" / "interpolated" stamps drop off automatically for any cell
backed by a measured number; you only need to remove the † footnote in
`gif_weather_robustness()` once the SeeingThroughFog numbers are in.

---

## 4. Honesty notes (still apply)

The pipeline now refuses to invent labels:

* **Detection head** — when GT can't be resolved (token missing, devkit
  not installed), the detection loss is **skipped for that sample** and the
  per-epoch log reports `gt_with_gt=N / skipped=M`. If skipped > 90 %, the
  warning suggests extending `extract_fused_embeddings()` to preserve the
  sample_token. (This was a hidden failure mode before; now it's visible.)
* **Trajectory head** — same thing: no L2-to-zero regulariser. Real ego-pose
  delta loss when available, skipped otherwise.
* **Segmentation head** — same panoptic-NPZ loader as before, but now gated
  on the same resolved token (used to silently fall back to JSON index).

If you want to commit the project to ALWAYS using real labels (no skips
allowed), set `MM_REQUIRE_GT=1` (TODO in a future commit — not yet wired).

---

## 5. Quick reference

| Tier | File | What it does |
|---|---|---|
| B1 | `models.py::DETR3DHead`, `DETRSetMatchLoss` | DETR-style detection head + Hungarian loss |
| B2 | `models.py::HeteroEdgeGNN`; `embedding_and_fusion.py::build_typed_edges` | Edge-type-aware GNN + typed-edge builder |
| B3 | `world_model_pretext.py` | 4D forward-dynamics pretext on fused-embedding sequences |
| B4 | `models.py::DepthWeightedFusionTransformer`, `compute_depth_stats` | Depth-bucket-weighted fusion |
| B5 | `quantize_int8.py` | Dynamic / static INT8 quantization CLI |
| A1 | `evaluators/eval_planning_uniad.py` | UniAD-protocol planning (L2 + collision %) |
| A2 | `evaluators/eval_adverse_weather.py` | SeeingThroughFog-style adverse-weather AP |
| A3 | `evaluators/eval_nuscenes_official.py` | nuScenes-devkit NDS / per-class / error metrics |
| A4 | `evaluators/run_multi_seed.py` | Multi-seed orchestrator + mean ± std table |
| GT | `nuscenes_gt.py` | Real-GT loaders (devkit optional) |
| smoke | `research_comparison/smoke_test.py` | Sanity-check all of the above |
