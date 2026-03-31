<div align="center">

# Multimodal-SSL-AD

### A Multimodal Self-Supervised Learning Framework for Scene Understanding in Autonomous Driving

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![IEEE](https://img.shields.io/badge/IEEE-Open%20Journal%20of%20Computer%20Society-blue)](https://doi.org/10.1109/XXXX.2022.1234567)

<img width="2200" height="1100" alt="workflow" src="https://github.com/user-attachments/assets/f23a61f6-8659-45c7-8865-d4286aec0879" />

</div>

---

## Overview

Official implementation of our multimodal self-supervised learning (SSL) framework for autonomous driving (AD) perception. The system jointly encodes **RGB, Thermal, LiDAR, and Radar** sensor streams without manual annotations, fuses them through a transformer-based architecture, and applies graph neural network reasoning to produce context-aware scene representations.

**Key results at a glance:**

| Metric | Value |
|---|---|
| mAP (100% fine-tune, nuScenes) | **0.597** |
| mIoU (zero-label linear probe) | **46.3** (+4.6 over DINO) |
| End-to-end FPS (A100, FP16) | **18.4 FPS** |
| Full pipeline parameters | **119.8 M** |
| GNN mAP gain over fusion-only | **+8.2%** |
| Trajectory MSE reduction | **34.2%** |

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pipeline](#pipeline)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Zoo](#model-zoo)
- [Results](#results)
- [Inference Server](#inference-server)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Annotation-free multimodal pretraining** — NT-Xent contrastive loss and BYOL-style bootstrap objectives adapted independently for RGB, Thermal, LiDAR, and Radar encoders. No bounding box or segmentation labels needed during SSL pretraining.
- **Phantom modality solution** — no public dataset provides all four modalities synchronously. Cross-modal objectives are explicitly decoupled across datasets (nuScenes for RGB/LiDAR/Radar; FLIR ADAS for RGB/Thermal), resolving this constraint at the training objective level.
- **Uncertainty-aware transformer fusion** — a 6-layer, 8-head transformer fuses heterogeneous modality tokens with learned confidence weighting and modality dropout, yielding stable performance under sensor degradation and occlusion.
- **Self-supervised scene graph edges** — nodes are initialized from a frozen Faster R-CNN detector (COCO); all edge semantics and relational attributes are learned without annotation via geometric proximity, Doppler similarity, and a learned relational head.
- **Competitive in low-label regimes** — with only 10% of nuScenes labels, the framework reaches 95% of CenterPoint's fully supervised mAP (0.481 vs 0.503).

---

## Architecture

![Arch](https://github.com/user-attachments/assets/576d0897-7440-4f7d-a4b9-b8e311d87a0f)

The framework is a five-stage end-to-end pipeline:

```
Stage 1 — Sensor Inputs
  RGB (640×480, 3ch)  ·  Thermal (LWIR, 1ch)  ·  LiDAR (N×4, BEV)  ·  Radar (range-Doppler map)

Stage 2 — Modality-Specific SSL Encoders
  RGB / Thermal  →  ResNet-50 backbone  →  512-D embedding
  LiDAR          →  BEV-CNN             →  512-D embedding
  Radar          →  Lightweight CNN     →  512-D embedding
  (each pretrained independently via NT-Xent + BYOL objectives)

Stage 3 — Transformer Fusion Hub
  [h_RGB ; h_Thermal ; h_LiDAR ; h_Radar]  →  6-layer × 8-head transformer
  Uncertainty-aware attention · Modality dropout  →  Shared latent Z ∈ R^{B×512}

Stage 4 — Scene Graph Construction + GNN Reasoning
  Nodes  : Faster R-CNN proposals (frozen, COCO-pretrained) → RoI-pooled features
  Edges  : BEV proximity · Doppler similarity · Learned relational score  (self-supervised)
  GNN    : 3-layer GraphSAGE (hidden=256)  →  Relational embeddings H

Stage 5 — Downstream Task Heads
  Object Detection       :  Faster R-CNN head
  Semantic Segmentation  :  DeepLabV3 head
  Trajectory Prediction  :  MLP head
```
---

## Installation

**Requirements:** Python 3.10 · CUDA 12.1 · 16 GB+ GPU VRAM (A100 / T4 recommended)

```bash
# 1. Clone the repository
git clone [https://github.com/<your-org>/multimodal-ssl-ad](https://github.com/ManvithMadhuvarsu/multimodal-autonomous-driving-ssl-pipeline).git
cd multimodal-ssl-ad

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify CUDA setup
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

<details>
<summary><b>Core dependencies</b></summary>

```
torch>=2.0.0
torchvision>=0.15.0
open3d>=0.17.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
fastapi>=0.100.0
uvicorn>=0.22.0
wandb>=0.15.0
tqdm>=4.65.0
Pillow>=10.0.0
scikit-learn>=1.3.0
```

</details>

---

## Data Preparation

The pipeline trains on a curated corpus from three public datasets. Download and arrange as follows:

```
data/
├── nuScenes/
│   ├── samples/           # RGB frames
│   ├── sweeps/            # LiDAR + Radar sweeps
│   ├── lidarseg/          # LiDAR segmentation labels
│   ├── panoptic/          # Panoptic labels
│   └── v1.0-trainval/     # Metadata + calibration
├── nuImages/
│   ├── samples/
│   └── v1.0-train/
└── FLIR_ADAS/
    ├── train/
    │   ├── RGB/
    │   └── thermal_8_bit/
    └── val/
        ├── RGB/
        └── thermal_8_bit/
```

| Modality | Raw Files | Training Cap |
|---|---|---|
| RGB Images | 178,348 | 50,000 |
| Thermal Images | 15,638 | 15,000 |
| LiDAR scans | 45,125 | 30,000 |
| Radar frames | 46,567 | 30,000 |
| Panoptic Labels | 7,956 | — |
| Temporal Sweeps | 35,000+ | — |
| Maps / Metadata | 54,966 | — |

Set your data root in `01_setup.py`:

```python
ROOT = Path("/path/to/your/data")
```

Then build the unified dataset index:

```bash
python 02_dataset_indexing.py
# → unified_dataset.json (~350k entries across all modalities)
```

---

## Pipeline

```
02_dataset_indexing.py      →  unified_dataset.json
        │
        ▼
04_ssl_training.py          →  ssl_checkpoints/  +  ssl_embeddings.json
        │
        ▼
05_embedding_and_fusion.py  →  fused_embeddings.json  +  scene_graphs.json
        │
        ▼
06_gnn_training.py          →  gnn_embeddings.json
        │
        ▼
07_perception_heads.py      →  full_model_inference.pt  +  test_results.json
        │
        ▼
08_rl_agent.py              →  aggregated_metrics.json   (optional — requires CARLA)
        │
        ▼
09_inference_server.py      →  REST API on :8000
```

Every stage writes a `.done` flag on completion. Interrupted runs are fully resumable — re-running any script safely skips completed stages.

---

## Training

### Option A — Full pipeline (recommended)

```bash
python run_pipeline.py

# Skip RL stage if no driving simulator is available
python run_pipeline.py --skip-rl
```

### Option B — Stage by stage

```bash
# SSL pretraining — 100 epochs (nuScenes) / 50 epochs (FLIR)
python 04_ssl_training.py --modality rgb     --epochs 100
python 04_ssl_training.py --modality thermal --epochs 50
python 04_ssl_training.py --modality lidar   --epochs 100
python 04_ssl_training.py --modality radar   --epochs 100

# Fusion transformer + scene graph construction
python 05_embedding_and_fusion.py

# GNN training
python 06_gnn_training.py

# Downstream task heads + TorchScript export
python 07_perception_heads.py
```

### Training configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 1 × 10⁻⁴ |
| LR schedule | Cosine annealing (T_max=100) |
| Batch size | 64 |
| Temperature τ (NT-Xent) | 0.07 |
| Momentum ξ (BYOL) | 0.996 |
| Transformer layers / heads | 6 / 8 |
| Transformer embedding dim | 512 |
| Projection head dim | 256 |
| GNN layers (GraphSAGE) | 3 |
| GNN hidden dim | 256 |
| Max proposals / frame (top-K) | 50 |
| Loss weights λ_nt / λ_cm / λ_byol / λ_rec | 1.0 / 0.5 / 0.5 / 0.1 |
| Random seed | 42 |

Mixed-precision (FP16) training is enabled by default via `torch.amp.GradScaler` with transparent CPU fallback. Only the most recent SSL checkpoint is retained (`KEEP_CKPTS=1`).

---

## Evaluation

```bash
# Zero-label linear probe on frozen SSL embeddings
python eval.py --mode linear_probe --checkpoint ssl_checkpoints/best.pt

# Fine-tune with N% of nuScenes labels
python eval.py --mode finetune --label-fraction 0.10

# Full benchmark (detection + segmentation + trajectory)
python eval.py --mode full --checkpoint full_model_inference.pt
```

---

## Model Zoo

| Model | Labels | mAP | mIoU | FPS | Download |
|---|---|---|---|---|---|
| SSL linear probe | 0% | — | 46.3 | — | *coming soon* |
| Fine-tuned 10% | 10% | 0.481 | 49.1 | — | *coming soon* |
| Fine-tuned 100% | 100% | 0.597 | 52.8 | 18.4 | *coming soon* |

> Checkpoints will be released upon paper acceptance.

---

## Results

### SSL Representation Quality

Linear probe accuracy on frozen embeddings — mean ± std over 3 seeds (42, 123, 7):

| Modality | Top-1 Accuracy |
|---|---|
| Radar | 53.8 ± 0.7% |
| Thermal | 59.4 ± 0.6% |
| RGB | 67.1 ± 0.4% |
| LiDAR | 71.0 ± 0.5% |
| **Multimodal Fusion** | **76.9 ± 0.3%** |

t-SNE/UMAP projections confirm well-separated, compact per-modality clusters. All encoders converge smoothly across 100 epochs with no representation collapse.

---

### Ablation: Modality Combinations & Architectural Components

Evaluated on nuScenes / nuImages val splits. All variants use identical hyperparameters and seed.

| Configuration | mAP | mIoU | ΔmAP | Night | Occlusion | Rain |
|---|---|---|---|---|---|---|
| RGB Only | 0.312 | 34.1 | — | ✗ Failure | Low | Moderate |
| RGB + LiDAR | 0.421 | 38.6 | +10.9 | Moderate | High | Moderate |
| RGB + Thermal | 0.389 | 41.3 | +7.7 | High | Low | High |
| RGB + LiDAR + Radar | 0.448 | 39.7 | +13.6 | Moderate | High | High |
| Full Fusion, no Scene Graph | 0.451 | 43.2 | +13.9 | High | Moderate | High |
| Full Fusion + Graph, no GNN | 0.469 | 44.8 | +15.7 | High | High | High |
| **Full Fusion (Proposed)** | **0.508** | **49.1** | **+19.6** | High | **Very High** | High |

---

### Comparison with Prior Work

| Method | Modalities | Supervision | Labels | mAP | mIoU |
|---|---|---|---|---|---|
| PointPillars | LiDAR | Supervised | 100% | 0.305 | — |
| CenterPoint | LiDAR | Supervised | 100% | 0.503 | — |
| IS-Fusion | RGB + LiDAR | Supervised | 100% | 0.642 | — |
| BEVFusion | RGB + LiDAR | Supervised | 100% | 0.676 | — |
| SimCLR (linear probe) | RGB | SSL | 0% | — | 34.1 |
| DINO (linear probe) | RGB | SSL | 0% | — | 41.7 |
| **Ours (linear probe)** | **RGB+TH+L+R** | **SSL** | **0%** | **—** | **46.3** |
| **Ours (10% fine-tune)** | **RGB+TH+L+R** | **SSL+FT** | **10%** | **0.481** | **49.1** |
| **Ours (100% fine-tune)** | **RGB+TH+L+R** | **SSL+FT** | **100%** | **0.597** | **52.8** |

---

### GNN Reasoning Impact

| Variant | Relative mAP Gain |
|---|---|
| Fusion only | +0.0% |
| Fusion + Scene Graph (no GNN) | +4.1% |
| **Fusion + Scene Graph + GNN** | **+8.2%** |

Downstream task improvements after full GNN refinement:

| Task | Improvement |
|---|---|
| Object detection loss | **−17.8%** |
| Semantic segmentation loss | **−16.3%** |
| Trajectory prediction MSE | **−34.2%** |

---

### Cross-Modal Retrieval (Recall@1)

| Query → Gallery | Recall@1 |
|---|---|
| Thermal → RGB | 51.0% |
| RGB → Thermal | 53.7% |
| LiDAR → Radar | 52.4% |
| RGB → LiDAR | 57.8% |
| **Multimodal → All** | **69.5%** |

---

### Inference Computational Profile

Measured on NVIDIA A100 80 GB · CUDA 12.1 · FP16 · batch size 1  
Input: 640×480 RGB/Thermal · 40k LiDAR points · 125 Radar points

| Component | GFLOPs | Params (M) | Latency (ms) |
|---|---|---|---|
| RGB Encoder (ResNet-50) | 4.1 | 25.6 | 8.2 |
| Thermal Encoder (ResNet-50) | 4.1 | 25.6 | 8.3 |
| LiDAR Encoder (BEV-CNN) | 7.4 | 18.3 | 12.7 |
| Radar Encoder (Lightweight CNN) | 0.6 | 3.1 | 2.1 |
| Fusion Transformer (6L, 8H) | 11.3 | 42.0 | 18.4 |
| GNN (GraphSAGE, 3L) | 1.8 | 5.2 | 4.6 |
| **Full Pipeline** | **29.3** | **119.8** | **54.3 ms / 18.4 FPS** |

Peak VRAM: **6.3 GB** — within NVIDIA Drive Orin (64 GB) automotive limits.  
119.8 M params / 29.3 GFLOPs vs. BEVFusion's 200 M+ params, in a strictly harder 4-modality SSL setting.

![combined_6_modalities_no_model_text](https://github.com/user-attachments/assets/25acf4fa-868c-4c9e-a029-7771e37e26f9)

---

## Inference Server

A FastAPI REST server exposes the full pipeline for real-time inference.

```bash
python 09_inference_server.py
# Serving at http://0.0.0.0:8000
```

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Run full pipeline on a single frame |
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Latency and throughput statistics |

```bash
# Example — single-frame prediction
curl -X POST http://localhost:8000/predict \
  -F "rgb=@frame.jpg" \
  -F "thermal=@thermal.png" \
  -F "lidar=@points.bin" \
  -F "radar=@radar.pcd"
```

---

## File Structure

```
multimodal-ssl-ad/
├── 01_setup.py                          # Paths, seeds, device, logging, checkpoint helpers
├── 02_dataset_indexing.py               # Dataset scan → unified_dataset.json
├── 03_models.py                         # All architectures (encoders, transformer, GNN, heads)
├── 04_ssl_training.py                   # Per-modality SSL pretraining
├── 05_embedding_and_fusion.py           # Embedding extraction, fusion training, scene graphs
├── 06_gnn_training.py                   # GraphSAGE training + evaluation
├── 07_perception_heads_and_export.py    # Task heads, model assembly, TorchScript export
├── 08_rl_agent.py                       # PPO trainer, reward shaping, rollout buffer
├── 09_inference_server.py               # FastAPI inference server
├── eval.py                              # Evaluation and benchmarking
├── run_pipeline.py                      # Master orchestrator
├── requirements.txt
└── README.md
```

**Generated artifacts:**

| File | Description |
|---|---|
| `unified_dataset.json` | Flat list of `{modality, path}` records |
| `ssl_embeddings.json` | Per-entry `{rgb_emb, thermal_emb, lidar_emb, radar_emb}` |
| `fused_embeddings.json` | Per-entry fused 512-D representation |
| `scene_graphs.json` | Per-entry `{nodes, edges}` scene graph |
| `gnn_embeddings.json` | Per-entry 512-D GNN-refined embedding |
| `full_model_inference.pt` | Full model state dict |
| `full_model_traced.pt` | TorchScript traced model |
| `test_results.json` | Per-frame detection / segmentation / trajectory outputs |
| `aggregated_metrics.json` | Summary benchmark statistics |
| `test_results_hist.png` | Output norm distribution histogram |

---

## Limitations

- **Throughput** — 18.4 FPS is below the 30 FPS threshold for production AD stacks. INT8 quantization and encoder pruning are planned for edge deployment targets (Drive Orin, Jetson AGX).
- **Extreme conditions** — performance degrades in heavy rain, dense fog, and sub-lux nighttime environments where both RGB and Thermal signals are compromised.
- **Sensor misalignment** — uncalibrated Radar–LiDAR pairs produce noisy cross-modal embeddings; the system falls back to learned alignment with reduced accuracy.
- **GNN over-smoothing** — dense scene graphs with high node counts can homogenize embeddings across GraphSAGE layers, occasionally degrading fine-grained segmentation quality.
- **RL agent** — requires a compatible driving simulator (CARLA). Use `--skip-rl` to bypass cleanly.

---

## Hardware

| Resource | Spec |
|---|---|
| GPU (training) | NVIDIA Tesla T4 · A100 80 GB SXM4 |
| CUDA | 12.1 |
| Precision | FP16 mixed precision |
| Framework | PyTorch 2.x |
| Python | 3.10 |
| Point cloud processing | Open3D |
| Experiment tracking | Weights & Biases |

---

---

## Acknowledgements

This work builds on the following open-source projects and datasets:

- [nuScenes](https://www.nuscenes.org/) — Caesar et al., CVPR 2020
- [FLIR ADAS Dataset](https://www.flir.com/oem/adas/adas-dataset-form/) — FLIR Systems
- [SimCLR](https://github.com/google-research/simclr) — Chen et al., ICML 2020
- [BYOL](https://github.com/deepmind/deepmind-research/tree/master/byol) — Grill et al., NeurIPS 2021
- [BEVFusion](https://github.com/mit-han-lab/bevfusion) — Liu et al., ICRA 2023
- [IS-Fusion](https://github.com/yinjunbo/IS-Fusion) — Yin et al., CVPR 2024
- [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn) — Ren et al., NeurIPS 2015
