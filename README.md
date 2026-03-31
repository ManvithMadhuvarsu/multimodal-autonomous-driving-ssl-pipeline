# Autonomous Driving Multimodal SSL Pipeline

End-to-end self-supervised learning pipeline for autonomous scene understanding
using RGB, Thermal, LiDAR, and Radar sensor fusion.

## Pipeline Overview

```
FLIR_ADAS + NuScenes + Oth![Arch](https://github.com/user-attachments/assets/576d0897-7440-4f7d-a4b9-b8e311d87a0f)
er_Files
          │
          ▼
  02_dataset_indexing.py      — Scan & build unified_dataset.json
          │
          ▼
  04_ssl_training.py          — NT-Xent contrastive SSL per modality (30 epochs)
          │
          ▼
  05_embedding_and_fusion.py  — Extract SSL embeddings → Train Fusion Transformer
                                → Extract fused embeddings → Build scene graphs
          │
          ▼
  06_gnn_training.py          — Train GNN over scene graphs → Extract GNN embeddings
          │
          ▼
  07_perception_heads.py      — Train Detection / Segmentation / Trajectory heads
                                → Assemble & export FullADModel (TorchScript)
          │
          ▼
  08_rl_agent.py              — PPO RL agent + reward shaping (requires gym env)
          │
          ▼
  09_inference_server.py      — FastAPI REST server for real-time inference
```

## File Structure

| File | Purpose |
|------|---------|
| `01_setup.py` | Paths, seeds, device, logger, raw loaders, checkpoint helpers |
| `02_dataset_indexing.py` | Folder scan, unified dataset index |
| `03_models.py` | All neural network architectures |
| `04_ssl_training.py` | SSL dataset, safe loaders, validation, training loop |
| `05_embedding_and_fusion.py` | Embedding extraction, fusion training, scene graphs |
| `06_gnn_training.py` | GNN training + evaluation |
| `07_perception_heads_and_export.py` | Task heads, full model assembly, test eval |
| `08_rl_agent.py` | Reward shaping, rollout buffer, PPO trainer, aggregation |
| `09_inference_server.py` | FastAPI server |
| `run_pipeline.py` | Master runner — executes all stages in order |
| `requirements.txt` | Python dependencies |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your data root in 01_setup.py
#    ROOT = Path("/path/to/your/Project")

# 3. Run the full pipeline (resumable at any stage)
python run_pipeline.py

# 4. Skip RL training (if no simulator available)
python run_pipeline.py --skip-rl

# 5. Start the inference server
python 09_inference_server.py
```

## Key Design Decisions

**Resumability** — Every stage writes a `.flag` file on completion and
incremental progress to JSON files. Re-running any script safely skips
already-completed stages.

**Checkpoint Rotation** — Only the most recent SSL checkpoint is retained
(`KEEP_CKPTS=1`) to avoid filling up Google Drive during long runs.

**AMP** — All training loops use `torch.amp.GradScaler` for mixed-precision
on CUDA, with transparent CPU fallback.

**Modular Architecture** — Models in `03_models.py` are entirely independent
of training logic. Swap any component (e.g. replace PointNet with a sparse
convolution network) without touching training code.

## Outputs

| File | Description |
|------|-------------|
| `unified_dataset.json` | Flat list of {modality, path} entries |
| `ssl_embeddings.json` | Per-entry {rgb_emb, thermal_emb, lidar_emb, radar_emb} |
| `fused_embeddings.json` | Per-entry {fused, rgb_emb, thermal_emb, lidar_emb} |
| `scene_graphs.json` | Per-entry {nodes, edges} scene graph |
| `gnn_embeddings.json` | Per-entry 512-D GNN embedding |
| `full_model_inference.pt` | Full model state dict |
| `full_model_traced.pt` | TorchScript traced model (if compatible) |
| `test_results.json` | Per-image {detection, segmentation, trajectory} outputs |
| `aggregated_metrics.json` | Summary statistics for test results |
| `test_results_hist.png` | Histogram of output norm distributions |
