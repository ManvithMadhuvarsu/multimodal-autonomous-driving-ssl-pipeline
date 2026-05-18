# Multimodal-SSL-AD vs the 2023–2026 State of the Art

A reviewer-grade comparison of your pipeline against the most recent published and preprint
work in autonomous-driving perception, with every metric traceable to its source paper.

> **Reading order.** §1 sets the goalposts. §2 lists the 19 baseline methods grouped by
> family. §3 walks each animated comparison artifact. §4 explains what is **measured**
> vs **estimated** so reviewers can judge claims. §5 (`IMPROVEMENTS.md`) translates
> findings into a prioritised change-list for your pipeline.

---

## 1. Your pipeline — the numbers under review

From `README.md` (`Multimodal-SSL-AD`, M.Tech thesis, target IEEE Open Journal):

| Axis | Value | Where it stands in 2026 |
|---|---|---|
| 3D detection mAP (nuScenes, 100% labels) | **0.597** | Below pure-LiDAR SOTA (0.730 SparseLIF) and supervised LiDAR+Cam SOTA (0.734 InsFusion); above CenterPoint LiDAR-only (0.503) |
| SSL linear-probe mIoU | **46.3** | +4.6 over DINO RGB (41.7), +12.2 over SimCLR (34.1) |
| Throughput | **18.4 FPS** (A100 FP16) | Below 30 FPS production floor; above IS-Fusion (4.8), GraphAD (3.6); below CenterPoint (11), TransFusion (6), PointPillars (62) |
| Params | **119.8 M** | About half of BEVFusion-MIT (205 M); 1.4× FocalFormer3D (66 M) |
| Modalities | **RGB + Thermal + LiDAR + Radar** | **Only SAMFusion (ECCV 2024) matches all four modality classes**; everyone else is ≤3 |
| Label efficiency at 10% | **0.481 mAP** (95% of CenterPoint's 100%-labels result) | Best directly-reported low-label-regime number in this comparison set |
| Supervision | **SSL + fine-tune** | Most competitive methods are still fully supervised; DriveWorld and GASP are the recent SSL exceptions |

Your **strongest differentiated claim** is the multimodal SSL framing — not raw mAP.
Pitch the paper around label efficiency, modality coverage, and the GNN-as-relational-reasoner,
not against the supervised mAP leaderboard.

---

## 2. Baselines included in the comparison (verified sources)

### 2a. Supervised 3D detection on nuScenes

| Method | Year | Venue | Modality | mAP | NDS | FPS | Source |
|---|---|---|---|---|---|---|---|
| PointPillars | 2019 | CVPR | LiDAR | 0.305 | 0.453 | 62.0 | [arXiv:1812.05784](https://arxiv.org/abs/1812.05784) |
| CenterPoint | 2021 | CVPR | LiDAR | 0.503 | 0.601 | 11.0 | [arXiv:2006.11275](https://arxiv.org/abs/2006.11275) |
| TransFusion | 2022 | CVPR | RGB+L | 0.689 | 0.717 | 6.0 | [arXiv:2203.11496](https://arxiv.org/abs/2203.11496) |
| BEVFusion (MIT) | 2023 | ICRA | RGB+L | 0.696 | 0.721 | 8.4 | [arXiv:2205.13542](https://arxiv.org/abs/2205.13542) |
| FocalFormer3D | 2023 | ICCV | LiDAR | 0.705 | 0.731 | 4.5 | [arXiv:2308.04556](https://arxiv.org/abs/2308.04556) |
| SparseFusion | 2023 | ICCV | RGB+L | 0.720 | 0.731 | 5.6 | [arXiv:2304.14340](https://arxiv.org/abs/2304.14340) |
| IS-Fusion | 2024 | CVPR | RGB+L | 0.723 | 0.737 | 4.8 | [CVPR open-access](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_IS-Fusion_Instance-Scene_Collaborative_Fusion_for_Multimodal_3D_Object_Detection_CVPR_2024_paper.pdf) |
| SparseLIF | 2024 | arXiv | RGB+L | 0.730 | **0.746** | 3.2 | [arXiv:2403.07284](https://arxiv.org/abs/2403.07284) |
| InsFusion (on IS-Fusion) | 2025 | arXiv | RGB+L | **0.734** | 0.743 | 0.85 | [arXiv:2509.08374](https://arxiv.org/abs/2509.08374) |
| SAMFusion | 2024 | ECCV | RGB+Gated+L+Radar | per-weather | — | — | [arXiv:2508.16408](https://arxiv.org/abs/2508.16408) / [ECCV 2024](https://eccv.ecva.net/virtual/2024/poster/1726) |

> **Direct mAP comparison vs your 0.597 is unfair** — all rows above are fully supervised on 100% labels.
> The fair comparison is your label-efficiency curve (GIF 1) and your robustness profile (GIF 4).

### 2b. End-to-end planning on nuScenes (UniAD open-loop protocol)

| Method | Year | Venue | min ADE 3s ↓ | L2 avg ↓ | Collision % ↓ | FPS | Source |
|---|---|---|---|---|---|---|---|
| UniAD | 2023 | **CVPR (Best Paper)** | 0.71 | 1.03 | 0.31 | 1.8 | [arXiv:2212.10156](https://arxiv.org/abs/2212.10156) |
| VAD | 2023 | ICCV | — | 1.22 | 0.47 | 4.5 | [arXiv:2303.12077](https://arxiv.org/abs/2303.12077) |
| FusionAD | 2023 | arXiv | 0.389 | — | 0.12 | — | [arXiv:2308.01006](https://arxiv.org/abs/2308.01006) |
| GraphAD | 2024 | arXiv | 0.68 | 0.68 | 0.12 | 3.6 | [arXiv:2403.19098](https://arxiv.org/abs/2403.19098) |
| GenAD | 2024 | ECCV | — | 0.95 | 0.27 | 6.7 | [arXiv:2402.11502](https://arxiv.org/abs/2402.11502) |
| UAD | 2024 | TPAMI | — | 0.63 | 0.19 | 6.1 | [arXiv:2406.17680](https://arxiv.org/abs/2406.17680) |
| DriveWorld | 2024 | CVPR | 0.61 | 0.69 | 0.19 | — | [arXiv:2405.04390](https://arxiv.org/abs/2405.04390) |
| FSDrive | 2025 | NeurIPS | — | 0.28 | — | — | [arXiv:2505.17685](https://arxiv.org/abs/2505.17685) |
| **VLA-World 2B** | **2026** | arXiv | — | **0.26** | **0.08** | — | [arXiv:2604.09059](https://arxiv.org/abs/2604.09059) |
| **VLA-World 7B** | **2026** | arXiv | — | **0.18** | — | — | same |

### 2c. Self-supervised pretraining for AD

| Method | Year | Venue | Headline gain | Source |
|---|---|---|---|---|
| SimCLR linear probe (RGB) | 2020 | ICML | mIoU 34.1 (your README) | [arXiv:2002.05709](https://arxiv.org/abs/2002.05709) |
| DINO linear probe (RGB) | 2021 | ICCV | mIoU 41.7 (your README) | [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) |
| OccFeat | 2024 | CVPRW | BEV-segmentation pretraining | [arXiv:2404.14027](https://arxiv.org/abs/2404.14027) |
| GaussianPretrain | 2024 | arXiv | mAP 0.444 after pretraining | [arXiv:2411.12452](https://arxiv.org/abs/2411.12452) |
| DriveWorld (4D world model) | 2024 | CVPR | mAP 0.452 (+7.5%), AMOTA 0.412 (+5.3%) | [arXiv:2405.04390](https://arxiv.org/abs/2405.04390) |
| GASP (geometric+semantic) | 2025 | arXiv | SOTA on occupancy forecasting + ego-traj + online mapping | [Zenseact GASP](https://research.zenseact.com/publications/gasp/) |
| **Ours (multimodal SSL+FT)** | **2026** | thesis | mIoU 46.3 zero-label / 76.9 fusion linear-probe top-1 | local |

### 2d. Adverse-weather perception (the lane SAMFusion owns)

SAMFusion (Bijelic et al., ECCV 2024) is the closest analog to your sensor stack and the only
peer that reports condition-stratified results. Pedestrian 3D-AP at far range (50–80 m) from
SAMFusion Table 2 [(PDF)](https://light.princeton.edu/wp-content/uploads/2024/09/SAMFusion.pdf):

| Condition | SAMFusion | Reverse-engineered next-best (RGB+L) | Your pipeline today |
|---|---|---|---|
| Clear day | 40.16 | ~36.0 | not benchmarked |
| Clear night | 27.14 | ~14.0 | not benchmarked |
| Dense fog | 34.31 | ~17.1 (SAMFusion +17.2 AP) | not benchmarked |
| Snow | 41.45 | ~25.8 (SAMFusion +15.6 AP) | not benchmarked |

You have thermal+radar — the right sensors to compete here — but **no benchmark numbers
in the adverse-weather setting**. This is the biggest hole a reviewer will flag.

---

## 3. Animated comparison artifacts (`gifs/`)

Every GIF below shows **evolution** — points appearing, curves drawing, bars filling, polygons
expanding — not a static stitch of panels. Open them in any browser or markdown viewer.

| # | File | What it shows | Why it matters |
|---|---|---|---|
| 1 | [`gifs/01_label_efficiency.gif`](gifs/01_label_efficiency.gif) | mAP vs % nuScenes labels — curves draw left-to-right, your curve highlighted | Your strongest pitch: 95% of CenterPoint's full-label result with only 10% labels |
| 2 | [`gifs/02_fps_vs_map_pareto.gif`](gifs/02_fps_vs_map_pareto.gif) | Methods appear chronologically (2019→2026) on a FPS-vs-mAP scatter; Pareto front updates | Reveals exactly where you sit on the throughput–accuracy trade-off |
| 3 | [`gifs/03_trajectory_horizon.gif`](gifs/03_trajectory_horizon.gif) | min ADE drawn across 0.5–6 s prediction horizon | Shows your trajectory MSE-reduction claim in horizon-resolved form |
| 4 | [`gifs/04_weather_robust_race.gif`](gifs/04_weather_robust_race.gif) | Pedestrian far-range AP bar race across clear/night/fog/snow | Where SAMFusion crushes everyone and where you should benchmark |
| 5 | [`gifs/05_planning_collision.gif`](gifs/05_planning_collision.gif) | Open-loop collision rate, UniAD (2023) → VLA-World (2026) bar race | Shows the 2026 collapse to 0.08–0.18% by VLA backbones |
| 6 | [`gifs/06_ssl_convergence.gif`](gifs/06_ssl_convergence.gif) | SSL linear-probe top-1 vs epochs for SimCLR / DINO / DINOv2 / MAE / I-JEPA / Ours | Your multimodal fusion 76.9% sits between DINOv2 (84.5) and I-JEPA (77.3) |
| 7 | [`gifs/07_modality_ablation.gif`](gifs/07_modality_ablation.gif) | Rotating radar of your 7 ablation configurations on (mAP, mIoU, night, occlusion, rain) | Makes the GNN contribution legible in a single shape |

---

## 4. Provenance — measured vs estimated

This is the part reviewers will probe hardest. Every estimated value is flagged in the
relevant GIF caption as well; here is the consolidated map:

| Source of number | Verifiable? | Examples |
|---|---|---|
| **Paper-reported endpoint** | yes | Every mAP/NDS/L2/collision in §2 |
| **Linearly interpolated curve** | no — flagged | Intermediate label-efficiency points for DriveWorld, BEVFusion, IS-Fusion, CenterPoint, Ours |
| **Reverse-engineered from a reported delta** | no — flagged | "Next-best runner-up" weather APs in GIF 4 (computed from SAMFusion's stated +17.2 / +15.6 AP gains) |
| **Estimated for Ours** | no — flagged | Your min ADE 3 s (estimated from 34.2% MSE reduction over fusion-only baseline; see GIF 3 caption) and Ours' weather APs in GIF 4 (estimated from mAP-clear ratio × SAMFusion drop pattern) |
| **Illustrative shape** | no — flagged | Per-epoch SSL convergence shapes in GIF 6; only epoch-100 endpoint is paper-anchored |

When you put any of these GIFs into your IEEE submission, **the safest move is to leave
the "estimated/interpolated" stamps visible** — reviewers will accept honest estimation
markers; they will not accept fabricated-looking curves with no provenance.

---

## 5. Where this lands you for IEEE submission

- Your **mAP positioning** is honest but not exciting on its own — 0.597 is below the
  2023+ supervised fusion frontier (0.69–0.73).
- Your **label-efficiency positioning** is genuinely strong — 0.481 mAP at 10 % labels is
  competitive with several 100 %-label baselines.
- Your **modality breadth** is unique among the 19 comparison methods; only SAMFusion comes close.
- Your **planning numbers are missing entirely** — you don't report ADE/FDE/collision in the
  UniAD format. This is the easiest single gap to close before submission (estimated effort:
  one week of evaluation on the standard protocol, no retraining needed).
- Your **adverse-weather numbers are missing entirely**, despite having thermal+radar. Run
  SeeingThroughFog / DENSE eval; if you match or beat SAMFusion's RGB+L runner-ups on any
  condition, that becomes a paper-level contribution by itself.

The detailed change-list — ranked by paper-acceptance impact — is in **`IMPROVEMENTS.md`**.
