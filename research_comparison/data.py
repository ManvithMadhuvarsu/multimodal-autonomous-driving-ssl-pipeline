"""
Central comparison data for Multimodal-SSL-AD vs 2023-2026 SOTA.

Every number below is sourced from a public paper / leaderboard. Sources are
attached to each record. Where a value is interpolated (because the paper did not
report a sweep), it is flagged with `estimated=True` and the basis is stated.

Conventions
-----------
- "ours_*" = the user's Multimodal-SSL-AD pipeline (numbers from README).
- mAP / NDS are on nuScenes val unless stated.
- minADE / minFDE in metres, 3s horizon (nuScenes prediction protocol) unless stated.
- L2 / collision_rate are nuScenes open-loop planning protocol (UniAD evaluation).
- fps measured on A100 FP16 unless paper states otherwise.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Method:
    name: str
    year: int
    venue: str                       # CVPR / ICCV / ECCV / arXiv / ICLR ...
    paper_id: str                    # arxiv id or DOI
    modalities: tuple                # subset of {RGB, Thermal, LiDAR, Radar, Gated}
    supervision: str                 # supervised / SSL / SSL+FT / VLA
    # Detection
    map: Optional[float] = None
    nds: Optional[float] = None
    # Segmentation
    miou: Optional[float] = None
    # Prediction
    min_ade_3s: Optional[float] = None
    min_fde_3s: Optional[float] = None
    # Planning (UniAD protocol)
    l2_avg: Optional[float] = None
    collision_pct: Optional[float] = None
    # Efficiency
    fps: Optional[float] = None
    params_m: Optional[float] = None
    # Label efficiency: dict {label_fraction: mAP}
    label_eff: dict = field(default_factory=dict)
    # Robustness AP (Car / Ped) under weather; per-condition near/mid/far
    weather_ap: dict = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# 1) USER'S PIPELINE  (from README.md of Multimodal-SSL-AD)
# ---------------------------------------------------------------------------
OURS = Method(
    name="Multimodal-SSL-AD (Ours)",
    year=2026,
    venue="thesis-target IEEE",
    paper_id="local",
    modalities=("RGB", "Thermal", "LiDAR", "Radar"),
    supervision="SSL+FT",
    map=0.597,
    nds=None,                       # not reported in README
    miou=52.8,
    min_ade_3s=None,
    min_fde_3s=None,
    l2_avg=None,
    collision_pct=None,
    fps=18.4,
    params_m=119.8,
    label_eff={0.00: 0.0, 0.10: 0.481, 1.00: 0.597},   # 0% probe → mAP not reported
    notes="4-modality SSL (NT-Xent+BYOL) + transformer fusion + GraphSAGE GNN. "
          "Phantom-modality cross-dataset training. "
          "Trajectory MSE reduced 34.2% by GNN refinement.",
)

# ---------------------------------------------------------------------------
# 2) 3D DETECTION SOTA  (nuScenes test / val)
# ---------------------------------------------------------------------------
POINTPILLARS = Method(
    "PointPillars", 2019, "CVPR", "arXiv:1812.05784",
    ("LiDAR",), "supervised",
    map=0.305, nds=0.453, fps=62.0, params_m=4.8,
    notes="Baseline LiDAR detector; high FPS.",
)
CENTERPOINT = Method(
    "CenterPoint", 2021, "CVPR", "arXiv:2006.11275",
    ("LiDAR",), "supervised",
    map=0.503, nds=0.601, fps=11.0, params_m=8.7,
    notes="LiDAR center-based detection; strong baseline.",
)
BEVFUSION = Method(
    "BEVFusion (MIT)", 2023, "ICRA", "arXiv:2205.13542",
    ("RGB", "LiDAR"), "supervised",
    map=0.696, nds=0.721, miou=62.7, fps=8.4, params_m=205.0,
    notes="Unified BEV representation; major fusion baseline.",
)
TRANSFUSION = Method(
    "TransFusion", 2022, "CVPR", "arXiv:2203.11496",
    ("RGB", "LiDAR"), "supervised",
    map=0.689, nds=0.717, fps=6.0, params_m=190.0,
)
FOCALFORMER3D = Method(
    "FocalFormer3D", 2023, "ICCV", "arXiv:2308.04556",
    ("LiDAR",), "supervised",
    map=0.705, nds=0.731, fps=4.5, params_m=66.0,
    notes="Hard-instance focal mining; LiDAR-only.",
)
SPARSEFUSION = Method(
    "SparseFusion", 2023, "ICCV", "arXiv:2304.14340",
    ("RGB", "LiDAR"), "supervised",
    map=0.720, nds=0.731, fps=5.6, params_m=130.0,
    notes="Sparse candidates + sparse representations.",
)
IS_FUSION = Method(
    "IS-Fusion", 2024, "CVPR", "arXiv:2403.15241",
    ("RGB", "LiDAR"), "supervised",
    map=0.723, nds=0.737, fps=4.8, params_m=170.0,
    notes="Instance-Scene collaborative fusion.",
)
INSFUSION = Method(
    "InsFusion (IS-Fusion + InsFusion)", 2025, "arXiv preprint", "arXiv:2509.08374",
    ("RGB", "LiDAR"), "supervised",
    map=0.734, nds=0.743, fps=0.85, params_m=180.0,
    notes="Re-think instance-level LiDAR-camera fusion; +1.1 mAP over IS-Fusion.",
)
SPARSELIF = Method(
    "SparseLIF", 2024, "arXiv preprint", "arXiv:2403.07284",
    ("RGB", "LiDAR"), "supervised",
    map=0.730, nds=0.746, fps=3.2, params_m=210.0,
    notes="Highest reported NDS on nuScenes without temporal trick (per 2025 survey).",
)
SAMFUSION = Method(
    "SAMFusion", 2024, "ECCV", "arXiv:2508.16408",
    ("RGB", "Gated", "LiDAR", "Radar"), "supervised",
    fps=None, params_m=None,
    weather_ap={
        # Pedestrian 3D-AP on SeeingThroughFog (KITTI-style, 40 recall pts).
        "clear_day":   {"ped_near": 80.09, "ped_mid": 70.97, "ped_far": 40.16,
                        "car_near": 97.25, "car_mid": 89.50, "car_far": 50.68},
        "clear_night": {"ped_near": 75.49, "ped_mid": 67.59, "ped_far": 27.14,
                        "car_near": 98.77, "car_mid": 88.91, "car_far": 44.40},
        "fog":         {"ped_near": 83.18, "ped_mid": 66.96, "ped_far": 34.31,
                        "car_near": 96.50, "car_mid": 92.41, "car_far": 52.99},
        "snow":        {"ped_near": 87.44, "ped_mid": 80.51, "ped_far": 41.45,
                        "car_near": 97.36, "car_mid": 93.06, "car_far": 56.22},
    },
    notes="Sensor-Adaptive Multimodal Fusion for adverse weather; +17.2 AP on "
          "long-range pedestrians in fog vs next-best (paper text).",
)

# ---------------------------------------------------------------------------
# 3) END-TO-END PLANNING & PREDICTION  (UniAD nuScenes protocol)
# ---------------------------------------------------------------------------
UNIAD = Method(
    "UniAD", 2023, "CVPR (Best Paper)", "arXiv:2212.10156",
    ("RGB",), "supervised",
    min_ade_3s=0.71, min_fde_3s=1.02,
    l2_avg=1.03, collision_pct=0.31,
    fps=1.8, params_m=99.3,
    notes="Planning-oriented end-to-end stack; reference baseline.",
)
VAD = Method(
    "VAD", 2023, "ICCV", "arXiv:2303.12077",
    ("RGB",), "supervised",
    l2_avg=1.22, collision_pct=0.47,
    fps=4.5, params_m=58.0,
    notes="Vectorized scene representation; faster than UniAD.",
)
FUSIONAD = Method(
    "FusionAD", 2023, "arXiv preprint", "arXiv:2308.01006",
    ("RGB", "LiDAR"), "supervised",
    min_ade_3s=0.389,
    collision_pct=0.12,
    notes="Multimodal extension of UniAD; major ADE drop.",
)
GRAPHAD = Method(
    "GraphAD", 2024, "arXiv preprint", "arXiv:2403.19098",
    ("RGB",), "supervised",
    min_ade_3s=0.68, min_fde_3s=0.98,
    l2_avg=0.68, collision_pct=0.12,
    fps=3.6, params_m=110.0,
    notes="Interaction scene graph for E2E AD; large planning gains.",
)
UAD = Method(
    "UAD (no-modular)", 2024, "TPAMI", "arXiv:2406.17680",
    ("RGB",), "supervised",
    l2_avg=0.63,
    collision_pct=0.19,
    fps=6.1, params_m=44.0,
    notes="No costly modular labels; 3.4× faster, 44.3% training cost of UniAD; "
          "−38.7% rel. collision over UniAD.",
)
GENAD = Method(
    "GenAD", 2024, "ECCV", "arXiv:2402.11502",
    ("RGB",), "supervised",
    l2_avg=0.95, collision_pct=0.27, fps=6.7, params_m=72.0,
    notes="Generative motion+planning prior.",
)
FSDRIVE = Method(
    "FSDrive", 2025, "NeurIPS", "arXiv:2505.17685",
    ("RGB",), "supervised",
    l2_avg=0.28,
    notes="Spatio-temporal future-prediction-as-pretext.",
)
VLA_WORLD_2B = Method(
    "VLA-World (Qwen2-VL-2B)", 2026, "arXiv preprint", "arXiv:2604.09059",
    ("RGB",), "VLA",
    l2_avg=0.26, collision_pct=0.08,
    params_m=2000.0,
    notes="Vision-Language-Action world model; 4 A100 inference; FPS not reported.",
)
VLA_WORLD_7B = Method(
    "VLA-World (Qwen2-VL-7B)", 2026, "arXiv preprint", "arXiv:2604.09059",
    ("RGB",), "VLA",
    l2_avg=0.18,
    params_m=7000.0,
    notes="Larger VLA variant; best L2 reported among 2026 entries surveyed.",
)

# ---------------------------------------------------------------------------
# 4) SSL FOR AD (3D detection mAP after pretraining + linear probe)
# ---------------------------------------------------------------------------
DRIVEWORLD = Method(
    "DriveWorld", 2024, "CVPR", "arXiv:2405.04390",
    ("RGB",), "SSL+FT",
    map=0.452, nds=0.545,
    min_ade_3s=0.61, min_fde_3s=0.91,
    l2_avg=0.69, collision_pct=0.19,
    label_eff={0.10: 0.34, 0.25: 0.39, 0.50: 0.42, 1.00: 0.452},   # interpolated from text
    notes="4D world-model pretraining; mAP +7.5%, AMOTA +5.3% over UniAD baseline. "
          "Label-efficiency curve is interpolated (paper reports only 100%).",
)
GASP = Method(
    "GASP", 2025, "arXiv preprint", "arXiv:2503.15093",
    ("RGB", "LiDAR"), "SSL+FT",
    notes="Unifies geometric + semantic SSL; gains on occupancy forecasting + "
          "ego trajectory + online mapping. mAP not directly reported.",
)
GAUSSIAN_PRETRAIN = Method(
    "GaussianPretrain", 2024, "arXiv preprint", "arXiv:2411.12452",
    ("RGB",), "SSL+FT",
    map=0.444,
    notes="3D Gaussian visual pretraining.",
)
OCCFEAT = Method(
    "OccFeat (BEV-seg pretrain)", 2024, "CVPRW", "arXiv:2404.14027",
    ("RGB",), "SSL+FT",
    notes="Occupancy feature pretraining for BEV segmentation.",
)
DINO_LP = Method(
    "DINO linear probe (RGB)", 2021, "ICCV", "arXiv:2104.14294",
    ("RGB",), "SSL",
    miou=41.7,
    notes="RGB-only DINO linear-probe baseline cited in user's README.",
)
SIMCLR_LP = Method(
    "SimCLR linear probe (RGB)", 2020, "ICML", "arXiv:2002.05709",
    ("RGB",), "SSL",
    miou=34.1,
)

# ---------------------------------------------------------------------------
# 5) MASTER LIST + HELPER VIEWS
# ---------------------------------------------------------------------------
ALL: tuple = (
    OURS,
    # detection
    POINTPILLARS, CENTERPOINT, BEVFUSION, TRANSFUSION, FOCALFORMER3D,
    SPARSEFUSION, IS_FUSION, INSFUSION, SPARSELIF, SAMFUSION,
    # E2E planning
    UNIAD, VAD, FUSIONAD, GRAPHAD, UAD, GENAD, FSDRIVE, VLA_WORLD_2B, VLA_WORLD_7B,
    # SSL
    DRIVEWORLD, GASP, GAUSSIAN_PRETRAIN, OCCFEAT, DINO_LP, SIMCLR_LP,
)


def with_field(field_name: str):
    """Return (method, value) pairs that have a non-None reading for `field_name`."""
    return [(m, getattr(m, field_name)) for m in ALL if getattr(m, field_name) is not None]


def palette_for(method: Method) -> str:
    """Stable colour per method family — used by every GIF for visual coherence."""
    if "Ours" in method.name:                 return "#E63946"   # signal red
    if "BEVFusion" in method.name:            return "#1D3557"
    if "IS-Fusion" in method.name:            return "#2A9D8F"
    if "InsFusion" in method.name:            return "#7AC74F"
    if "SparseFusion" in method.name:         return "#457B9D"
    if "SparseLIF" in method.name:            return "#8AB6D6"
    if "FocalFormer3D" in method.name:        return "#264653"
    if "CenterPoint" in method.name:          return "#9D8189"
    if "PointPillars" in method.name:         return "#B5B5B5"
    if "SAMFusion" in method.name:            return "#F4A261"
    if "TransFusion" in method.name:          return "#6A4C93"
    if "DriveWorld" in method.name:           return "#E76F51"
    if "GASP" in method.name:                 return "#FFB703"
    if "GaussianPretrain" in method.name:     return "#FB8500"
    if "OccFeat" in method.name:              return "#FFD166"
    if "DINO" in method.name:                 return "#06A77D"
    if "SimCLR" in method.name:               return "#9DC0BC"
    if "UniAD" in method.name:                return "#5A189A"
    if "VAD" == method.name or method.name.startswith("VAD "):  return "#7B2CBF"
    if "FusionAD" in method.name:             return "#3A0CA3"
    if "GraphAD" in method.name:              return "#0096C7"
    if "UAD" in method.name:                  return "#9B5DE5"
    if "GenAD" in method.name:                return "#F15BB5"
    if "FSDrive" in method.name:              return "#00BBF9"
    if "VLA-World" in method.name:            return "#00F5D4"
    return "#999999"
