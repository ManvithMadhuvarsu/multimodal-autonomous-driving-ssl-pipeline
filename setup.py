"""
setup.py - Environment setup, paths, logger, loaders, checkpoint manager.
DATA_ROOT   : D:/Mtech/Sem_3/Case Study/Data  (Windows path, update if needed)
OUTPUT_ROOT : D:/Mtech/Sem_4/output
"""

import os, sys, json, time, random, logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

# ── 1. Root directories ────────────────────────────────────────────────────
DATA_ROOT = Path(r"D:\Mtech\Sem_3\Case Study\Data")
if not DATA_ROOT.exists():
    DATA_ROOT = Path("/content/drive/MyDrive/Manvith/Project/Data")
if not DATA_ROOT.exists():
    DATA_ROOT = Path.cwd() / "Data"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

ROOT = DATA_ROOT   # backward-compat alias

OUTPUT_ROOT = Path(r"D:\Mtech\Sem_4\output")
if not OUTPUT_ROOT.exists():
    OUTPUT_ROOT = Path("/content/drive/MyDrive/Manvith/Project/output")
if not OUTPUT_ROOT.exists():
    OUTPUT_ROOT = Path.cwd() / "output"

CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
PROGRESS_ROOT   = OUTPUT_ROOT / "progress"
LOG_ROOT        = OUTPUT_ROOT / "logs"

SSL_DIR    = CHECKPOINT_ROOT / "ssl"
FUSION_DIR = CHECKPOINT_ROOT / "fusion"
GNN_DIR    = CHECKPOINT_ROOT / "gnn"
HEAD_DIR   = CHECKPOINT_ROOT / "heads"
RL_DIR     = CHECKPOINT_ROOT / "rl"

FLIR_ROOT  = DATA_ROOT / "FLIR_ADAS"
NUS_ROOT   = DATA_ROOT / "NuScences"
OTHER_ROOT = DATA_ROOT / "Other_Files"

for _p in (OUTPUT_ROOT, CHECKPOINT_ROOT, PROGRESS_ROOT, LOG_ROOT,
           SSL_DIR, FUSION_DIR, GNN_DIR, HEAD_DIR, RL_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ── 2. Logger (console + rotating file) ───────────────────────────────────
def get_logger(name: str = "pipeline") -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        lg.addHandler(ch)
        fh = logging.FileHandler(LOG_ROOT / "pipeline.log", encoding="utf-8")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    return lg

logger = get_logger()

# ── 3. Seed & Device ──────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except Exception: pass

def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                        f"VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**2} MB")
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return "cpu"

set_seed(42)
DEVICE = get_device()
logger.info(f"Data root  : {DATA_ROOT}")
logger.info(f"Output root: {OUTPUT_ROOT}")
logger.info(f"Device     : {DEVICE}")

# ── 4. JSON helpers ────────────────────────────────────────────────────────
def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

def load_progress(path: Path) -> Dict[str, Any]:
    if path.exists():
        try: return read_json(path)
        except Exception: return {}
    return {}

def save_progress(data: Dict[str, Any], path: Path): write_json(path, data)

# ── 5. Raw data loaders ────────────────────────────────────────────────────
def load_image(path: Path) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(str(path)).convert("RGB"))

def load_tiff(path: Path) -> np.ndarray:
    """FLIR 16-bit LWIR TIFF → float32 normalised to [0,1]."""
    from PIL import Image
    arr = np.array(Image.open(str(path)), dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)

def load_pcd(path: Path):
    import open3d as o3d
    return o3d.io.read_point_cloud(str(path))

def load_bin_lidar(path: Path) -> np.ndarray:
    """nuScenes LiDAR .bin — 5 float32/pt (x,y,z,intensity,ring) → (N,3) xyz."""
    data = np.fromfile(str(path), dtype=np.float32)
    if data.size == 0: return np.zeros((0, 3), dtype=np.float32)
    for stride in (5, 4, 3):
        if data.size % stride == 0:
            return data.reshape(-1, stride)[:, :3]
    return np.zeros((0, 3), dtype=np.float32)

def load_radar_pcd(path: Path) -> np.ndarray:
    """nuScenes Radar .pcd via Open3D → (N,3) xyz float32."""
    try:
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
        return pts if pts.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)

# ── 6. Checkpoint helpers ──────────────────────────────────────────────────
def save_ckpt(path: Path, model=None, optim=None, epoch: int = 0, extra=None):
    import torch
    obj = {"epoch": epoch}
    if model is not None: obj["model"] = model.state_dict()
    if optim  is not None: obj["optim"] = optim.state_dict()
    if extra  is not None: obj["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(path))

def load_ckpt(path: Path, model=None, optim=None):
    import torch
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    if model is not None and "model" in data:
        model.load_state_dict(data["model"], strict=False)
    if optim  is not None and "optim"  in data:
        try: optim.load_state_dict(data["optim"])
        except Exception: pass
    return data.get("epoch", 0), data.get("extra", None)

def find_latest(folder: Path, prefix: str) -> Optional[Path]:
    files = sorted(folder.glob(f"{prefix}_epoch_*.pth"),
                   key=lambda x: x.stat().st_mtime)
    return files[-1] if files else None

logger.info("01_setup ready.")
