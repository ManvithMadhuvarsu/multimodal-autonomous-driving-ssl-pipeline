"""
================================================================================
  prepare_inference_inputs.py  v2
================================================================================
  Selects TIME-SYNCED frames from ALL 6 nuScenes cameras + LiDAR + Radar and
  copies them into:
    D:\Mtech\Sem_4\output\inference\input\

  Camera layout (matches the 6-panel GIF grid in inference_pipeline_fixed.py):

      CAM_FRONT_LEFT    CAM_FRONT       CAM_FRONT_RIGHT
      CAM_BACK_LEFT     CAM_BACK        CAM_BACK_RIGHT

  All 6 cameras are time-synchronised in nuScenes v1.0-mini — every index i
  across all 6 folders corresponds to the exact same scene timestamp.

  USAGE:
    python prepare_inference_inputs.py               # 25 frames, nuScenes only
    python prepare_inference_inputs.py --n 50
    python prepare_inference_inputs.py --source flir --n 25
    python prepare_inference_inputs.py --source both --n 25
    python prepare_inference_inputs.py --n 404       # full v1.0-mini
================================================================================
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path(r"D:\Mtech\Sem_3\Case Study\Data")
OUTPUT_ROOT = Path(r"D:\Mtech\Sem_4\output\inference")
INPUT_DIR   = OUTPUT_ROOT / "input"

# ── Dataset sub-paths ─────────────────────────────────────────────────────────
FLIR_ROOT  = DATA_ROOT / "FLIR_ADAS"
NUS_ROOT   = DATA_ROOT / "NuScences"

FLIR_RGB_TRAIN     = FLIR_ROOT / "images_rgb_train"     / "data"
FLIR_RGB_VAL       = FLIR_ROOT / "images_rgb_val"       / "data"
FLIR_THERMAL_TRAIN = FLIR_ROOT / "images_thermal_train" / "analyticsData"
FLIR_THERMAL_VAL   = FLIR_ROOT / "images_thermal_val"   / "analyticsData"

NUS_MINI = NUS_ROOT / "v1.0-mini"

# All 6 cameras — ordered to match the 3×2 collage grid:
#   [0] FL  [1] F   [2] FR
#   [3] BL  [4] B   [5] BR
CAM_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

# Grid labels for panel composition
CAM_LABELS = {
    "CAM_FRONT_LEFT":  "Front Left",
    "CAM_FRONT":       "Front",
    "CAM_FRONT_RIGHT": "Front Right",
    "CAM_BACK_LEFT":   "Back Left",
    "CAM_BACK":        "Back",
    "CAM_BACK_RIGHT":  "Back Right",
}

NUS_LIDAR = NUS_MINI / "samples" / "LIDAR_TOP"
NUS_RADAR = NUS_MINI / "samples" / "RADAR_FRONT"

RGB_EXTS     = {".jpg", ".jpeg", ".png"}
THERMAL_EXTS = {".tiff", ".tif"}
LIDAR_EXTS   = {".bin"}
RADAR_EXTS   = {".pcd"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_path(p: Path, label: str) -> bool:
    if not p.exists():
        print(f"  ✗  {label}")
        print(f"     Not found: {p}")
        return False
    print(f"  ✓  {label}  ({p})")
    return True


def collect_sorted(folder: Path, exts: set, limit: int | None = None) -> list[Path]:
    if not folder.exists():
        return []
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
    return files[:limit] if limit else files


def copy_files(files: list[Path], dest: Path, label: str) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, src in enumerate(files):
        dst = dest / f"{i:04d}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        manifest.append({"index": i, "source": str(src), "dest": str(dst)})
    print(f"  → {label}: {len(files)} file(s)  →  {dest.name}/")
    return manifest


# ── nuScenes — all 6 cameras ──────────────────────────────────────────────────

def prepare_nuscenes(n: int, manifest: dict) -> None:
    print("\n[nuScenes v1.0-mini  — all 6 cameras + LiDAR + Radar]")

    cam_dirs = {cam: NUS_MINI / "samples" / cam for cam in CAM_NAMES}
    lidar_dir = NUS_LIDAR
    radar_dir = NUS_RADAR

    # Check all paths
    all_ok = True
    for cam, path in cam_dirs.items():
        if not check_path(path, f"{cam}"):
            all_ok = False
    check_path(lidar_dir, "LIDAR_TOP")
    check_path(radar_dir, "RADAR_FRONT")

    if not all_ok:
        print("  ✗  One or more camera paths missing — skipping nuScenes.")
        return

    # Collect + trim to n consecutive frames per camera
    cam_files: dict[str, list[Path]] = {}
    min_count = n
    for cam, cdir in cam_dirs.items():
        files = collect_sorted(cdir, RGB_EXTS)
        cam_files[cam] = files[:n]
        min_count = min(min_count, len(files))

    actual_n = min(n, min_count)
    for cam in CAM_NAMES:
        cam_files[cam] = cam_files[cam][:actual_n]

    lidar_files = collect_sorted(lidar_dir, LIDAR_EXTS)[:actual_n]
    radar_files = collect_sorted(radar_dir, RADAR_EXTS)[:actual_n]

    print(f"\n  Selecting {actual_n} consecutive frames across all 6 cameras.")

    cam_manifests = {}
    for cam in CAM_NAMES:
        dest = INPUT_DIR / f"cam_{cam.lower()}"
        cam_manifests[cam] = copy_files(cam_files[cam], dest, cam)

    # Build the flat_rgb folder — ALL 6 cameras interleaved in timestamp order.
    # Frame ordering: frame0_FL, frame0_F, frame0_FR, frame0_BL, frame0_B, frame0_BR,
    #                 frame1_FL, ...
    # This lets the pipeline process them as a sequence while keeping temporal sync.
    flat_dir = INPUT_DIR / "flat_rgb"
    flat_dir.mkdir(parents=True, exist_ok=True)
    flat_entries = []
    global_idx = 0
    for t in range(actual_n):
        for cam in CAM_NAMES:
            src_path = INPUT_DIR / f"cam_{cam.lower()}" / f"{t:04d}_{cam_files[cam][t].name}"
            dst_name = f"{global_idx:04d}_{cam}_{cam_files[cam][t].name}"
            dst_path = flat_dir / dst_name
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
            flat_entries.append({
                "global_index": global_idx,
                "timestamp":    t,
                "camera":       cam,
                "cam_label":    CAM_LABELS[cam],
                "source":       str(cam_files[cam][t]),
                "dest":         str(dst_path),
            })
            global_idx += 1

    print(f"  → flat_rgb/: {global_idx} files  (6 cameras × {actual_n} frames, interleaved)")

    # Also build a per-timestamp collage index so inference knows how to group them
    collage_index = []
    for t in range(actual_n):
        collage_index.append({
            "timestamp": t,
            "cameras": {
                cam: str(INPUT_DIR / f"cam_{cam.lower()}" / f"{t:04d}_{cam_files[cam][t].name}")
                for cam in CAM_NAMES
            }
        })

    collage_path = INPUT_DIR / "collage_index.json"
    with open(collage_path, "w") as f:
        json.dump(collage_index, f, indent=2)
    print(f"  → collage_index.json: {actual_n} timestamps")

    copy_files(lidar_files, INPUT_DIR / "lidar_nuscenes", "LIDAR_TOP (reference)")
    copy_files(radar_files, INPUT_DIR / "radar_nuscenes", "RADAR_FRONT (reference)")

    manifest["nuscenes"] = {
        "source_root":       str(NUS_MINI),
        "n_frames":          actual_n,
        "cameras":           CAM_NAMES,
        "cam_labels":        CAM_LABELS,
        "consecutive":       True,
        "flat_rgb_count":    global_idx,
        "flat_rgb_dir":      str(flat_dir),
        "collage_index":     str(collage_path),
        "cam_manifests":     cam_manifests,
        "collage_layout":    "3x2 — [FL, F, FR] / [BL, B, BR]",
    }


# ── FLIR ADAS ─────────────────────────────────────────────────────────────────

def prepare_flir(n: int, manifest: dict) -> None:
    print("\n[FLIR ADAS — RGB + Thermal]")

    rgb_src = FLIR_RGB_TRAIN if FLIR_RGB_TRAIN.exists() else FLIR_RGB_VAL
    thr_src = FLIR_THERMAL_TRAIN if FLIR_THERMAL_TRAIN.exists() else FLIR_THERMAL_VAL

    if not check_path(rgb_src, f"FLIR RGB  ({rgb_src.name})"):
        print("  Skipping FLIR — RGB path missing.")
        return
    check_path(thr_src, f"FLIR Thermal ({thr_src.name})")

    rgb_files = collect_sorted(rgb_src, RGB_EXTS)[:n]
    thr_files = collect_sorted(thr_src, THERMAL_EXTS)[:n]
    actual_n  = min(len(rgb_files), len(thr_files))
    rgb_files, thr_files = rgb_files[:actual_n], thr_files[:actual_n]
    print(f"\n  Selecting {actual_n} paired RGB + Thermal frames (consecutive)")

    rgb_dest = INPUT_DIR / "rgb_flir"
    rgb_m    = copy_files(rgb_files, rgb_dest, "FLIR RGB → input/rgb_flir")
    thr_m    = copy_files(thr_files, INPUT_DIR / "thermal_flir",
                          "FLIR Thermal → input/thermal_flir")

    # Merge FLIR RGB into flat_rgb too (after nuScenes if both selected)
    flat_dir = INPUT_DIR / "flat_rgb"
    flat_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(flat_dir.glob("*.jpg"))) + len(list(flat_dir.glob("*.png")))
    for i, src_path in enumerate(sorted(rgb_dest.glob("*.jpg"))):
        dst = flat_dir / f"{existing + i:04d}_FLIR_{src_path.name}"
        if not dst.exists():
            shutil.copy2(src_path, dst)
    print(f"  → flat_rgb/: FLIR frames appended  (+{actual_n})")

    manifest["flir"] = {
        "source_rgb":     str(rgb_src),
        "source_thermal": str(thr_src),
        "n_frames":       actual_n,
        "rgb":            rgb_m,
        "thermal":        thr_m,
        "note": "FLIR has only 1 camera (front-facing). No 6-cam collage for FLIR frames.",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare all-camera inference input from nuScenes + FLIR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_inference_inputs.py                      # 25 nuScenes frames (all 6 cams)
  python prepare_inference_inputs.py --n 50
  python prepare_inference_inputs.py --source flir --n 25
  python prepare_inference_inputs.py --source both --n 25
  python prepare_inference_inputs.py --n 404              # full v1.0-mini
        """)
    p.add_argument("--source", default="nuscenes",
                   choices=["nuscenes", "flir", "both"])
    p.add_argument("--n", type=int, default=25,
                   help="Consecutive frames per camera/source (default: 25)")
    p.add_argument("--data-root",   default=None)
    p.add_argument("--output-root", default=None)
    args = p.parse_args()

    global DATA_ROOT, OUTPUT_ROOT, INPUT_DIR
    global FLIR_ROOT, NUS_ROOT, NUS_MINI
    global FLIR_RGB_TRAIN, FLIR_RGB_VAL, FLIR_THERMAL_TRAIN, FLIR_THERMAL_VAL
    global NUS_LIDAR, NUS_RADAR

    if args.data_root:
        DATA_ROOT = Path(args.data_root)
    if args.output_root:
        OUTPUT_ROOT = Path(args.output_root)

    INPUT_DIR          = OUTPUT_ROOT / "input"
    FLIR_ROOT          = DATA_ROOT / "FLIR_ADAS"
    NUS_ROOT           = DATA_ROOT / "NuScences"
    NUS_MINI           = NUS_ROOT / "v1.0-mini"
    FLIR_RGB_TRAIN     = FLIR_ROOT / "images_rgb_train" / "data"
    FLIR_RGB_VAL       = FLIR_ROOT / "images_rgb_val"   / "data"
    FLIR_THERMAL_TRAIN = FLIR_ROOT / "images_thermal_train" / "analyticsData"
    FLIR_THERMAL_VAL   = FLIR_ROOT / "images_thermal_val"   / "analyticsData"
    NUS_LIDAR          = NUS_MINI / "samples" / "LIDAR_TOP"
    NUS_RADAR          = NUS_MINI / "samples" / "RADAR_FRONT"

    print("═" * 70)
    print("  INFERENCE INPUT PREPARATION  (6-Camera + LiDAR + Radar)")
    print("═" * 70)
    print(f"  Data root   : {DATA_ROOT}")
    print(f"  Output root : {OUTPUT_ROOT}")
    print(f"  Source      : {args.source}")
    print(f"  Frames (n)  : {args.n} per camera (× 6 = {args.n * 6} total flat_rgb files for nuScenes)")
    print("═" * 70)

    manifest: dict = {
        "data_root":   str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "source":      args.source,
        "n_requested": args.n,
        "cam_layout":  "3x2 — row1:[FL, F, FR]  row2:[BL, B, BR]",
    }

    if args.source in ("nuscenes", "both"):
        prepare_nuscenes(args.n, manifest)
    if args.source in ("flir", "both"):
        prepare_flir(args.n, manifest)

    flat_dir   = INPUT_DIR / "flat_rgb"
    output_dir = OUTPUT_ROOT / "output"

    manifest_path = INPUT_DIR / "input_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  ✓  Manifest saved → {manifest_path}")

    # ── Print run commands ────────────────────────────────────────────────────
    collage_index = INPUT_DIR / "collage_index.json"
    print("\n" + "═" * 70)
    print("  READY — two ways to run inference:")
    print("═" * 70)
    print("""
  Option A — Full 6-camera collage mode (generates 3×2 grid GIF):
    python inference_pipeline_fixed.py \\
      --input  "{flat}" \\
      --output "{out}" \\
      --collage-index "{ci}" \\
      --device cuda \\
      --gif-fps 4

  Option B — Single-camera pass (original flat mode):
    python inference_pipeline_fixed.py \\
      --input  "{flat}/cam_front_only/" \\
      --output "{out}" \\
      --device cuda \\
      --gif-fps 4
""".format(flat=flat_dir, out=output_dir, ci=collage_index))

    print("  Staged folder structure:")
    if INPUT_DIR.exists():
        for child in sorted(INPUT_DIR.iterdir()):
            if child.is_dir():
                count = sum(1 for _ in child.iterdir())
                print(f"    input/{child.name}/  ({count} files)")
    print()


if __name__ == "__main__":
    main()
