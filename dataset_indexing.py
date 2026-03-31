"""
=============================================================================
02_dataset_indexing.py  —  Folder Scan + Balanced Dataset Indexing
=============================================================================
FIXES vs original:
  • detect_modality() BUG FIXED: FLIR RGB files are in images_rgb_train/data/
    They have NO "cam" or "/samples/" in path → were silently dropped as "other"
  • All 7 trainval blob folders (trainval01..07_blobs) explicitly scanned
  • Only SENSOR DATA scanned (iCars, Geolife, OSM maps excluded)
  • CAPS enforce balanced modality counts; WeightedRandomSampler weights saved
  • Outputs written to OUTPUT_ROOT (not DATA_ROOT)
  • build_fusion_sync_index() added for time-synced 4-modality fusion training
=============================================================================
"""
import json, random
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from setup import (
    OUTPUT_ROOT, PROGRESS_ROOT, FLIR_ROOT, NUS_ROOT,
    logger, read_json, write_json, load_progress, save_progress
)

# ── Output paths (all in OUTPUT_ROOT) ─────────────────────────────────────
OUT_INDEX        = OUTPUT_ROOT / "unified_dataset.json"
OUT_COUNTS       = OUTPUT_ROOT / "unified_counts.json"
OUT_WEIGHTS      = OUTPUT_ROOT / "sample_weights.json"
INDEX_FLAG       = PROGRESS_ROOT / "index_complete.flag"

FUSION_INDEX_PATH = OUTPUT_ROOT / "fusion_sync_index.json"
FUSION_INDEX_FLAG = PROGRESS_ROOT / "fusion_index_complete.flag"

# ── Per-modality training caps ─────────────────────────────────────────────
# Goal: roughly equal representation so no modality dominates SSL training.
# RGB total available ~124K → cap at 25K
# Thermal total ~10.7K → use all
# LiDAR raw trainval ~23.9K → use all (one sensor per blob)
# Radar raw trainval ~23.9K → use all
CAPS = {
    "rgb":     25_000,
    "thermal": 11_000,   # slightly above actual count → use all
    "lidar":   24_000,
    "radar":   24_000,
}

# ── Explicit source directory map ─────────────────────────────────────────
# Each entry: (directory_path, extension, modality)
# Only real sensor data is listed here. No iCars, no OSM, no Geolife.
def _source_dirs() -> list:
    sources = []

    # ── FLIR RGB (images_rgb_train/data/*.jpg) ──
    for split in ["images_rgb_train", "images_rgb_val"]:
        d = FLIR_ROOT / split / "data"
        if d.exists():
            sources.append((d, ".jpg", "rgb"))
            logger.info(f"  RGB    source: {d}")

    # ── FLIR Thermal (images_thermal_train/analyticsData/*.tiff) ──
    for split in ["images_thermal_train", "images_thermal_val"]:
        d = FLIR_ROOT / split / "analyticsData"
        if d.exists():
            sources.append((d, ".tiff", "thermal"))
            logger.info(f"  Therm  source: {d}")

    # ── nuScenes LiDAR raw scans (trainval blobs 01-07 + mini) ──
    for i in range(1, 8):
        d = NUS_ROOT / f"v1.0-trainval0{i}_blobs" / "samples" / "LIDAR_TOP"
        if d.exists():
            sources.append((d, ".bin", "lidar"))
    d = NUS_ROOT / "v1.0-mini" / "samples" / "LIDAR_TOP"
    if d.exists():
        sources.append((d, ".bin", "lidar"))
    logger.info("  LiDAR  sources: trainval01-07_blobs/LIDAR_TOP + mini")

    # ── nuScenes Radar (RADAR_FRONT from trainval blobs + mini) ──
    for i in range(1, 8):
        d = NUS_ROOT / f"v1.0-trainval0{i}_blobs" / "samples" / "RADAR_FRONT"
        if d.exists():
            sources.append((d, ".pcd", "radar"))
    d = NUS_ROOT / "v1.0-mini" / "samples" / "RADAR_FRONT"
    if d.exists():
        sources.append((d, ".pcd", "radar"))
    logger.info("  Radar  sources: trainval01-07_blobs/RADAR_FRONT + mini")

    return sources


# ── Step 1: Build unified_dataset.json ────────────────────────────────────
def run_dataset_indexing() -> list:
    if INDEX_FLAG.exists():
        logger.info("Dataset index already built. Loading.")
        return read_json(OUT_INDEX)

    logger.info("Building dataset index...")
    all_files: dict = {m: [] for m in CAPS}

    for directory, ext, modality in _source_dirs():
        files = [str(f) for f in directory.rglob(f"*{ext}")]
        all_files[modality].extend(files)

    # Deduplicate
    for m in all_files:
        all_files[m] = sorted(set(all_files[m]))
        logger.info(f"  {m:8}: {len(all_files[m]):,} total files found")

    # Balanced sampling: cap each modality
    rng = random.Random(42)
    final_entries, counts = [], {}
    for modality, files in all_files.items():
        cap      = CAPS.get(modality, 5000)
        selected = rng.sample(files, cap) if len(files) > cap else files[:]
        counts[modality] = len(selected)
        for p in selected:
            final_entries.append({"modality": modality, "path": p})

    # Compute per-sample weights for WeightedRandomSampler
    total     = sum(counts.values())
    n_mod     = len([m for m in counts if counts[m] > 0])
    mod_weight = {m: total / (n_mod * max(counts[m], 1)) for m in counts}
    weights    = [mod_weight[e["modality"]] for e in final_entries]

    # Shuffle deterministically
    pairs = list(zip(final_entries, weights))
    rng.shuffle(pairs)
    final_entries, weights = zip(*pairs) if pairs else ([], [])
    final_entries = list(final_entries)
    weights       = list(weights)

    write_json(OUT_INDEX,   final_entries)
    write_json(OUT_COUNTS,  {"counts": counts, "caps": CAPS, "total": len(final_entries)})
    write_json(OUT_WEIGHTS, weights)
    INDEX_FLAG.touch()

    logger.info("\nDataset index summary:")
    for m, c in counts.items():
        logger.info(f"  {m:8} | {c:>6,} | weight={mod_weight[m]:.3f}")
    logger.info(f"  {'TOTAL':8} | {len(final_entries):>6,}")
    return final_entries


# ── Step 2: Build time-synced fusion index from v1.0-mini ─────────────────
# Fusion Transformer must see ALL 4 modalities from the SAME frame.
# Only v1.0-mini has verified sample-token alignment across all sensors.
def build_fusion_sync_index() -> list:
    if FUSION_INDEX_FLAG.exists():
        return read_json(FUSION_INDEX_PATH)

    mini = NUS_ROOT / "v1.0-mini"
    if not mini.exists():
        logger.warning("v1.0-mini not found — fusion sync index empty.")
        write_json(FUSION_INDEX_PATH, [])
        FUSION_INDEX_FLAG.touch()
        return []

    samples_dir = mini / "samples"
    meta_json   = mini / "v1.0-mini" / "sample_data.json"

    synced = []
    if meta_json.exists():
        try:
            sample_data = read_json(meta_json)
            token_map: dict = {}
            for sd in sample_data:
                ch    = sd.get("channel", "")
                tok   = sd.get("sample_token", "")
                fname = sd.get("filename", "")
                if not fname: continue
                full = str(NUS_ROOT / "v1.0-mini" / fname)
                if   "CAM_FRONT"   == ch: token_map.setdefault(tok, {})["rgb"]   = full
                elif "LIDAR_TOP"   == ch: token_map.setdefault(tok, {})["lidar"] = full
                elif "RADAR_FRONT" == ch: token_map.setdefault(tok, {})["radar"] = full
            for tok, s in token_map.items():
                if "rgb" in s and "lidar" in s and "radar" in s:
                    synced.append({"rgb": s["rgb"], "thermal": None,
                                   "lidar": s["lidar"], "radar": s["radar"],
                                   "token": tok})
        except Exception as e:
            logger.warning(f"Meta-based sync failed: {e}, using positional fallback")

    if not synced:
        # Positional fallback
        cams   = sorted((samples_dir / "CAM_FRONT").glob("*.jpg"))
        lidars = sorted((samples_dir / "LIDAR_TOP").glob("*.bin"))
        radars = sorted((samples_dir / "RADAR_FRONT").glob("*.pcd"))
        n      = min(len(cams), len(lidars), len(radars))
        synced = [{"rgb": str(cams[i]), "thermal": None,
                   "lidar": str(lidars[i]), "radar": str(radars[i]),
                   "token": str(i)} for i in range(n)]

    write_json(FUSION_INDEX_PATH, synced)
    FUSION_INDEX_FLAG.touch()
    logger.info(f"Fusion sync index: {len(synced)} scenes")
    return synced


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    entries = run_dataset_indexing()
    synced  = build_fusion_sync_index()
    logger.info(f"Done. SSL entries: {len(entries)}, Fusion scenes: {len(synced)}")
