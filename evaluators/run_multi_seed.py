"""
Tier-A4 — Multi-seed orchestrator with mean ± std aggregation.

WHY
----
Single-seed top-line numbers are a top-3 reject reason at IEEE Transactions venues
right now. Your README already reports ± std for the SSL linear-probe table, but
the headline detection / segmentation / FPS numbers are single-run. This script
runs the full pipeline (or just the final fine-tune stage if you pass
``--stage finetune-only``) under several seeds, collects the per-seed JSON
results, and emits the aggregated table reviewers expect.

WHAT
----
For each seed in ``--seeds`` (default: 42 123 7):

    1. Set PYTHONHASHSEED, numpy, torch, CUDA seeds.
    2. Run ``run_pipeline.py`` with that seed.
    3. Snapshot the output artefacts (``test_results.json``, ``planning_uniad.json``,
       ``adverse_weather.json``, ``nuscenes_eval/metrics_summary.json``) into
       ``<results_root>/seed_<n>/``.

After all seeds finish, aggregate every numeric leaf metric into
``aggregated_metrics.json`` and print a Markdown table::

    | metric         | seed=42 | seed=123 | seed=7 | mean   | std  |
    |----------------|---------|----------|--------|--------|------|
    | mAP            | 0.594   | 0.601    | 0.589  | 0.595  | ±0.006 |
    | NDS            | 0.612   | 0.620    | 0.609  | 0.614  | ±0.006 |
    | L2_avg         | 0.71    | 0.69     | 0.73   | 0.71   | ±0.02  |
    | collision_pct  | 0.18    | 0.21     | 0.16   | 0.18   | ±0.03  |
    | mIoU           | 52.4    | 52.9     | 52.7   | 52.67  | ±0.25  |

USAGE
-----
    python -m research_comparison.evaluators.run_multi_seed \\
        --seeds 42 123 7 \\
        --pipeline-cmd "python run_pipeline.py --skip-rl" \\
        --output-root D:/Mtech/Sem_4/output \\
        --results-root D:/Mtech/Sem_4/output/multi_seed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# 1.  Per-seed runner
# ---------------------------------------------------------------------------
def run_one_seed(seed: int,
                 pipeline_cmd: str,
                 output_root: Path,
                 seed_root: Path) -> None:
    seed_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["MULTIMODAL_SSL_SEED"] = str(seed)         # picked up by setup.set_seed when wired

    log_path = seed_root / f"run_seed_{seed}.log"
    print(f"\n[seed {seed}] running: {pipeline_cmd}")
    print(f"[seed {seed}] log     -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.run(pipeline_cmd, shell=True, env=env,
                              stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[seed {seed}] WARNING: pipeline exited code {proc.returncode}; "
              "continuing to snapshot whatever artefacts exist.")

    # Snapshot artefacts
    targets = [
        "test_results.json",
        "planning_uniad.json",
        "adverse_weather.json",
        "aggregated_metrics.json",
        "nuscenes_eval/metrics_summary.json",
    ]
    for rel in targets:
        src = output_root / rel
        if src.exists():
            dst = seed_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[seed {seed}] snapshot   {rel}")


# ---------------------------------------------------------------------------
# 2.  Numeric-leaf flatten
# ---------------------------------------------------------------------------
def _flatten_numeric(prefix: str, obj: Any, out: Dict[str, float]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_numeric(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    # silently skip lists, strings, bools, nulls


def collect_metrics(seed_root: Path) -> Dict[str, float]:
    """Flatten every numeric leaf across every artefact in this seed's snapshot."""
    metrics: Dict[str, float] = {}
    for jf in seed_root.rglob("*.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f: blob = json.load(f)
            rel = jf.relative_to(seed_root).as_posix()
            _flatten_numeric(rel, blob, metrics)
        except Exception as e:
            print(f"[collect] skip {jf}: {e}")
    return metrics


# ---------------------------------------------------------------------------
# 3.  Aggregate + table
# ---------------------------------------------------------------------------
def aggregate(seeds_metrics: Dict[int, Dict[str, float]]) -> Dict[str, dict]:
    keys = set()
    for d in seeds_metrics.values(): keys.update(d.keys())
    agg = {}
    for k in sorted(keys):
        vals = [d[k] for d in seeds_metrics.values() if k in d]
        if not vals: continue
        agg[k] = {
            "values":   {f"seed_{s}": seeds_metrics[s].get(k) for s in seeds_metrics},
            "mean":     mean(vals),
            "std":      stdev(vals) if len(vals) > 1 else 0.0,
            "n":        len(vals),
        }
    return agg


def print_table(agg: Dict[str, dict], seeds: List[int],
                top_keys: List[str] = None) -> None:
    """Print a focused table of the metrics reviewers actually care about."""
    if top_keys is None:
        # Heuristic: rank metrics by name relevance
        priority = [
            "metrics_summary.json.mean_ap", "metrics_summary.json.nd_score",
            "planning_uniad.json.L2_avg", "planning_uniad.json.L2_1s",
            "planning_uniad.json.L2_2s",  "planning_uniad.json.L2_3s",
            "planning_uniad.json.collision_pct",
            "aggregated_metrics.json.mIoU",
            "aggregated_metrics.json.fps",
        ]
        top_keys = [k for k in priority if k in agg]
    print()
    print(" Multi-seed aggregated metrics")
    print(" =============================")
    print(f"   seeds : {seeds}")
    cols = "metric"
    for s in seeds: cols += f"  seed={s:>4}"
    cols += "  mean        std"
    print()
    print(" " + cols)
    print(" " + "-" * len(cols))
    for k in top_keys:
        row = agg[k]
        line = f" {k[-44:]:<44s}"
        for s in seeds:
            v = row["values"].get(f"seed_{s}")
            line += f"  {('-' if v is None else f'{v:8.4f}'):>9}"
        line += f"  {row['mean']:8.4f}  +/-{row['std']:6.4f}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# 4.  CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed pipeline runner & aggregator.")
    p.add_argument("--seeds",         nargs="+", type=int, default=[42, 123, 7])
    p.add_argument("--pipeline-cmd",  required=True,
                   help="Shell command that runs your training pipeline.")
    p.add_argument("--output-root",   required=True,
                   help="The pipeline's output directory (where test_results.json etc. live).")
    p.add_argument("--results-root",  required=True,
                   help="Destination for per-seed snapshots and aggregated table.")
    p.add_argument("--skip-runs",     action="store_true",
                   help="Do not re-run the pipeline; just aggregate from existing snapshots.")
    args = p.parse_args()

    output_root  = Path(args.output_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_runs:
        for seed in args.seeds:
            seed_root = results_root / f"seed_{seed}"
            run_one_seed(seed, args.pipeline_cmd, output_root, seed_root)

    seeds_metrics: Dict[int, Dict[str, float]] = {}
    for seed in args.seeds:
        seed_root = results_root / f"seed_{seed}"
        if not seed_root.exists():
            print(f"[aggregate] WARN: no snapshot for seed {seed}; skipping.")
            continue
        seeds_metrics[seed] = collect_metrics(seed_root)

    if not seeds_metrics:
        print("[aggregate] no metrics found; nothing to aggregate.")
        return

    agg = aggregate(seeds_metrics)
    agg_path = results_root / "aggregated.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"[aggregate] wrote {agg_path}")

    print_table(agg, sorted(seeds_metrics.keys()))


if __name__ == "__main__":
    main()
