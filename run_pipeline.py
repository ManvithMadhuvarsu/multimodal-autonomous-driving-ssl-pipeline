"""
=============================================================================
run_pipeline.py — Master Entry Point: Run Complete ML Pipeline
=============================================================================
FIXES vs original:
  - Python cannot import modules with numeric prefixes (01_setup, etc.)
    All modules renamed to non-numeric names. Import paths fixed here.
  - All 12 stages complete with correct function names
  - --stage flag allows running from a specific stage (useful for resume)

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-rl
  python run_pipeline.py --stage 4      # resume from stage 4
=============================================================================
"""

import argparse
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="QuadFusion-AD Pipeline")
    parser.add_argument("--skip-rl",      action="store_true")
    parser.add_argument("--stage",        type=int, default=1,
                        help="Start from this stage (1-12)")
    parser.add_argument("--ssl-epochs",   type=int, default=30)
    parser.add_argument("--fusion-epochs",type=int, default=10)
    parser.add_argument("--gnn-epochs",   type=int, default=5)
    parser.add_argument("--head-epochs",  type=int, default=3)
    parser.add_argument("--batch-size",   type=int, default=32)
    args = parser.parse_args()

    from setup import OUTPUT_ROOT, logger

    logger.info("=" * 70)
    logger.info("  QuadFusion-AD — Autonomous Driving Multimodal SSL Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Starting from stage {args.stage}")

    def run_stage(n, name, fn, *fargs, **fkwargs):
        if args.stage > n:
            logger.info(f"[Stage {n:2d}] {name} — skipped (--stage {args.stage})")
            return None
        logger.info(f"\n[Stage {n:2d}] {name}")
        return fn(*fargs, **fkwargs)

    # ── Stage 1: Dataset Indexing ──────────────────────────────────────────
    def stage1():
        from dataset_indexing import run_dataset_indexing, build_fusion_sync_index
        run_dataset_indexing()
        build_fusion_sync_index()
        unified_path = OUTPUT_ROOT / "unified_dataset.json"
        assert unified_path.exists(), "unified_dataset.json not created."
        return unified_path

    unified_path = run_stage(1, "Dataset Indexing", stage1)
    if unified_path is None:
        unified_path = OUTPUT_ROOT / "unified_dataset.json"

    # ── Stage 2: SSL Training ──────────────────────────────────────────────
    def stage2():
        from ssl_training import train_ssl
        train_ssl(unified_path, num_epochs=args.ssl_epochs,
                  batch_size=args.batch_size)

    run_stage(2, "SSL Contrastive Training", stage2)

    # ── Stage 3: SSL Embedding Extraction ────────────────────────────────
    def stage3():
        from embedding_and_fusion import extract_ssl_embeddings
        extract_ssl_embeddings()

    run_stage(3, "SSL Embedding Extraction", stage3)

    # ── Stage 4: Fusion Training ──────────────────────────────────────────
    def stage4():
        from embedding_and_fusion import train_fusion
        train_fusion(num_epochs=args.fusion_epochs)

    run_stage(4, "Fusion Transformer Training", stage4)

    # ── Stage 5: Fused Embedding Extraction ──────────────────────────────
    def stage5():
        from embedding_and_fusion import extract_fused_embeddings
        extract_fused_embeddings()

    run_stage(5, "Fused Embedding Extraction", stage5)

    # ── Stage 6: Scene Graph Construction ────────────────────────────────
    def stage6():
        from embedding_and_fusion import build_scene_graphs
        build_scene_graphs()

    run_stage(6, "Scene Graph Construction", stage6)

    # ── Stage 7: GNN Training ────────────────────────────────────────────
    def stage7():
        from gnn_training import train_gnn
        train_gnn(num_epochs=args.gnn_epochs)

    run_stage(7, "GNN Training", stage7)

    # ── Stage 8: GNN Evaluation ──────────────────────────────────────────
    def stage8():
        from gnn_training import eval_gnn
        eval_gnn()

    run_stage(8, "GNN Evaluation", stage8)

    # ── Stage 9: Perception Heads Training ───────────────────────────────
    def stage9():
        from perception_heads_and_export import train_perception_heads
        train_perception_heads(num_epochs=args.head_epochs)

    run_stage(9, "Perception Heads Training", stage9)

    # ── Stage 10: Full Model Assembly & Export ───────────────────────────
    def stage10():
        from perception_heads_and_export import assemble_full_model
        return assemble_full_model()

    model = run_stage(10, "Full Model Assembly + Export", stage10)

    # ── Stage 11: Test Evaluation ─────────────────────────────────────────
    def stage11():
        if model is None:
            from perception_heads_and_export import assemble_full_model, evaluate_on_test
            m = assemble_full_model()
            evaluate_on_test(m, max_per_sensor=200)
        else:
            from perception_heads_and_export import evaluate_on_test
            evaluate_on_test(model, max_per_sensor=200)

    run_stage(11, "Test-Set Evaluation", stage11)

    # ── Stage 12: Results Aggregation ────────────────────────────────────
    def stage12():
        from rl_agent import aggregate_test_results
        aggregate_test_results()

    run_stage(12, "Results Aggregation", stage12)

    # ── Stage 13: RL (optional) ───────────────────────────────────────────
    if not args.skip_rl:
        logger.info("\n[Stage 13] RL Agent — provide a gym env and call "
                    "train_ppo() from rl_agent.py manually.")

    logger.info("\n" + "=" * 70)
    logger.info("  Pipeline complete!")
    logger.info("=" * 70)
    logger.info(f"  Full model   : {OUTPUT_ROOT / 'full_model_inference.pt'}")
    logger.info(f"  Test results : {OUTPUT_ROOT / 'test_results.json'}")
    logger.info(f"  Metrics      : {OUTPUT_ROOT / 'aggregated_metrics.json'}")


if __name__ == "__main__":
    main()
