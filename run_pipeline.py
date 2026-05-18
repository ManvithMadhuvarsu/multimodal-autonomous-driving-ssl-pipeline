"""
=============================================================================
run_pipeline.py — Master Entry Point: Run Complete ML Pipeline
=============================================================================
15 stages, in execution order:

   1. Dataset Indexing                  (dataset_indexing.py)
   2. SSL Contrastive Training          (ssl_training.py)
   3. SSL Embedding Extraction          (embedding_and_fusion.py)
   4. Fusion Transformer Training       (embedding_and_fusion.py)
   5. Fused Embedding Extraction        (embedding_and_fusion.py)
   6. 4D World-Model Pretext  [Tier-B3] (world_model_pretext.py)
   7. Scene Graph Construction          (embedding_and_fusion.py)  *** writes
                                        edges_by_type for HeteroEdgeGNN ***
   8. GNN Training                      (gnn_training.py)
                                        — set GNN_TYPE=hetero to use Tier-B2.
   9. GNN Evaluation                    (gnn_training.py)
  10. Perception Heads Training         (perception_heads_and_export.py)
                                        — DETR3DHead (Tier-B1) with Hungarian
                                          loss + real nuScenes GT; falls back
                                          to skipping samples when GT
                                          unresolvable.
  11. Full Model Assembly + Export      (perception_heads_and_export.py)
  12. Test-Set Evaluation               (perception_heads_and_export.py)
  13. Results Aggregation               (rl_agent.py)
  14. INT8 Quantization     [Tier-B5]   (quantize_int8.py)  — pass
                                        --quantize-int8 to enable.
  15. RL Agent                          (rl_agent.py) — pass --skip-rl to skip.

After this pipeline, run the evaluators in evaluators/ to produce the
UniAD-protocol planning table, SeeingThroughFog adverse-weather table,
official nuScenes-devkit NDS table, and multi-seed mean ± std table.

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-rl
  python run_pipeline.py --skip-world-model
  python run_pipeline.py --quantize-int8
  python run_pipeline.py --stage 7      # resume from stage 7
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
    parser.add_argument("--skip-world-model", action="store_true",
                        help="Skip stage 6 (4D world-model pretext).")
    parser.add_argument("--quantize-int8", action="store_true",
                        help="Run stage 14 — INT8 dynamic quantization "
                             "of the exported FP32 model.")
    parser.add_argument("--stage",        type=int, default=1,
                        help="Start from this stage (1-14)")
    parser.add_argument("--ssl-epochs",   type=int, default=30)
    parser.add_argument("--fusion-epochs",type=int, default=10)
    parser.add_argument("--gnn-epochs",   type=int, default=5)
    parser.add_argument("--head-epochs",  type=int, default=3)
    parser.add_argument("--world-model-epochs", type=int, default=30)
    parser.add_argument("--world-model-past", type=int, default=2)
    parser.add_argument("--world-model-future", type=int, default=3)
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

    # ── Stage 6: 4D World-Model Pretext (Tier-B3) ────────────────────────
    # Trains a small causal-transformer dynamics model on fused embedding
    # sequences; produces a checkpoint that can later be loaded as
    # auxiliary supervision to fine-tune the fusion encoder.
    def stage6():
        if args.skip_world_model:
            logger.info("[Stage 6] World-model pretext skipped (--skip-world-model)")
            return
        try:
            import subprocess, sys as _sys
            fused_json = str(OUTPUT_ROOT / "fused_embeddings.json")
            ckpt_out   = str(OUTPUT_ROOT / "checkpoints" / "fusion" / "world_model.pth")
            cmd = [_sys.executable, "world_model_pretext.py",
                   "--fused-json", fused_json,
                   "--ckpt-out",   ckpt_out,
                   "--past-frames", str(args.world_model_past),
                   "--future-frames", str(args.world_model_future),
                   "--epochs",       str(args.world_model_epochs)]
            logger.info("[Stage 6] " + " ".join(cmd))
            subprocess.run(cmd, check=False)
        except Exception as e:
            logger.warning(f"World-model pretext failed (non-fatal): {e}")

    run_stage(6, "4D World-Model Pretext (Tier-B3)", stage6)

    # ── Stage 7: Scene Graph Construction ────────────────────────────────
    def stage7():
        from embedding_and_fusion import build_scene_graphs
        build_scene_graphs()

    run_stage(7, "Scene Graph Construction", stage7)

    # ── Stage 8: GNN Training ────────────────────────────────────────────
    def stage8():
        from gnn_training import train_gnn
        train_gnn(num_epochs=args.gnn_epochs)

    run_stage(8, "GNN Training", stage8)

    # ── Stage 9: GNN Evaluation ──────────────────────────────────────────
    def stage9():
        from gnn_training import eval_gnn
        eval_gnn()

    run_stage(9, "GNN Evaluation", stage9)

    # ── Stage 10: Perception Heads Training ──────────────────────────────
    def stage10():
        from perception_heads_and_export import train_perception_heads
        train_perception_heads(num_epochs=args.head_epochs)

    run_stage(10, "Perception Heads Training", stage10)

    # ── Stage 11: Full Model Assembly & Export ───────────────────────────
    def stage11():
        from perception_heads_and_export import assemble_full_model
        return assemble_full_model()

    model = run_stage(11, "Full Model Assembly + Export", stage11)

    # ── Stage 12: Test Evaluation ─────────────────────────────────────────
    def stage12():
        if model is None:
            from perception_heads_and_export import assemble_full_model, evaluate_on_test
            m = assemble_full_model()
            evaluate_on_test(m, max_per_sensor=200)
        else:
            from perception_heads_and_export import evaluate_on_test
            evaluate_on_test(model, max_per_sensor=200)

    run_stage(12, "Test-Set Evaluation", stage12)

    # ── Stage 13: Results Aggregation ────────────────────────────────────
    def stage13():
        from rl_agent import aggregate_test_results
        aggregate_test_results()

    run_stage(13, "Results Aggregation", stage13)

    # ── Stage 14: INT8 Quantization (Tier-B5, optional) ──────────────────
    def stage14():
        if not args.quantize_int8:
            logger.info("[Stage 14] INT8 quantization skipped (pass --quantize-int8 to enable)")
            return
        try:
            import subprocess, sys as _sys
            ckpt   = str(OUTPUT_ROOT / "full_model_inference.pt")
            out_p  = str(OUTPUT_ROOT / "full_model_int8_dynamic.pt")
            cmd = [_sys.executable, "quantize_int8.py",
                   "--ckpt", ckpt, "--mode", "dynamic", "--out", out_p, "--bench"]
            logger.info("[Stage 14] " + " ".join(cmd))
            subprocess.run(cmd, check=False)
        except Exception as e:
            logger.warning(f"INT8 quantization failed (non-fatal): {e}")

    run_stage(14, "INT8 Quantization (Tier-B5)", stage14)

    # ── Stage 15: RL (optional) ───────────────────────────────────────────
    if not args.skip_rl:
        logger.info("\n[Stage 15] RL Agent — provide a gym env and call "
                    "train_ppo() from rl_agent.py manually.")

    logger.info("\n" + "=" * 70)
    logger.info("  Pipeline complete!")
    logger.info("=" * 70)
    logger.info(f"  Full model   : {OUTPUT_ROOT / 'full_model_inference.pt'}")
    logger.info(f"  INT8 model   : {OUTPUT_ROOT / 'full_model_int8_dynamic.pt'} "
                f"(if --quantize-int8 used)")
    logger.info(f"  Test results : {OUTPUT_ROOT / 'test_results.json'}")
    logger.info(f"  Metrics      : {OUTPUT_ROOT / 'aggregated_metrics.json'}")
    logger.info("")
    logger.info("  Next steps — run the evaluators against the produced outputs:")
    logger.info("    python -m evaluators.eval_planning_uniad build-gt --nuscenes-root ... --out gt.json")
    logger.info("    python -m evaluators.eval_planning_uniad eval --pred-json test_results.json --gt-json gt.json --out planning.json")
    logger.info("    python -m evaluators.eval_nuscenes_official ...")
    logger.info("    python -m evaluators.run_multi_seed --seeds 42 123 7 ...")


if __name__ == "__main__":
    main()
