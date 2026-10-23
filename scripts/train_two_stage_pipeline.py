#!/usr/bin/env python3
"""
End-to-End Two-Stage Coarse-to-Fine Training & Evaluation Pipeline.

Orchestrates:
  Step 1: Train Stage 1 Coarse Model (640x640 full cephalograms) -> artifacts/stage1_best.pth
  Step 2: Train Stage 2 Fine Model (448x448 vertebrae ROIs with jitter) -> artifacts/stage2_best.pth
  Step 3: Run Full Cascaded Evaluation across Validation and Test sets, reporting:
          - Stage 1 standalone performance
          - Stage 2 oracle performance (GT BBox)
          - Two-Stage cascaded pipeline performance
          - Delta improvement table
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval_two_stage import run_two_stage_evaluation
from src.train import main as train_stage1_main
from src.train_stage2 import train_stage2_main


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Two-Stage Coarse-to-Fine Pipeline: Train Stage 1 -> Train Stage 2 -> Cascaded Evaluation"
    )
    # Stage 1 Parameters
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 training and use existing checkpoint")
    parser.add_argument("--stage1-weights", type=str, default="artifacts/stage1_best.pth", help="Stage 1 checkpoint path")
    parser.add_argument("--stage1-epochs", type=int, default=100, help="Epochs for Stage 1 training")
    parser.add_argument("--stage1-size", type=int, default=640, help="Stage 1 input resolution")

    # Stage 2 Parameters
    parser.add_argument("--skip-stage2", action="store_true", help="Skip Stage 2 training and use existing checkpoint")
    parser.add_argument("--stage2-weights", type=str, default="artifacts/stage2_best.pth", help="Stage 2 checkpoint path")
    parser.add_argument("--stage2-epochs", type=int, default=60, help="Epochs for Stage 2 training")
    parser.add_argument("--stage2-size", type=int, default=448, choices=[448, 672], help="Stage 2 ROI resolution (448 or 672)")
    parser.add_argument("--margin-ratio", type=float, default=0.18, help="ROI margin ratio (default: 0.18)")
    parser.add_argument("--sigma", type=float, default=2.5, help="Heatmap Gaussian sigma in Stage 2 ROI space (default: 2.5 for sub-millimeter precision)")
    parser.add_argument("--no-warmstart", action="store_true", help="Train Stage 2 from ImageNet instead of warmstarting from Stage 1")

    # Shared Optimization Parameters
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW-LLRD")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision")
    parser.add_argument(
        "--save-optimizer",
        dest="save_optimizer",
        action="store_true",
        default=True,
        help="Also save full checkpoint with optimizer/scheduler states locally (default: True)",
    )
    parser.add_argument(
        "--no-save-optimizer",
        dest="save_optimizer",
        action="store_false",
        help="Do not save the large full checkpoint locally (only save clean weights)",
    )

    # Dataset Directories
    parser.add_argument("--train-img-dir", type=str, default="dataset/train/images")
    parser.add_argument("--train-npz-dir", type=str, default="dataset/train/labels")
    parser.add_argument("--val-img-dir", type=str, default="dataset/val/images")
    parser.add_argument("--val-npz-dir", type=str, default="dataset/val/labels")
    parser.add_argument("--test-img-dir", type=str, default="dataset/test/images")
    parser.add_argument("--test-npz-dir", type=str, default="dataset/test/labels")
    parser.add_argument("--native-img-dir", type=str, default="data/raw/images")

    # Evaluation
    parser.add_argument("--output-dir", type=str, default="evaluation/two_stage", help="Evaluation output root")
    parser.add_argument("--refine-2pass", action="store_true", help="Enable 2-pass iterative refinement during evaluation")
    parser.add_argument("--visualize", type=int, default=10, help="Number of visualization panels to save")

    args = parser.parse_args()

    pipeline_start = time.time()
    print("\n" + "#" * 88)
    print("      LAUNCHING FULL TWO-STAGE COARSE-TO-FINE PIPELINE")
    print("#" * 88)

    # -------------------------------------------------------------
    # PHASE 1: STAGE 1 TRAINING (Coarse Model on Full Images)
    # -------------------------------------------------------------
    if not args.skip_stage1 and not os.path.exists(args.stage1_weights):
        print("\n" + "=" * 80)
        print("PHASE 1: TRAINING STAGE 1 (COARSE MODEL ON 640x640 RADIOGRAPHS)")
        print("=" * 80)
        train_stage1_main(
            epochs=args.stage1_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            lr=args.lr,
            img_size=args.stage1_size,
            save_path=args.stage1_weights,
            use_amp=not args.no_amp,
            run_name="stage1_coarse_full",
        )
    elif os.path.exists(args.stage1_weights):
        print(f"\n--> Phase 1 Skipped: Found pre-existing Stage 1 checkpoint at '{args.stage1_weights}'")
    else:
        print("\n--> Phase 1 Skipped via --skip-stage1")

    # -------------------------------------------------------------
    # PHASE 2: STAGE 2 TRAINING (Fine Model on Vertebrae ROIs)
    # -------------------------------------------------------------
    if not args.skip_stage2 and not os.path.exists(args.stage2_weights):
        print("\n" + "=" * 80)
        print(f"PHASE 2: TRAINING STAGE 2 (FINE MODEL ON {args.stage2_size}x{args.stage2_size} VERTEBRAE ROIs)")
        print("=" * 80)
        train_stage2_main(
            epochs=args.stage2_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            target_size=args.stage2_size,
            sigma=args.sigma,
            margin_ratio=args.margin_ratio,
            stage1_weights=args.stage1_weights,
            warmstart_from_stage1=not args.no_warmstart,
            save_path=args.stage2_weights,
            train_img_dir=args.train_img_dir,
            train_npz_dir=args.train_npz_dir,
            val_img_dir=args.val_img_dir,
            val_npz_dir=args.val_npz_dir,
            test_img_dir=args.test_img_dir,
            test_npz_dir=args.test_npz_dir,
            eval_visualizations=args.visualize,
            native_img_dir=args.native_img_dir,
            patience=args.patience,
            use_amp=not args.no_amp,
            save_optimizer=args.save_optimizer,
            run_name=f"stage2_fine_roi_{args.stage2_size}",
        )
    elif os.path.exists(args.stage2_weights):
        print(f"\n--> Phase 2 Skipped: Found pre-existing Stage 2 checkpoint at '{args.stage2_weights}'")
    else:
        print("\n--> Phase 2 Skipped via --skip-stage2")

    # -------------------------------------------------------------
    # PHASE 3: COMPREHENSIVE CASCADED EVALUATION (TEST SET)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3: COMPREHENSIVE TWO-STAGE CASCADED EVALUATION (TEST SPLIT)")
    print("=" * 80)
    test_metrics, test_eval_dir = run_two_stage_evaluation(
        stage1_weights=args.stage1_weights,
        stage2_weights=args.stage2_weights,
        img_dir=args.test_img_dir,
        npz_dir=args.test_npz_dir,
        output_dir=args.output_dir,
        run_name="test_split_cascaded_evaluation",
        stage1_img_size=args.stage1_size,
        stage2_img_size=args.stage2_size,
        margin_ratio=args.margin_ratio,
        refine_2pass=args.refine_2pass,
        num_visualizations=args.visualize,
    )

    # -------------------------------------------------------------
    # PHASE 4: VALIDATION SPLIT CASCADED EVALUATION
    # -------------------------------------------------------------
    if os.path.exists(args.val_img_dir) and os.path.exists(args.val_npz_dir):
        print("\n" + "=" * 80)
        print("PHASE 4: VALIDATION SPLIT CASCADED EVALUATION (HYPERPARAMETER SELECTION)")
        print("=" * 80)
        val_metrics, val_eval_dir = run_two_stage_evaluation(
            stage1_weights=args.stage1_weights,
            stage2_weights=args.stage2_weights,
            img_dir=args.val_img_dir,
            npz_dir=args.val_npz_dir,
            output_dir=args.output_dir,
            run_name="val_split_cascaded_evaluation",
            stage1_img_size=args.stage1_size,
            stage2_img_size=args.stage2_size,
            margin_ratio=args.margin_ratio,
            refine_2pass=args.refine_2pass,
            num_visualizations=5,
        )

    total_time_min = (time.time() - pipeline_start) / 60.0
    print("\n" + "#" * 88)
    print(f"🎉 TWO-STAGE PIPELINE COMPLETED SUCCESSFULLY IN {total_time_min:.1f} MINUTES!")
    print(f"   • Test Evaluation Report & Artifacts: {test_eval_dir}")
    print("#" * 88)


if __name__ == "__main__":
    main()
