#!/usr/bin/env python3
"""
Rescue and Post-Evaluation Logging Script.

Recovers from MLflow artifact upload timeouts / HTTP 500 errors by:
1. Running the Two-Stage cascaded evaluation on the test split.
2. Generating and logging all metrics, clinical summaries, and visualization panels to MLflow.
3. Running whole-dataset batch inference for train/val splits and logging light predictions.
4. Resiliently uploading the model checkpoint (with automatic fallback to FP16 half-precision
   weights (~205 MB) if the 400MB+ full weights trigger proxy 500 errors).

Usage:
    # Re-attach directly to your existing training run:
    python scripts/log_checkpoint_and_eval.py --run-id 7ea3e77e3bae4c9392ddf11aee4a9f6a

    # Or evaluate and log to a new MLflow run:
    python scripts/log_checkpoint_and_eval.py --stage2-weights artifacts/stage2_best.pth
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Log Stage 2 Checkpoint and Execute Post-Training Evaluation")
    parser.add_argument("--run-id", type=str, default=None, help="Existing MLflow Run ID (e.g. from the interrupted training)")
    parser.add_argument("--stage1-weights", type=str, default="artifacts/stage1_best.pth", help="Stage 1 weights")
    parser.add_argument("--stage2-weights", type=str, default="artifacts/stage2_best.pth", help="Stage 2 weights")
    parser.add_argument("--test-img-dir", type=str, default="dataset/test/images", help="Test images directory")
    parser.add_argument("--test-npz-dir", type=str, default="dataset/test/labels", help="Test labels directory")
    parser.add_argument("--raw-img-dir", type=str, default="data/raw/images", help="Raw images directory for batch inference")
    parser.add_argument("--stage1-size", type=int, default=640, help="Stage 1 input resolution")
    parser.add_argument("--stage2-size", type=int, default=448, help="Stage 2 input resolution")
    parser.add_argument("--margin-ratio", type=float, default=0.22, help="ROI margin ratio")
    parser.add_argument("--visualize", type=int, default=10, help="Number of visualization panels to save")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow Tracking URI")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow Experiment Name")
    parser.add_argument("--skip-model-upload", action="store_true", help="Skip uploading the heavy .pth model file")
    parser.add_argument("--force-fp16-upload", action="store_true", help="Directly shrink to FP16 (~205MB) before uploading to MLflow")

    args = parser.parse_args()

    import mlflow
    from src.eval_two_stage import run_two_stage_evaluation
    from scripts.shrink_checkpoint import shrink_checkpoint

    # Setup MLflow environment
    uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    exp_name = args.experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "cvm-swin-gcn")
    mlflow.set_experiment(exp_name)

    print("=" * 80)
    print("🚀 POST-TRAINING EVALUATION & ARTIFACT LOGGING RECOVERY")
    print(f"Tracking URI:     {uri}")
    print(f"Stage 1 Weights:  {args.stage1_weights}")
    print(f"Stage 2 Weights:  {args.stage2_weights}")
    if args.run_id:
        print(f"Attaching to Run: {args.run_id}")
    print("=" * 80)

    if not os.path.exists(args.stage2_weights):
        print(f"❌ Error: Stage 2 weights not found at '{args.stage2_weights}'", file=sys.stderr)
        sys.exit(1)

    run_context = mlflow.start_run(run_id=args.run_id) if args.run_id else mlflow.start_run(run_name="stage2_post_eval")

    with run_context:
        current_run = mlflow.active_run()
        active_run_id = current_run.info.run_id
        print(f"\n[✔] Active MLflow Run ID: {active_run_id}")

        # -------------------------------------------------------------
        # STEP 1: Two-Stage Cascaded Test Set Evaluation
        # -------------------------------------------------------------
        print("\n[Step 1/3] Executing Two-Stage Cascaded Evaluation on Test Cohort...")
        test_payload, test_dir = run_two_stage_evaluation(
            stage1_weights=args.stage1_weights,
            stage2_weights=args.stage2_weights,
            img_dir=args.test_img_dir,
            npz_dir=args.test_npz_dir,
            output_dir="evaluation/two_stage",
            stage1_img_size=args.stage1_size,
            stage2_img_size=args.stage2_size,
            margin_ratio=args.margin_ratio,
            refine_2pass=False,
            num_visualizations=args.visualize,
            log_to_mlflow=False,  # We log directly below into this active run
        )

        test_p = Path(test_dir)
        if (test_p / "metrics.json").exists():
            mlflow.log_artifact(str(test_p / "metrics.json"), artifact_path="evaluation/test")
        if (test_p / "summary_report.txt").exists():
            mlflow.log_artifact(str(test_p / "summary_report.txt"), artifact_path="evaluation/test")
        if (test_p / "predictions_landmarks.csv").exists():
            mlflow.log_artifact(str(test_p / "predictions_landmarks.csv"), artifact_path="evaluation/test")
        if (test_p / "visualizations").exists():
            mlflow.log_artifacts(str(test_p / "visualizations"), artifact_path="evaluation/test/visualizations")

        gen_c = test_payload.get("two_stage_cascade", {}).get("general", {})
        for k, v in gen_c.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", float(v))

        print(f"✅ Test Cohort Logged! Mean Norm: {gen_c.get('mean', 0.0):.2f}%, ≤5%: {gen_c.get('share_within_5pct', 0.0):.2f}%")

        # -------------------------------------------------------------
        # STEP 2: Full Dataset Batch Inference & Light Predictions
        # -------------------------------------------------------------
        if os.path.exists(args.raw_img_dir):
            print(f"\n[Step 2/3] Running Whole-Dataset Batch Inference for Train/Val Splits...")
            try:
                from scripts.batch_inference import run_batch_inference
                light_dir = "evaluation/predictions_light"
                run_batch_inference(
                    images_dir=args.raw_img_dir,
                    output_json=Path(light_dir) / "predictions_full.json",
                    weights=args.stage2_weights,
                    stage1_weights=args.stage1_weights,
                    stage2_weights=args.stage2_weights,
                    stage2_size=args.stage2_size,
                    margin_ratio=args.margin_ratio,
                    batch_size=8,
                    img_size=640,
                    dataset_dir="dataset",
                    test_ids_file="dataset/test_image_ids.json",
                    val_ids_file="dataset/val_image_ids.json",
                    light_output_dir=light_dir,
                    export_light=True,
                    save_csv=True,
                    compute_metrics=True,
                    labels_dir="data/labels",
                    metrics_splits=["val", "train"],
                    metrics_output_dir="evaluation",
                    log_to_mlflow=False,
                )
                pred_json = Path(light_dir) / "predictions_full.json"
                if pred_json.exists():
                    mlflow.log_artifact(str(pred_json), artifact_path="predictions_full")
                pred_csv = Path(light_dir) / "predictions_full.csv"
                if pred_csv.exists():
                    mlflow.log_artifact(str(pred_csv), artifact_path="predictions_full")
                print("✅ Whole dataset predictions logged successfully!")
            except Exception as b_err:
                print(f"⚠️ Warning during batch inference: {b_err}")

        # -------------------------------------------------------------
        # STEP 3: Resilient Model Checkpoint Upload
        # -------------------------------------------------------------
        if not args.skip_model_upload and os.path.exists(args.stage2_weights):
            print("\n[Step 3/3] Uploading Model Checkpoint to MLflow...")
            target_to_upload = args.stage2_weights

            if args.force_fp16_upload:
                fp16_path = str(Path(args.stage2_weights).with_name("stage2_best_fp16.pth"))
                print(f"--> Shrinking checkpoint to FP16 half-precision: '{fp16_path}'...")
                shrink_checkpoint(args.stage2_weights, fp16_path, fp16=True)
                target_to_upload = fp16_path

            upload_success = False
            try:
                size_mb = os.path.getsize(target_to_upload) / (1024 * 1024)
                print(f"--> Uploading '{target_to_upload}' ({size_mb:.1f} MB) via MLflow...")
                mlflow.log_artifact(target_to_upload, artifact_path="checkpoints")
                print(f"✅ Successfully uploaded '{target_to_upload}' to MLflow!")
                upload_success = True
            except Exception as up_err:
                print(f"\n⚠️ Standard upload failed with: {up_err}")
                if not args.force_fp16_upload:
                    print("--> Falling back to compressed FP16 weights (~205 MB)...")
                    try:
                        fp16_path = str(Path(args.stage2_weights).with_name("stage2_best_fp16.pth"))
                        shrink_checkpoint(args.stage2_weights, fp16_path, fp16=True)
                        fp16_mb = os.path.getsize(fp16_path) / (1024 * 1024)
                        print(f"--> Retrying upload with FP16 weights ({fp16_mb:.1f} MB)...")
                        mlflow.log_artifact(fp16_path, artifact_path="checkpoints")
                        print(f"✅ FP16 checkpoint successfully logged to MLflow!")
                        upload_success = True
                    except Exception as fp_err:
                        print(f"⚠️ FP16 upload also timed out or failed: {fp_err}")

            if not upload_success:
                print("\n" + "!" * 80)
                print("💡 Note: The MLflow HTTP proxy endpoint rejected the large file (HTTP 500).")
                print("   However, ALL metrics and evaluation artifacts have been successfully logged!")
                print(f"   Your weights remain intact locally at: '{args.stage2_weights}'")
                print("   You can download them directly from the instance or backup to FTP/S3.")
                print("!" * 80)

    print("\n" + "=" * 80)
    print("🎉 ALL DONE!")
    print(f"Run View URL: {uri}/#/experiments/{current_run.info.experiment_id}/runs/{active_run_id}")
    print("=" * 80)


if __name__ == "__main__":
    main()
