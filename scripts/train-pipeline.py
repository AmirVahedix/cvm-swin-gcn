import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path if needed
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.utils.verify_env import verify_env
from src.data.preprocessing.download_data import download_export_and_images
from src.data.preprocessing.resize_images import resize_images
from src.data.preprocessing.generate_labels import generate_labels
from src.data.preprocessing.split_dataset import split_dataset
from src.train import main as train_main
from src.eval import run_evaluation
from src.eval_two_stage import run_two_stage_evaluation
from src.train_stage2 import train_stage2_main
from src.utils.test_mlflow import run_mlflow_test
from src.utils.ftp_utils import test_ftp_connection


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Execute full end-to-end data downloading, preprocessing, label generation, splitting, model training, and evaluation pipeline."
    )
    parser.add_argument(
        "--data-source",
        "--source",
        type=str,
        choices=["api", "label-studio"],
        default=os.getenv("DATA_SOURCE", "api"),
        help="Data source to load annotations and images from: 'api' (default: CVM Backend API) or 'label-studio' (legacy).",
    )
    parser.add_argument(
        "--legacy-label-studio",
        action="store_true",
        help="Keep legacy Label Studio as the data source instead of the default CVM Backend API.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("CVM_API_URL", "http://localhost:8000"),
        help="URL for CVM Backend API (default: http://localhost:8000 or $CVM_API_URL).",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=os.getenv("CVM_API_TOKEN", "O0s66Vi.r8e8WBGfY9I3x4neIm"),
        help="Bearer authentication token for CVM Backend API.",
    )
    parser.add_argument(
        "--skip-review-needed",
        dest="skip_review_needed",
        action="store_true",
        default=True,
        help="Skip images marked as review_needed and do not load them in training process (default: True).",
    )
    parser.add_argument(
        "--skip-excluded",
        dest="skip_excluded",
        action="store_true",
        default=True,
        help="Skip images marked as excluded and do not load them in training process (default: True).",
    )
    parser.add_argument(
        "--no-skip-excluded",
        dest="skip_excluded",
        action="store_false",
        help="Do not skip images marked as excluded.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=float(os.getenv("LEARNING_RATE", 1e-4)),
        help="Base learning rate for AdamW-LLRD optimizer (default: 1e-4 or $LEARNING_RATE).",
    )
    parser.add_argument(
        "--llrd-decay-rate",
        type=float,
        default=float(os.getenv("LLRD_DECAY_RATE", 0.8)),
        help="Layer-wise Learning Rate Decay factor (default: 0.8 or $LLRD_DECAY_RATE).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading raw export and images.",
    )
    parser.add_argument(
        "--skip-resize",
        action="store_true",
        help="Skip image resizing step.",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip landmark and heatmap label generation step.",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip dataset split step.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip preprocessing steps 1-4 and run model training directly.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip all downloading, preprocessing, and training steps, executing only Step 6 (Evaluation) and Step 7 (Batch Inference).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=640,
        help="Target square image dimension (default: 640).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Gaussian heatmap sigma parameter (default: 4.0).",
    )
    parser.add_argument(
        "--eval-weights",
        type=str,
        default="./artifacts/best.pth",
        help="Model checkpoint path to evaluate.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=8,
        help="Number of test image visualizations to save as PNG during evaluation.",
    )

    parser.add_argument(
        "--threshold-px",
        type=float,
        default=2.5,
        help="Radial error tolerance threshold in pixels for evaluation detection metrics.",
    )
    parser.add_argument(
        "--use-tta",
        "--tta",
        action="store_true",
        help="Enable Test-Time Augmentation (multi-scale inference) during evaluation.",
    )
    parser.add_argument(
        "--tta-scales",
        type=str,
        default="0.95,1.0,1.05",
        help="Comma-separated scale factors for TTA (default: '0.95,1.0,1.05').",
    )
    parser.add_argument(
        "--tta-shifts",
        type=str,
        default=None,
        help="Optional comma-separated pixel shifts in format 'dx:dy,dx:dy' (e.g. '-4:-4,4:4').",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip final model evaluation step after training.",
    )
    # Two-Stage Coarse-to-Fine Options (Enabled by default on two-stage branch)
    parser.add_argument(
        "--two-stage",
        dest="two_stage",
        action="store_true",
        default=True,
        help="Enable full Two-Stage Coarse-to-Fine training (default: True).",
    )
    parser.add_argument(
        "--single-stage",
        "--no-two-stage",
        dest="two_stage",
        action="store_false",
        help="Disable two-stage training and run single-stage baseline.",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=60,
        help="Number of epochs for Stage 2 Fine ROI training (default: 60).",
    )
    parser.add_argument(
        "--stage2-size",
        type=int,
        default=448,
        choices=[448, 672],
        help="Stage 2 ROI resolution (default: 448).",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.18,
        help="ROI margin ratio (default: 0.18).",
    )
    parser.add_argument(
        "--refine-2pass",
        action="store_true",
        help="Enable 2-pass iterative refinement during evaluation.",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 coarse training and use existing checkpoint.",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip Stage 2 fine training and use existing checkpoint.",
    )
    parser.add_argument(
        "--stage1-weights",
        type=str,
        default="artifacts/stage1_best.pth",
        help="Stage 1 checkpoint path (default: artifacts/stage1_best.pth).",
    )
    parser.add_argument(
        "--stage1-run-id",
        type=str,
        default=None,
        help="MLflow Run ID to automatically download Stage 1 checkpoint if not present locally.",
    )
    parser.add_argument(
        "--stage2-weights",
        type=str,
        default="artifacts/stage2_best.pth",
        help="Stage 2 checkpoint path (default: artifacts/stage2_best.pth).",
    )
    parser.add_argument(
        "--stage2-run-id",
        type=str,
        default=None,
        help="MLflow Run ID to automatically download Stage 2 checkpoint if not present locally.",
    )
    parser.add_argument(
        "--test-mlflow",
        action="store_true",
        help="Run MLflow connection test (upload test artifact & fake metrics) before pipeline execution.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (e.g., http://localhost:5000 or http://141.11.107.165:5000).",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="MLflow run name.",
    )
    parser.add_argument(
        "--mlflow-username",
        type=str,
        default=None,
        help="MLflow tracking username.",
    )
    parser.add_argument(
        "--mlflow-password",
        type=str,
        default=None,
        help="MLflow tracking password.",
    )
    parser.add_argument(
        "--test-ftp",
        action="store_true",
        help="Run FTP connection test before pipeline execution.",
    )
    parser.add_argument(
        "--ftp-host",
        type=str,
        default=None,
        help="FTP host (e.g., ftp.example.com or IP).",
    )
    parser.add_argument(
        "--ftp-port",
        type=str,
        default=None,
        help="FTP port (default: 21).",
    )
    parser.add_argument(
        "--ftp-user",
        "--ftp-username",
        type=str,
        default=None,
        help="FTP username.",
    )
    parser.add_argument(
        "--ftp-password",
        "--ftp-pass",
        type=str,
        default=None,
        help="FTP password.",
    )
    parser.add_argument(
        "--ftp-remote-dir",
        "--ftp-dir",
        type=str,
        default=None,
        help="Remote directory path on FTP server.",
    )
    parser.add_argument(
        "--ftp-tls",
        action="store_true",
        help="Use FTPS / TLS encryption for FTP connection.",
    )
    parser.add_argument(
        "--skip-ftp",
        action="store_true",
        help="Skip uploading final model and metrics JSON to FTP server.",
    )
    parser.add_argument(
        "--save-optimizer",
        dest="save_optimizer",
        action="store_true",
        default=True,
        help="Also save full checkpoint with optimizer state locally on VPS (not sent to MLflow, default: True).",
    )
    parser.add_argument(
        "--no-save-optimizer",
        dest="save_optimizer",
        action="store_false",
        help="Do not save full checkpoint with optimizer state locally.",
    )
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (AMP FP16) training (default: True).",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable Automatic Mixed Precision and train in full FP32.",
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        default=bool(int(os.getenv("TORCH_COMPILE", "0"))),
        help="Enable PyTorch 2.0+ model compilation via torch.compile() (default: False or $TORCH_COMPILE).",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile_model",
        action="store_false",
        help="Disable PyTorch model compilation and force eager mode.",
    )
    parser.add_argument(
        "--warmup-coord-epochs",
        type=int,
        default=5,
        help="Number of initial epochs to linearly ramp coordinate/graph losses from 0.0 to full weight (default: 5).",
    )
    parser.add_argument(
        "--lambda-hm",
        type=float,
        default=float(os.getenv("LAMBDA_HM", 1.0)),
        help="Weight for Adaptive Wing Loss on heatmaps (default: 1.0 or $LAMBDA_HM).",
    )
    parser.add_argument(
        "--lambda-cd",
        type=float,
        default=float(os.getenv("LAMBDA_CD", 5.0)),
        help="Weight for Wing Loss on coordinates (default: 5.0 or $LAMBDA_CD).",
    )
    parser.add_argument(
        "--lambda-graph",
        type=float,
        default=float(os.getenv("LAMBDA_GRAPH", 1.0)),
        help="Weight for Anatomical Graph Loss on landmark edges (default: 1.0 or $LAMBDA_GRAPH).",
    )
    parser.add_argument(
        "--lambda-morph",
        type=float,
        default=float(os.getenv("LAMBDA_MORPH", 0.0)),
        help="Weight for Morphometric Shape Loss on Shape Index and Taper Ratio (default: 0.0 or $LAMBDA_MORPH).",
    )
    parser.add_argument(
        "--test-ids-file",
        type=str,
        default=None,
        help="Path to JSON file containing exact test image IDs to isolate for the test split (default: check dataset/test_image_ids.json).",
    )
    parser.add_argument(
        "--val-ids-file",
        type=str,
        default=None,
        help="Path to JSON file containing exact validation image IDs to isolate for the val split (default: check dataset/val_image_ids.json).",
    )
    parser.add_argument(
        "--use-test-cohort",
        dest="use_test_cohort",
        action="store_true",
        default=True,
        help="Use images marked as test cohort in API/export strictly as the evaluation test split (default: True).",
    )
    parser.add_argument(
        "--no-test-cohort",
        dest="use_test_cohort",
        action="store_false",
        help="Disable automatic test cohort isolation for the test split.",
    )
    parser.add_argument(
        "--use-fixed-cohorts",
        dest="use_fixed_cohorts",
        action="store_true",
        default=True,
        help="Use fixed cohorts directly from CVM Backend API/export (train, val, test) with zero random splitting (default: True).",
    )
    parser.add_argument(
        "--no-fixed-cohorts",
        dest="use_fixed_cohorts",
        action="store_false",
        help="Disable fixed API cohorts and perform ratio-based splitting.",
    )
    parser.add_argument(
        "--balanced-sampling",
        dest="use_balanced_sampling",
        action="store_true",
        default=True,
        help="Enable class-weighted balanced sampling on finalStage (1-6) to prevent gradient bias (default: True).",
    )
    parser.add_argument(
        "--no-balanced-sampling",
        dest="use_balanced_sampling",
        action="store_false",
        help="Disable class-weighted balanced sampling and use standard uniform shuffling.",
    )
    parser.add_argument(
        "--balanced-sampling-power",
        type=float,
        default=float(os.getenv("BALANCED_SAMPLING_POWER", "0.5")),
        help="Smoothing exponent for class-weighted balanced sampling (1.0 = strict 1/N, 0.5 = smoothed 1/sqrt(N), 0.0 = uniform, default: 0.5).",
    )
    parser.add_argument(
        "--max-workers",
        "-j",
        type=int,
        default=int(os.getenv("MAX_WORKERS", "10")),
        help="Number of concurrent worker threads for downloading images (default: 10 or $MAX_WORKERS).",
    )
    parser.add_argument(
        "--batch-inference",
        dest="run_batch_inference_step",
        action="store_true",
        default=True,
        help="Run batch inference on the whole dataset and generate lightweight predicted landmark location files when training completes (default: True).",
    )
    parser.add_argument(
        "--no-batch-inference",
        dest="run_batch_inference_step",
        action="store_false",
        help="Disable automatic batch inference on the whole dataset after training.",
    )
    parser.add_argument(
        "--whole-data-images-dir",
        type=str,
        default="data/images",
        help="Directory containing the whole un-split dataset images for batch inference (default: data/images).",
    )
    parser.add_argument(
        "--light-predictions-dir",
        type=str,
        default="evaluation/predictions_light",
        help="Output directory to store lightweight predicted landmark location files (default: evaluation/predictions_light).",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=8,
        help="Batch size for batch inference (default: 8).",
    )

    args = parser.parse_args()
    if args.legacy_label_studio:
        args.data_source = "label-studio"

    if args.eval_only:
        test_ready = (
            os.path.isdir("dataset/test/images")
            and len(os.listdir("dataset/test/images")) > 0
            and os.path.isdir("dataset/test/labels")
            and len(os.listdir("dataset/test/labels")) > 0
        )
        if not test_ready:
            print("⚠️ Test dataset ('dataset/test') not found locally. Executing data download & prep (Steps 1-4) first...")
            args.skip_download = False
            args.skip_resize = False
            args.skip_labels = False
            args.skip_split = False
            args.skip_stage1 = True
            args.skip_stage2 = True
            args.train_only = False
        else:
            print("✅ Test dataset ('dataset/test') found locally. Skipping Steps 1-5 and proceeding directly to Evaluation & Batch Inference.")
            args.train_only = True
            args.skip_stage1 = True
            args.skip_stage2 = True

    verify_env(data_source=args.data_source)

    print("Initializing pipeline execution...\n" + "=" * 40)
    print(f"Data source configured: {args.data_source.upper()}")
    print(f"Training configured for {args.epochs} epochs.")

    try:
        # Pre-flight MLflow Connection Test
        if args.test_mlflow:
            print("\n[MLflow Pre-Flight] Running MLflow connection test...")
            success = run_mlflow_test(
                tracking_uri=args.mlflow_tracking_uri,
                experiment_name=args.mlflow_experiment_name,
                run_name=f"preflight-{args.mlflow_run_name}"
                if args.mlflow_run_name
                else None,
                tracking_username=args.mlflow_username,
                tracking_password=args.mlflow_password,
            )
            if not success:
                print(
                    "❌ MLflow connection test failed. Aborting pipeline.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("✅ MLflow connection test passed.\n" + "-" * 20)

        # Pre-flight FTP Connection Test
        if args.test_ftp:
            print("\n[FTP Pre-Flight] Running FTP connection test...")
            ftp_ok = test_ftp_connection(
                ftp_host=args.ftp_host,
                ftp_port=args.ftp_port,
                ftp_user=args.ftp_user,
                ftp_password=args.ftp_password,
                remote_dir=args.ftp_remote_dir,
                use_tls=args.ftp_tls if args.ftp_tls else None,
            )
            if not ftp_ok:
                print(
                    "❌ FTP connection test failed. Aborting pipeline.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("✅ FTP connection test passed.\n" + "-" * 20)

        if not args.train_only:
            # Step 1: Download Images & Exports
            if not args.skip_download:
                print(f"\n[1/6] Executing: download_export_and_images(data_source='{args.data_source}')")
                download_export_and_images(
                    export_dir="data/raw/exports",
                    img_dir="data/raw/images",
                    data_source=args.data_source,
                    api_url=args.api_url,
                    api_token=args.api_token,
                    skip_review_needed=args.skip_review_needed,
                    skip_excluded=args.skip_excluded,
                    max_workers=args.max_workers,
                )
                print("-" * 20)
            else:
                print("\n[1/6] Skipping download as requested.")

            # Step 2: Resize Images & Adjust Coordinates
            if not args.skip_resize:
                print("\n[2/6] Executing: resize_images()")
                raw_exports_dir = Path("data/raw/exports")
                raw_json_files = sorted(
                    raw_exports_dir.glob("*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                raw_json_path = (
                    str(raw_json_files[0])
                    if raw_json_files
                    else "data/raw/exports/export.json"
                )

                resize_images(
                    img_dir="data/raw/images",
                    json_path=raw_json_path,
                    out_img_dir="data/images",
                    out_json_path="data/exports/export.json",
                    target_size=args.target_size,
                )
                print("-" * 20)
            else:
                print("\n[2/6] Skipping image resize (--skip-resize set).")

            # Step 3: Generate Heatmaps & GCN Landmark Labels
            if not args.skip_labels:
                print("\n[3/6] Executing: generate_labels()")
                generate_labels(
                    json_path="data/exports/export.json",
                    images_dir="data/images",
                    output_dir="data/labels",
                    sigma=args.sigma,
                )
                print("-" * 20)
            else:
                print("\n[3/6] Skipping label generation (--skip-labels set).")

            # Step 4: Generate Data Splits
            if not args.skip_split:
                print("\n[4/6] Executing: split_dataset()")
                split_info = split_dataset(
                    images_dir="data/images",
                    labels_dir="data/labels",
                    output_dir="./dataset",
                    train_ratio=0.70,
                    val_ratio=0.15,
                    test_ratio=0.15,
                    seed=42,
                    fixed_test_ids_path=args.test_ids_file,
                    fixed_val_ids_path=args.val_ids_file,
                    use_test_cohort=args.use_test_cohort,
                    use_fixed_cohorts=args.use_fixed_cohorts,
                )
                train_cnt = split_info.get("train", 0)
                val_cnt = len(split_info.get("val_ids", []))
                test_cnt = len(split_info.get("test_ids", []))
                print(f"--> Split artifacts ready: 'train' ({train_cnt} IDs), 'val_image_ids.json' ({val_cnt} IDs), 'test_image_ids.json' ({test_cnt} IDs)")
                print("-" * 20)
            else:
                print("\n[4/6] Skipping dataset splitting (--skip-split set).")
        else:
            print("\n[1-4/6] Skipping steps 1 to 4 (--train-only flag set).")

        # Step 5: Model Training
        if args.two_stage:
            print("\n" + "=" * 50)
            print("🌟 TWO-STAGE COARSE-TO-FINE PIPELINE SELECTED")
            print("=" * 50)

            # Step 5a: Stage 1 Training
            if not args.skip_stage1 and not os.path.exists(args.stage1_weights):
                print(f"\n[5a/6] Executing Stage 1 Coarse Model Training ({args.epochs} epochs)...")
                train_main(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    llrd_decay_rate=args.llrd_decay_rate,
                    lambda_hm=args.lambda_hm,
                    lambda_cd=args.lambda_cd,
                    lambda_graph=args.lambda_graph,
                    lambda_morph=args.lambda_morph,
                    img_size=args.target_size,
                    experiment_name=args.mlflow_experiment_name,
                    tracking_uri=args.mlflow_tracking_uri,
                    run_name=f"{args.mlflow_run_name}_stage1" if args.mlflow_run_name else "stage1_coarse",
                    tracking_username=args.mlflow_username,
                    tracking_password=args.mlflow_password,
                    skip_eval=True,
                    skip_ftp=True,
                    save_optimizer=args.save_optimizer,
                    use_amp=args.use_amp,
                    compile_model=args.compile_model,
                    warmup_coord_epochs=args.warmup_coord_epochs,
                    use_balanced_sampling=args.use_balanced_sampling,
                    balanced_sampling_power=args.balanced_sampling_power,
                )
                if os.path.exists("artifacts/best.pth"):
                    import shutil
                    shutil.copy("artifacts/best.pth", args.stage1_weights)
                    print(f"--> Copied 'artifacts/best.pth' to '{args.stage1_weights}'")
            elif os.path.exists(args.stage1_weights):
                print(f"\n--> Stage 1 Skipped: Found existing checkpoint at '{args.stage1_weights}'")
            else:
                print("\n--> Stage 1 Skipped via --skip-stage1")

            # Check if Stage 1 checkpoint is needed for Stage 2 warmstart and fetch from MLflow if missing
            run_id_for_stage1 = getattr(args, "stage1_run_id", None) or os.getenv("STAGE1_RUN_ID")
            if not os.path.exists(args.stage1_weights) and run_id_for_stage1:
                print(f"\n--> Stage 1 weights not found locally at '{args.stage1_weights}'. Automatically downloading from MLflow run '{run_id_for_stage1}'...")
                from scripts.download_checkpoint import download_checkpoint_from_mlflow
                download_checkpoint_from_mlflow(
                    run_id=run_id_for_stage1,
                    dest_path=args.stage1_weights,
                    tracking_uri=args.mlflow_tracking_uri,
                    username=args.mlflow_username,
                    password=args.mlflow_password,
                )

            # Step 5b: Stage 2 Training
            if not args.skip_stage2 and not os.path.exists(args.stage2_weights):
                print(f"\n[5b/6] Executing Stage 2 Fine Model Training ({args.stage2_epochs} epochs, {args.stage2_size}x{args.stage2_size})...")
                train_stage2_main(
                    epochs=args.stage2_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    llrd_decay_rate=args.llrd_decay_rate,
                    target_size=args.stage2_size,
                    sigma=args.sigma,
                    margin_ratio=args.margin_ratio,
                    stage1_weights=args.stage1_weights if os.path.exists(args.stage1_weights) else "artifacts/best.pth",
                    warmstart_from_stage1=True,
                    save_path=args.stage2_weights,
                    train_img_dir="dataset/train/images",
                    train_npz_dir="dataset/train/labels",
                    val_img_dir="dataset/val/images",
                    val_npz_dir="dataset/val/labels",
                    native_img_dir="data/raw/images" if os.path.exists("data/raw/images") else None,
                    export_json_path="data/exports/export.json" if os.path.exists("data/exports/export.json") else None,
                    experiment_name=args.mlflow_experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "cvm-swin-gcn"),
                    tracking_uri=args.mlflow_tracking_uri,
                    tracking_username=args.mlflow_username,
                    tracking_password=args.mlflow_password,
                    run_name=f"{args.mlflow_run_name}_stage2" if args.mlflow_run_name else "stage2_fine",
                    use_amp=args.use_amp,
                    use_balanced_sampling=args.use_balanced_sampling,
                    balanced_sampling_power=args.balanced_sampling_power,
                    lambda_morph=args.lambda_morph,
                    save_optimizer=args.save_optimizer,
                )
            elif os.path.exists(args.stage2_weights):
                print(f"\n--> Stage 2 Skipped: Found existing checkpoint at '{args.stage2_weights}'")
            else:
                print("\n--> Stage 2 Skipped via --skip-stage2")

            # Check if Stage 2 checkpoint is needed and fetch from MLflow if missing
            run_id_for_stage2 = getattr(args, "stage2_run_id", None) or os.getenv("STAGE2_RUN_ID")
            if not os.path.exists(args.stage2_weights) and run_id_for_stage2:
                print(f"\n--> Stage 2 weights not found locally at '{args.stage2_weights}'. Automatically downloading from MLflow run '{run_id_for_stage2}'...")
                from scripts.download_checkpoint import download_checkpoint_from_mlflow
                download_checkpoint_from_mlflow(
                    run_id=run_id_for_stage2,
                    artifact_path="stage2_best.pth",
                    dest_path=args.stage2_weights,
                    tracking_uri=args.mlflow_tracking_uri,
                    username=args.mlflow_username,
                    password=args.mlflow_password,
                )

            # Step 6: Two-Stage Cascaded Clinical Evaluation
            if not args.skip_eval:
                print("\n[6/6] Executing Two-Stage Cascaded Clinical Evaluation on Test Split...")
                s1_w = (
                    args.stage1_weights
                    if (args.stage1_weights and os.path.exists(args.stage1_weights))
                    else ("artifacts/stage1_best.pth" if os.path.exists("artifacts/stage1_best.pth") else "artifacts/best.pth")
                )
                s2_w = (
                    args.stage2_weights
                    if (args.stage2_weights and os.path.exists(args.stage2_weights))
                    else ("artifacts/stage2_best.pth" if os.path.exists("artifacts/stage2_best.pth") else None)
                )
                run_two_stage_evaluation(
                    stage1_weights=s1_w,
                    stage2_weights=s2_w or "artifacts/stage2_best.pth",
                    img_dir="dataset/test/images",
                    npz_dir="dataset/test/labels",
                    output_dir="evaluation/two_stage",
                    run_name=args.mlflow_run_name or "test_split_cascade_eval",
                    stage1_img_size=args.target_size,
                    stage2_img_size=args.stage2_size,
                    margin_ratio=args.margin_ratio,
                    refine_2pass=args.refine_2pass,
                    num_visualizations=args.eval_samples,
                    log_to_mlflow=True,
                    tracking_uri=args.mlflow_tracking_uri,
                    tracking_username=args.mlflow_username,
                    tracking_password=args.mlflow_password,
                    experiment_name=args.mlflow_experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "cvm-swin-gcn"),
                )

                if not args.skip_ftp and args.ftp_host and args.ftp_user and args.ftp_password:
                    from src.utils.ftp_utils import backup_artifacts_to_ftp
                    backup_files = [f for f in [s1_w, s2_w] if os.path.exists(f)]
                    if backup_files:
                        print(f"--> Uploading Two-Stage checkpoints {backup_files} to FTP server...")
                        backup_artifacts_to_ftp(
                            host=args.ftp_host,
                            port=args.ftp_port,
                            user=args.ftp_user,
                            password=args.ftp_password,
                            remote_dir=args.ftp_remote_dir,
                            files=backup_files,
                            use_tls=args.ftp_tls if args.ftp_tls else None,
                        )
        else:
            print(
                f"\n[5/6] Executing: train_main() with {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}"
            )
            train_main(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                llrd_decay_rate=args.llrd_decay_rate,
                lambda_hm=args.lambda_hm,
                lambda_cd=args.lambda_cd,
                lambda_graph=args.lambda_graph,
                lambda_morph=args.lambda_morph,
                img_size=args.target_size,
                experiment_name=args.mlflow_experiment_name,
                tracking_uri=args.mlflow_tracking_uri,
                run_name=args.mlflow_run_name,
                tracking_username=args.mlflow_username,
                tracking_password=args.mlflow_password,
                skip_eval=args.skip_eval,
                ftp_host=args.ftp_host,
                ftp_port=args.ftp_port,
                ftp_user=args.ftp_user,
                ftp_password=args.ftp_password,
                ftp_remote_dir=args.ftp_remote_dir,
                ftp_tls=args.ftp_tls if args.ftp_tls else None,
                skip_ftp=args.skip_ftp,
                save_optimizer=args.save_optimizer,
                use_amp=args.use_amp,
                compile_model=args.compile_model,
                warmup_coord_epochs=args.warmup_coord_epochs,
                use_balanced_sampling=args.use_balanced_sampling,
                balanced_sampling_power=args.balanced_sampling_power,
            )
            print("-" * 20)

            # Step 6: Standalone Model Evaluation
            # Note: train_main() already runs evaluation and logs all metrics/artifacts to the training MLflow run.
            # Step 6 will only execute if custom eval-weights were provided.
            if not args.skip_eval and args.eval_weights != "./artifacts/best.pth":
                print(f"\n[6/6] Executing standalone evaluation for custom weights: {args.eval_weights}")
                parsed_eval_tta_scales = (
                    [float(s.strip()) for s in args.tta_scales.split(",") if s.strip()]
                    if args.tta_scales
                    else [0.95, 1.0, 1.05]
                )
                parsed_eval_tta_shifts = []
                if args.tta_shifts:
                    for pair in args.tta_shifts.split(","):
                        if ":" in pair:
                            dx, dy = pair.strip().split(":")
                            parsed_eval_tta_shifts.append((int(dx), int(dy)))

                run_evaluation(
                    weights_path=args.eval_weights,
                    test_img_dir="dataset/test/images",
                    test_npz_dir="dataset/test/labels",
                    output_dir="evaluation",
                    img_size=args.target_size,
                    num_samples=args.eval_samples,
                    threshold_px=args.threshold_px,
                    tracking_uri=args.mlflow_tracking_uri,
                    experiment_name=args.mlflow_experiment_name,
                    tracking_username=args.mlflow_username,
                    tracking_password=args.mlflow_password,
                    use_tta=args.use_tta,
                    tta_scales=parsed_eval_tta_scales,
                    tta_shifts=parsed_eval_tta_shifts if len(parsed_eval_tta_shifts) > 0 else None,
                )
                print("-" * 20)
            elif not args.skip_eval:
                print("\n[6/6] Clinical test evaluation completed inside Step 5 (logged directly into the MLflow training run).")
                print("-" * 20)
            else:
                print("\n[6/6] Skipping model evaluation step (--skip-eval set).")

        # Step 7: Batch Inference on Whole Dataset & Lightweight Landmark Location Generation
        if args.run_batch_inference_step:
            print(
                f"\n[7/7] Executing: Batch Inference on Whole Data & Generating Lightweight Landmark Location Files..."
            )
            s1_w = None
            for cand in [args.stage1_weights, "artifacts/stage1_best.pth", "artifacts/best.pth", args.eval_weights]:
                if cand and os.path.exists(cand):
                    s1_w = cand
                    break

            s2_w = None
            for cand in [args.stage2_weights, "artifacts/stage2_best.pth"]:
                if cand and os.path.exists(cand):
                    s2_w = cand
                    break

            best_weights_path = s2_w or s1_w or (args.eval_weights if (args.eval_weights and os.path.exists(args.eval_weights)) else "artifacts/best.pth")

            if not os.path.exists(best_weights_path) and not (s1_w and os.path.exists(s1_w)):
                print(
                    f"⚠️ Warning: Checkpoint weights '{best_weights_path}' not found. Skipping Step 7."
                )
            elif not os.path.isdir(args.whole_data_images_dir):
                print(
                    f"⚠️ Warning: Whole data images directory '{args.whole_data_images_dir}' not found. Skipping Step 7."
                )
            else:
                from scripts.batch_inference import run_batch_inference

                run_batch_inference(
                    images_dir=args.whole_data_images_dir,
                    output_json=Path(args.light_predictions_dir) / "predictions_full.json",
                    weights=s1_w or best_weights_path,
                    stage1_weights=s1_w,
                    stage2_weights=s2_w if args.two_stage else None,
                    stage2_size=args.stage2_size,
                    margin_ratio=args.margin_ratio,
                    device="auto",
                    batch_size=args.inference_batch_size,
                    img_size=args.target_size,
                    dataset_dir="dataset",
                    test_ids_file="dataset/test_image_ids.json",
                    val_ids_file="dataset/val_image_ids.json",
                    export_json="data/exports/export.json",
                    light_output_dir=args.light_predictions_dir,
                    export_light=True,
                    save_csv=True,
                    compute_metrics=True,
                    labels_dir="data/labels",
                    metrics_splits=["val", "train", "test"],
                    metrics_output_dir="evaluation",
                    log_to_mlflow=True,
                )
                print(
                    f"--> Step 7 completed: Lightweight landmark files generated in '{args.light_predictions_dir}'"
                    f" and full metrics files ('metrics_val.json', 'metrics_train.json') saved to 'evaluation/'."
                )
                print("-" * 20)
        else:
            print("\n[7/7] Skipping batch inference on whole data (--no-batch-inference set).")

        print("\n" + "=" * 40)
        print("Pipeline execution completed successfully.")

    except Exception as e:
        import traceback
        print(f"\n[!] Pipeline failed during execution: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
