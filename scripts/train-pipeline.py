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
from src.utils.test_mlflow import run_mlflow_test
from src.utils.ftp_utils import test_ftp_connection


def main():
    verify_env()

    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Execute full end-to-end data downloading, preprocessing, label generation, splitting, model training, and evaluation pipeline."
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
        default=float(os.getenv("LEARNING_RATE", 5e-5)),
        help="Base learning rate for AdamW-LLRD optimizer (default: 5e-5 or $LEARNING_RATE).",
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
        help="Skip downloading raw export and images from Label Studio.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip preprocessing steps 1-4 and run model training directly.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=640,
        help="Target square image dimension (e.g. 640 for 640x640).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian heatmap sigma parameter.",
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
        "--skip-eval",
        action="store_true",
        help="Skip final model evaluation step after training.",
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
        action="store_true",
        help="Save optimizer state dict along with model weights (useful only for resuming training, adds ~900MB).",
    )

    args = parser.parse_args()

    print("Initializing pipeline execution...\n" + "=" * 40)
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
                print("\n[1/6] Executing: download_export_and_images()")
                download_export_and_images(
                    export_dir="data/raw/exports",
                    img_dir="data/raw/images",
                )
                print("-" * 20)
            else:
                print("\n[1/6] Skipping download as requested.")

            # Step 2: Resize Images & Adjust Coordinates
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

            # Step 3: Generate Heatmaps & GCN Landmark Labels
            print("\n[3/6] Executing: generate_labels()")
            generate_labels(
                json_path="data/exports/export.json",
                images_dir="data/images",
                output_dir="data/labels",
                sigma=args.sigma,
            )
            print("-" * 20)

            # Step 4: Generate Data Splits
            print("\n[4/6] Executing: split_dataset()")
            split_info = split_dataset(
                images_dir="data/images",
                labels_dir="data/labels",
                output_dir="./dataset",
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=42,
            )
            val_cnt = len(split_info.get("val_ids", []))
            test_cnt = len(split_info.get("test_ids", []))
            print(f"--> Split artifacts ready: 'val_image_ids.json' ({val_cnt} IDs), 'test_image_ids.json' ({test_cnt} IDs)")
            print("-" * 20)
        else:
            print("\n[1-4/6] Skipping steps 1 to 4 (--train-only flag set).")

        # Step 5: Model Training
        print(
            f"\n[5/6] Executing: train_main() with {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}"
        )
        train_main(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            llrd_decay_rate=args.llrd_decay_rate,
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
        )
        print("-" * 20)

        # Step 6: Model Evaluation
        if not args.skip_eval:
            print("\n[6/6] Executing: run_evaluation()")
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
            )
            print("-" * 20)
        else:
            print("\n[6/6] Skipping model evaluation step (--skip-eval set).")

        print("\n" + "=" * 40)
        print("Pipeline execution completed successfully.")

    except Exception as e:
        print(f"\n[!] Pipeline failed during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
