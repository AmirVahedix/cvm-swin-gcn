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
        "--no-labels",
        action="store_true",
        help="Disable drawing text landmark labels on visualization PNGs.",
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

    args = parser.parse_args()

    print("Initializing pipeline execution...\n" + "=" * 40)
    print(f"Training configured for {args.epochs} epochs.")

    try:
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
            split_dataset(
                images_dir="data/images",
                labels_dir="data/labels",
                output_dir="./dataset",
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=42,
            )
            print("-" * 20)
        else:
            print("\n[1-4/6] Skipping steps 1 to 4 (--train-only flag set).")

        # Step 5: Model Training
        print(f"\n[5/6] Executing: train_main() with {args.epochs} epochs")
        train_main(epochs=args.epochs)
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
                show_labels=not args.no_labels,
                threshold_px=args.threshold_px,
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
