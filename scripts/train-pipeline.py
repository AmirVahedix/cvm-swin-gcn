import sys
import argparse
from pathlib import Path


from src.utils.verify_env import verify_env
from src.data.preprocessing.download_data import download_export_and_images
from src.data.preprocessing.resize_images import process_and_resize_dataset
from src.data.preprocessing.generate_labels import generate_labels
from src.data.preprocessing.split_dataset import split_dataset
from src.train import main as train_main

# Add project root to sys.path if needed
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main():
    verify_env()

    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Execute full end-to-end data downloading, preprocessing, label generation, splitting, and model training pipeline."
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

    args = parser.parse_args()

    print("Initializing pipeline execution...\n" + "=" * 40)
    print(f"Training configured for {args.epochs} epochs.")

    try:
        # Step 1: Download Images & Exports
        if not args.skip_download:
            print("\n[1/5] Executing: download_export_and_images()")
            download_export_and_images(
                export_dir="data/raw/exports",
                img_dir="data/raw/images",
            )
            print("-" * 20)
        else:
            print("\n[1/5] Skipping download as requested.")

        # Step 2: Resize Images & Adjust Coordinates
        print("\n[2/5] Executing: process_and_resize_dataset()")
        # Find latest raw export JSON or fallback to standard export path
        raw_exports_dir = Path("data/raw/exports")
        raw_json_files = sorted(
            raw_exports_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        raw_json_path = (
            str(raw_json_files[0]) if raw_json_files else "data/raw/exports/export.json"
        )

        process_and_resize_dataset(
            img_dir="data/raw/images",
            json_path=raw_json_path,
            out_img_dir="data/images",
            out_json_path="data/exports/export.json",
            target_size=args.target_size,
        )
        print("-" * 20)

        # Step 3: Generate Heatmaps & GCN Landmark Labels
        print("\n[3/5] Executing: generate_labels()")
        generate_labels(
            json_path="data/exports/export.json",
            images_dir="data/images",
            output_dir="data/labels",
            sigma=args.sigma,
        )
        print("-" * 20)

        # Step 4: Generate Data Splits
        print("\n[4/5] Executing: split_dataset()")
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

        # Step 5: Model Training
        print(f"\n[5/5] Executing: train_main() with {args.epochs} epochs")
        # Pass epochs parameter to train_main via sys.argv override for standard execution
        sys.argv = [sys.argv[0], f"--epochs={args.epochs}"]
        train_main()
        print("-" * 20)

        print("\n" + "=" * 40)
        print("Pipeline execution completed successfully.")

    except Exception as e:
        print(f"\n[!] Pipeline failed during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
