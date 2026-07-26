import sys
import argparse

# ---------------------------------------------------------
# Pipeline Imports
# ---------------------------------------------------------
from src.data.download_images_minio import download_images_from_minio
from src.data.download_exports import download_export
from src.data.generate_masks import generate_masks_from_label_studio_export
from src.data.train_test_split import generate_monai_split
from src.train import train_swin_unetr


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Run the data preparation and training pipeline."
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        help="Number of epochs for training (default: 100)",
    )
    args = parser.parse_args()

    print("Initializing pipeline execution...\n" + "=" * 40)
    print(f"Training configured for {args.epochs} epochs.")

    try:
        # Step 1: Download Images
        print("\n[1/5] Executing: download_images_from_minio()")
        download_images_from_minio(
            "perio-cv-xray-sync",
            "./data/images",
        )
        print("-" * 20)

        # Step 2: Fetch Exports
        print("\n[2/5] Executing: fetch_project_export()")
        download_export(["10", "13"], "data/images", "data")
        print("-" * 20)

        # Step 3: Process Annotations
        print("\n[3/5] Executing: generate_masks_from_label_studio_export()")
        generate_masks_from_label_studio_export(
            "./data/export.json",
            "./data/images",
            "./data/labels",
        )
        print("-" * 20)

        # Step 4: Generate Data Splits
        print("\n[4/5] Executing: generate_monai_split()")
        generate_monai_split(
            "./data/images",
            "./data/labels",
            "./data/dataset.json",
        )
        print("-" * 20)

        # Step 5: Model Training
        print(f"\n[5/5] Executing: train_swin_unetr() with {args.epochs} epochs")
        train_swin_unetr("./data/dataset.json", epochs=args.epochs)
        print("-" * 20)

        print("\n" + "=" * 40)
        print("Pipeline execution completed successfully.")

    except Exception as e:
        print(f"\n[!] Pipeline failed during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
