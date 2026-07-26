import sys
import argparse

# ---------------------------------------------------------
# Pipeline Imports
# ---------------------------------------------------------
from src.data.download_data import download_export_and_images
from src.utils.verify_env import verify_env


def main():
    verify_env()

    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    print("Initializing pipeline execution...\n" + "=" * 40)
    print(f"Training configured for {args.epochs} epochs.")

    try:
        # Step 1: Download Images
        print("\n[1/5] Executing: download_export_and_images()")
        download_export_and_images(
            export_dir="data/raw/exports",
            img_dir="data/raw/images",
        )
        print("-" * 20)

        # # Step 2: Fetch Exports
        # print("\n[2/5] Executing: fetch_project_export()")
        # download_export(["10", "13"], "data/images", "data")
        # print("-" * 20)

        # # Step 3: Process Annotations
        # print("\n[3/5] Executing: generate_masks_from_label_studio_export()")
        # generate_masks_from_label_studio_export(
        #     "./data/export.json",
        #     "./data/images",
        #     "./data/labels",
        # )
        # print("-" * 20)

        # # Step 4: Generate Data Splits
        # print("\n[4/5] Executing: generate_monai_split()")
        # generate_monai_split(
        #     "./data/images",
        #     "./data/labels",
        #     "./data/dataset.json",
        # )
        # print("-" * 20)

        # # Step 5: Model Training
        # print(f"\n[5/5] Executing: train_swin_unetr() with {args.epochs} epochs")
        # train_swin_unetr("./data/dataset.json", epochs=args.epochs)
        # print("-" * 20)

        print("\n" + "=" * 40)
        print("Pipeline execution completed successfully.")

    except Exception as e:
        print(f"\n[!] Pipeline failed during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
