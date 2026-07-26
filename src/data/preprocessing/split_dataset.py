import argparse
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def setup_directories(base_output_dir, clean_output=True):
    """Creates the necessary folder structure for the splits, optionally clearing existing split contents."""
    splits = ["train", "val", "test"]
    subdirs = ["images", "labels"]

    base_path = Path(base_output_dir)

    for split in splits:
        split_dir = base_path / split
        if clean_output and split_dir.exists():
            shutil.rmtree(split_dir)
        for subdir in subdirs:
            (split_dir / subdir).mkdir(parents=True, exist_ok=True)


def get_paired_files(images_dir, labels_dir):
    """Matches images with their corresponding .npz label files in deterministic order."""
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Sort files to ensure deterministic splits across OS/filesystems
    image_files = sorted(
        [
            f
            for f in images_path.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
    )

    paired_data = []
    missing_labels = 0

    for img_file in image_files:
        # Assuming label files have the exact same base name but .npz extension
        label_file = labels_path / f"{img_file.stem}.npz"

        if label_file.exists():
            paired_data.append((str(img_file), str(label_file)))
        else:
            missing_labels += 1

    print(f"Found {len(paired_data)} paired image/label files.")
    if missing_labels > 0:
        print(
            f"Warning: {missing_labels} images were missing their corresponding .npz file."
        )

    return paired_data


def copy_files(file_pairs, split_name, base_output_dir):
    """Copies the paired files to their respective split directories."""
    if not file_pairs:
        print(f"No files to copy for {split_name}.")
        return

    for img_path, label_path in tqdm(
        file_pairs, desc=f"Copying files to {split_name}", unit="pair"
    ):
        img_dest = Path(base_output_dir) / split_name / "images" / Path(img_path).name
        label_dest = (
            Path(base_output_dir) / split_name / "labels" / Path(label_path).name
        )

        shutil.copy2(img_path, img_dest)
        shutil.copy2(label_path, label_dest)


def split_dataset(
    images_dir="data/images",
    labels_dir="data/labels",
    output_dir="./dataset",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    clean_output=True,
):
    """
    Creates train-val-test splits for image/label pairs and copies them into output_dir structure.

    Returns:
        dict: Counts of dataset split pairs {"train": int, "val": int, "test": int}.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(
            f"Train, val, and test ratios must sum to 1.0. Current sum: {total_ratio}"
        )

    setup_directories(output_dir, clean_output=clean_output)
    paired_files = get_paired_files(images_dir, labels_dir)

    if not paired_files:
        print("No paired files found. Please check your input directories.")
        return {"train": 0, "val": 0, "test": 0}

    # First split: Separate test set if test_ratio > 0
    if test_ratio > 0:
        temp_pairs, test_pairs = train_test_split(
            paired_files, test_size=test_ratio, random_state=seed
        )
    else:
        temp_pairs, test_pairs = paired_files, []

    # Second split: Separate train and validation sets
    if val_ratio > 0:
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_pairs, val_pairs = train_test_split(
            temp_pairs, test_size=relative_val_ratio, random_state=seed
        )
    else:
        train_pairs, val_pairs = temp_pairs, []

    print("\n--- Starting Data Transfer ---")
    copy_files(train_pairs, "train", output_dir)
    copy_files(val_pairs, "val", output_dir)
    copy_files(test_pairs, "test", output_dir)

    split_counts = {
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
    }

    print("\n--- Split Complete ---")
    print(f"Total dataset size: {len(paired_files)}")
    print(f"Train set: {split_counts['train']} pairs")
    print(f"Validation set: {split_counts['val']} pairs")
    print(f"Test set: {split_counts['test']} pairs")
    print(f"Output stored in: {os.path.abspath(output_dir)}")

    return split_counts


def main():
    parser = argparse.ArgumentParser(
        description="Create train-val-test splits for C2-C4 cephalometric data."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/images",
        help="Path to the directory containing resized 640x640 images.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="data/labels",
        help="Path to the directory containing .npz label files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dataset",
        help="Path to save the split dataset.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Proportion of the dataset for training.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion of the dataset for validation.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Proportion of the dataset for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_false",
        dest="clean_output",
        help="Do not clean target split directories prior to copying.",
    )

    args = parser.parse_args()

    split_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        clean_output=args.clean_output,
    )


if __name__ == "__main__":
    main()


