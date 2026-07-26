import argparse
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def setup_directories(base_output_dir):
    """Creates the necessary folder structure for the splits."""
    splits = ["train", "val", "test"]
    subdirs = ["images", "labels"]

    for split in splits:
        for subdir in subdirs:
            dir_path = Path(base_output_dir) / split / subdir
            dir_path.mkdir(parents=True, exist_ok=True)


def get_paired_files(images_dir, labels_dir):
    """Matches images with their corresponding .npz label files."""
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    image_files = [
        f for f in images_path.iterdir() if f.suffix.lower() in valid_extensions
    ]

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
    print(f"Copying {len(file_pairs)} files to {split_name}...")

    for img_path, label_path in file_pairs:
        # Define destinations
        img_dest = Path(base_output_dir) / split_name / "images" / Path(img_path).name
        label_dest = (
            Path(base_output_dir) / split_name / "labels" / Path(label_path).name
        )

        # Copy files
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
):
    """
    Creates train-val-test splits for image/label pairs and copies them into output_dir structure.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(
            f"Train, val, and test ratios must sum to 1.0. Current sum: {total_ratio}"
        )

    setup_directories(output_dir)
    paired_files = get_paired_files(images_dir, labels_dir)

    if not paired_files:
        print("No paired files found. Please check your input directories.")
        return

    X = [pair[0] for pair in paired_files]
    y = [pair[1] for pair in paired_files]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed
    )

    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_ratio, random_state=seed
    )

    train_pairs = list(zip(X_train, y_train))
    val_pairs = list(zip(X_val, y_val))
    test_pairs = list(zip(X_test, y_test))

    print("\n--- Starting Data Transfer ---")
    copy_files(train_pairs, "train", output_dir)
    copy_files(val_pairs, "val", output_dir)
    copy_files(test_pairs, "test", output_dir)

    print("\n--- Split Complete ---")
    print(f"Total dataset size: {len(paired_files)}")
    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")
    print(f"Test set: {len(test_pairs)} pairs")
    print(f"Output stored in: {os.path.abspath(output_dir)}")


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

    args = parser.parse_args()

    split_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

