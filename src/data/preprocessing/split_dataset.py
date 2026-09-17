import argparse
import os
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs
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


def load_label_studio_mapping(export_json_path=None) -> dict[str, int]:
    """
    Loads Label Studio JSON export and returns a mapping from image filename/stem to Label Studio task ID.
    """
    mapping = {}
    candidate_paths = []
    if export_json_path:
        candidate_paths.append(Path(export_json_path))
    candidate_paths.append(Path("data/exports/export.json"))
    raw_exports_dir = Path("data/raw/exports")
    if raw_exports_dir.exists():
        raw_files = sorted(
            raw_exports_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidate_paths.extend(raw_files)

    found_path = None
    for p in candidate_paths:
        if p.exists() and p.is_file():
            found_path = p
            break

    if not found_path:
        return mapping

    try:
        with open(found_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        for task in tasks:
            task_id = task.get("id")
            if task_id is None:
                continue
            raw_path = task.get("file_upload") or task.get("data", {}).get("img")
            if not raw_path:
                continue
            parsed_url = urlparse(raw_path)
            query_params = parse_qs(parsed_url.query)
            if "d" in query_params:
                filename = os.path.basename(query_params["d"][0])
            else:
                filename = os.path.basename(parsed_url.path)

            if "-" in filename and len(filename.split("-")[0]) == 8:
                clean_filename = "-".join(filename.split("-")[1:])
            else:
                clean_filename = filename

            stem = Path(clean_filename).stem
            mapping[clean_filename] = task_id
            mapping[stem] = task_id
        print(f"--> Loaded Label Studio task ID mapping for {len(mapping)//2} images from: {found_path}")
    except Exception as e:
        print(f"⚠️ Warning loading Label Studio mapping from {found_path}: {e}")

    return mapping


def load_fixed_id_set(file_path) -> set:
    """
    Loads fixed IDs from a JSON file. Supports:
    1. Flat list: [1, 2, 3] or ["1", "2"]
    2. List of dicts/objects: [{"id": 1, ...}], [{"task_id": 1, ...}], [{"image_id": 1, ...}]
    """
    if not file_path:
        return set()
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return set()

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"⚠️ Warning: {file_path} does not contain a JSON array/list.")
            return set()

        id_set = set()
        for item in data:
            if isinstance(item, dict):
                val = item.get("id")
                if val is None:
                    val = item.get("task_id")
                if val is None:
                    val = item.get("image_id")
            else:
                val = item
            if val is not None:
                id_set.add(int(val) if str(val).isdigit() else str(val))
        return id_set
    except Exception as e:
        print(f"⚠️ Warning reading fixed ID file {file_path}: {e}")
        return set()


def split_dataset(
    images_dir="data/images",
    labels_dir="data/labels",
    output_dir="./dataset",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    clean_output=True,
    export_json_path="data/exports/export.json",
    fixed_test_ids_path=None,
    fixed_val_ids_path=None,
):
    """
    Creates train-val-test splits for image/label pairs and copies them into output_dir structure.
    If fixed_test_ids_path is specified (or test_image_ids.json exists in output_dir),
    the test set will strictly contain those exact IDs.
    If fixed_val_ids_path is specified (or val_image_ids.json exists in output_dir),
    the validation set will strictly contain those exact IDs, and all remaining data
    will be allocated to the training set.

    Returns:
        dict: Counts of dataset split pairs {"train": int, "val": int, "test": int}.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(
            f"Train, val, and test ratios must sum to 1.0. Current sum: {total_ratio}"
        )

    # Load Label Studio task ID mapping early to support fixed test/val set assignment
    ls_mapping = load_label_studio_mapping(export_json_path=export_json_path)

    def resolve_image_id(file_path: str):
        stem = Path(file_path).stem
        name = Path(file_path).name
        if stem in ls_mapping:
            return ls_mapping[stem]
        if name in ls_mapping:
            return ls_mapping[name]
        return int(stem) if stem.isdigit() else stem

    # Check for fixed test IDs
    target_fixed_test_path = None
    if fixed_test_ids_path and Path(fixed_test_ids_path).exists():
        target_fixed_test_path = Path(fixed_test_ids_path)
    elif (Path(output_dir) / "test_image_ids.json").exists():
        target_fixed_test_path = Path(output_dir) / "test_image_ids.json"

    fixed_test_set = load_fixed_id_set(target_fixed_test_path)
    if fixed_test_set:
        print(f"--> Using fixed test set with {len(fixed_test_set)} IDs from: {target_fixed_test_path}")

    # Check for fixed val IDs
    target_fixed_val_path = None
    if fixed_val_ids_path and Path(fixed_val_ids_path).exists():
        target_fixed_val_path = Path(fixed_val_ids_path)
    elif (Path(output_dir) / "val_image_ids.json").exists():
        target_fixed_val_path = Path(output_dir) / "val_image_ids.json"

    fixed_val_set = load_fixed_id_set(target_fixed_val_path)
    if fixed_val_set:
        print(f"--> Using fixed val set with {len(fixed_val_set)} IDs from: {target_fixed_val_path}")

    # Check for overlap between fixed test and val sets
    overlap = fixed_test_set.intersection(fixed_val_set)
    if overlap:
        raise ValueError(
            f"Data leakage detected! Fixed test and validation sets share {len(overlap)} overlapping IDs: {overlap}"
        )

    setup_directories(output_dir, clean_output=clean_output)
    paired_files = get_paired_files(images_dir, labels_dir)

    if not paired_files:
        print("No paired files found. Please check your input directories.")
        return {"train": 0, "val": 0, "test": 0}

    # First split: Separate test set (using fixed test set if available)
    if fixed_test_set:
        test_pairs = [p for p in paired_files if resolve_image_id(p[0]) in fixed_test_set]
        temp_pairs = [p for p in paired_files if resolve_image_id(p[0]) not in fixed_test_set]
        print(f"--> Successfully matched {len(test_pairs)} exact test pairs using fixed test IDs.")
    elif test_ratio > 0:
        temp_pairs, test_pairs = train_test_split(
            paired_files, test_size=test_ratio, random_state=seed
        )
    else:
        temp_pairs, test_pairs = paired_files, []

    # Second split: Separate train and validation sets (using fixed val set if available)
    if fixed_val_set:
        val_pairs = [p for p in temp_pairs if resolve_image_id(p[0]) in fixed_val_set]
        train_pairs = [p for p in temp_pairs if resolve_image_id(p[0]) not in fixed_val_set]
        print(f"--> Successfully matched {len(val_pairs)} exact validation pairs using fixed val IDs.")
        print(f"--> Remaining {len(train_pairs)} pairs assigned to training set (zero randomness).")
    elif val_ratio > 0:
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

    # Extract and save validation and test image ID arrays (Label Studio IDs)
    val_ids = sorted(
        [resolve_image_id(p[0]) for p in val_pairs],
        key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)),
    )
    test_ids = sorted(
        [resolve_image_id(p[0]) for p in test_pairs],
        key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)),
    )

    val_json_path = Path(output_dir) / "val_image_ids.json"
    test_json_path = Path(output_dir) / "test_image_ids.json"

    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_ids, f, indent=2)

    split_counts["val_ids"] = val_ids
    split_counts["test_ids"] = test_ids
    split_counts["val_ids_path"] = str(val_json_path)
    split_counts["test_ids_path"] = str(test_json_path)

    print("\n--- Split Complete ---")
    print(f"Total dataset size: {len(paired_files)}")
    print(f"Train set: {split_counts['train']} pairs")
    print(f"Validation set: {split_counts['val']} pairs (Label Studio IDs saved to {val_json_path.name})")
    print(f"Test set: {split_counts['test']} pairs (Label Studio IDs saved to {test_json_path.name})")
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
    parser.add_argument(
        "--export_json_path",
        type=str,
        default="data/exports/export.json",
        help="Path to Label Studio JSON export file to map image filenames to Label Studio task IDs.",
    )
    parser.add_argument(
        "--test-ids-file",
        type=str,
        default=None,
        help="Path to JSON file containing exact test image IDs to isolate for the test split.",
    )
    parser.add_argument(
        "--val-ids-file",
        type=str,
        default=None,
        help="Path to JSON file containing exact validation image IDs to isolate for the validation split.",
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
        export_json_path=args.export_json_path,
        fixed_test_ids_path=args.test_ids_file,
        fixed_val_ids_path=args.val_ids_file,
    )


if __name__ == "__main__":
    main()


