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
            raw_data = json.load(f)
        tasks = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data
        for task in tasks:
            task_id = task.get("taskId") or task.get("task_id") or task.get("id")
            if task_id is None:
                continue
            clean_filename = task.get("filename")
            if not clean_filename:
                raw_path = task.get("file_upload") or task.get("data", {}).get("img") or task.get("data", {}).get("image")
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
            mapping[clean_filename] = int(task_id)
            mapping[stem] = int(task_id)
        source_desc = "CVM API" if (isinstance(raw_data, dict) and raw_data.get("dataSource") == "cvm-backend-api") else "dataset"
        print(f"--> Loaded {source_desc} task ID mapping for {len(mapping)//2} images from: {found_path}")
    except Exception as e:
        print(f"⚠️ Warning loading task ID mapping from {found_path}: {e}")

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


def extract_all_cohort_identifiers(export_json_path=None) -> dict:
    """
    Scans candidate export JSON files for tasks and categorizes them into:
    'test', 'val', 'train', 'excluded', and 'review_needed'.
    Returns:
        dict: {
            "test_ids": set, "test_stems": set,
            "val_ids": set, "val_stems": set,
            "train_ids": set, "train_stems": set,
            "excluded_ids": set, "excluded_stems": set,
            "review_needed_ids": set, "review_needed_stems": set,
        }
    """
    cohort_data = {
        "test_ids": set(),
        "test_stems": set(),
        "val_ids": set(),
        "val_stems": set(),
        "train_ids": set(),
        "train_stems": set(),
        "excluded_ids": set(),
        "excluded_stems": set(),
        "review_needed_ids": set(),
        "review_needed_stems": set(),
    }

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
        return cohort_data

    try:
        with open(found_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        tasks = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data
        for task in tasks:
            task_id = task.get("taskId") or task.get("task_id") or task.get("id")
            filename = task.get("filename")
            if not filename:
                raw_path = (
                    task.get("file_upload")
                    or task.get("data", {}).get("img")
                    or task.get("data", {}).get("image")
                )
                if raw_path:
                    filename = os.path.basename(urlparse(raw_path).path)

            ids = set()
            if task_id is not None:
                ids.add(int(task_id) if str(task_id).isdigit() else str(task_id))
                ids.add(str(task_id))

            stems = set()
            if filename:
                stems.add(filename)
                stems.add(Path(filename).stem)

            is_excluded = bool(
                task.get("excluded")
                or task.get("isExcluded")
                or task.get("is_excluded")
            )
            is_review_needed = bool(
                task.get("reviewNeeded")
                or task.get("review_needed")
            )

            if is_excluded:
                cohort_data["excluded_ids"].update(ids)
                cohort_data["excluded_stems"].update(stems)
                continue

            if is_review_needed:
                cohort_data["review_needed_ids"].update(ids)
                cohort_data["review_needed_stems"].update(stems)
                continue

            is_test = bool(
                task.get("isTestCohort")
                or task.get("is_test_cohort")
                or task.get("testCohort")
                or task.get("test_cohort")
                or task.get("cohort") == "test"
            )
            is_val = bool(
                task.get("isValCohort")
                or task.get("is_val_cohort")
                or task.get("valCohort")
                or task.get("val_cohort")
                or task.get("cohort") == "val"
            )

            if is_test:
                cohort_data["test_ids"].update(ids)
                cohort_data["test_stems"].update(stems)
            elif is_val:
                cohort_data["val_ids"].update(ids)
                cohort_data["val_stems"].update(stems)
            else:
                cohort_data["train_ids"].update(ids)
                cohort_data["train_stems"].update(stems)

        test_cnt = len(cohort_data["test_stems"]) // 2 if cohort_data["test_stems"] else len(cohort_data["test_ids"]) // 2
        val_cnt = len(cohort_data["val_stems"]) // 2 if cohort_data["val_stems"] else len(cohort_data["val_ids"]) // 2
        train_cnt = len(cohort_data["train_stems"]) // 2 if cohort_data["train_stems"] else len(cohort_data["train_ids"]) // 2
        print(f"--> Loaded cohort definitions from {found_path}: Train={train_cnt}, Val={val_cnt}, Test={test_cnt}")
    except Exception as e:
        print(f"⚠️ Warning reading cohorts from {found_path}: {e}")

    return cohort_data


def extract_test_cohort_identifiers(export_json_path=None) -> tuple[set, set]:
    """
    Backwards compatibility helper: returns (test_ids, test_stems).
    """
    cohorts = extract_all_cohort_identifiers(export_json_path=export_json_path)
    return cohorts["test_ids"], cohorts["test_stems"]


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
    use_test_cohort=True,
    use_fixed_cohorts=True,
):
    """
    Assigns image/label pairs into train, val, and test splits and copies them into output_dir.

    When use_fixed_cohorts=True (default), uses the fixed cohorts defined directly in the
    CVM Backend API / export JSON (train, val, test) with zero random splitting.
    Any images flagged as 'review_needed' or 'excluded' are strictly ignored and never placed
    into any cohort.

    If fixed cohort definitions are absent in the export or use_fixed_cohorts=False,
    falls back to ratio-based splitting.

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
        return f"file_{stem}" if ls_mapping else (int(stem) if stem.isdigit() else stem)

    # Extract all cohort identifiers (test, val, train, excluded, review_needed)
    cohort_info = extract_all_cohort_identifiers(export_json_path=export_json_path)

    # Check for fixed test IDs override file
    target_fixed_test_path = None
    if fixed_test_ids_path and Path(fixed_test_ids_path).exists():
        target_fixed_test_path = Path(fixed_test_ids_path)
    elif (Path(output_dir) / "test_image_ids.json").exists() and not (cohort_info["test_ids"] or cohort_info["test_stems"]):
        target_fixed_test_path = Path(output_dir) / "test_image_ids.json"

    fixed_test_set = load_fixed_id_set(target_fixed_test_path)
    if fixed_test_set:
        print(f"--> Using fixed test set with {len(fixed_test_set)} IDs from: {target_fixed_test_path}")

    # Check for fixed val IDs override file
    target_fixed_val_path = None
    if fixed_val_ids_path and Path(fixed_val_ids_path).exists():
        target_fixed_val_path = Path(fixed_val_ids_path)
    elif (Path(output_dir) / "val_image_ids.json").exists() and not (cohort_info["val_ids"] or cohort_info["val_stems"]):
        target_fixed_val_path = Path(output_dir) / "val_image_ids.json"

    fixed_val_set = load_fixed_id_set(target_fixed_val_path)
    if fixed_val_set:
        print(f"--> Using fixed val set with {len(fixed_val_set)} IDs from: {target_fixed_val_path}")

    # Check for overlap between explicitly provided fixed test and val files
    if fixed_test_set and fixed_val_set:
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

    # Filter out any files flagged as excluded or review_needed
    def is_ignored(img_path):
        stem = Path(img_path).stem
        name = Path(img_path).name
        img_id = resolve_image_id(img_path)
        return (
            (img_id in cohort_info["excluded_ids"])
            or (stem in cohort_info["excluded_stems"])
            or (name in cohort_info["excluded_stems"])
            or (img_id in cohort_info["review_needed_ids"])
            or (stem in cohort_info["review_needed_stems"])
            or (name in cohort_info["review_needed_stems"])
        )

    clean_pairs = [p for p in paired_files if not is_ignored(p[0])]
    skipped_count = len(paired_files) - len(clean_pairs)
    if skipped_count > 0:
        print(f"--> Strictly excluded {skipped_count} images flagged as 'review_needed' or 'excluded' from dataset splits.")

    def matches_test(img_path):
        if fixed_test_set:
            return resolve_image_id(img_path) in fixed_test_set
        if not use_test_cohort:
            return False
        stem = Path(img_path).stem
        name = Path(img_path).name
        img_id = resolve_image_id(img_path)
        return (
            (img_id in cohort_info["test_ids"])
            or (stem in cohort_info["test_stems"])
            or (name in cohort_info["test_stems"])
        )

    def matches_val(img_path):
        if fixed_val_set:
            return resolve_image_id(img_path) in fixed_val_set
        stem = Path(img_path).stem
        name = Path(img_path).name
        img_id = resolve_image_id(img_path)
        return (
            (img_id in cohort_info["val_ids"])
            or (stem in cohort_info["val_stems"])
            or (name in cohort_info["val_stems"])
        )

    has_fixed_api_cohorts = bool(
        (cohort_info["test_ids"] or cohort_info["test_stems"] or fixed_test_set)
        and (cohort_info["val_ids"] or cohort_info["val_stems"] or fixed_val_set)
    )

    if use_fixed_cohorts and has_fixed_api_cohorts:
        # Strict API cohorts: zero random splitting
        test_pairs = [p for p in clean_pairs if matches_test(p[0])]
        val_pairs = [p for p in clean_pairs if not matches_test(p[0]) and matches_val(p[0])]
        train_pairs = [p for p in clean_pairs if not matches_test(p[0]) and not matches_val(p[0])]
        print(f"\n--> ✅ Using Fixed API Cohorts (Zero Random Splitting):")
        print(f"    • Train cohort:      {len(train_pairs)} pairs")
        print(f"    • Validation cohort: {len(val_pairs)} pairs")
        print(f"    • Test cohort:       {len(test_pairs)} pairs")
    else:
        # Fallback: Ratio-based splitting or partial fixed sets
        if fixed_test_set or (use_test_cohort and (cohort_info["test_ids"] or cohort_info["test_stems"])):
            test_pairs = [p for p in clean_pairs if matches_test(p[0])]
            temp_pairs = [p for p in clean_pairs if not matches_test(p[0])]
            print(f"--> Isolated {len(test_pairs)} test cohort images strictly for evaluation test split.")
        elif test_ratio > 0:
            temp_pairs, test_pairs = train_test_split(
                clean_pairs, test_size=test_ratio, random_state=seed
            )
        else:
            temp_pairs, test_pairs = clean_pairs, []

        if fixed_val_set or (cohort_info["val_ids"] or cohort_info["val_stems"]):
            val_pairs = [p for p in temp_pairs if matches_val(p[0])]
            train_pairs = [p for p in temp_pairs if not matches_val(p[0])]
            print(f"--> Matched {len(val_pairs)} validation pairs from fixed validation set.")
        elif val_ratio > 0 and len(temp_pairs) > 0:
            relative_val_ratio = val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio > 0) else val_ratio
            train_pairs, val_pairs = train_test_split(
                temp_pairs, test_size=relative_val_ratio, random_state=seed
            )
        else:
            train_pairs, val_pairs = temp_pairs, []

    # Verify no data leakage between splits
    test_ids_set = {resolve_image_id(p[0]) for p in test_pairs}
    val_ids_set = {resolve_image_id(p[0]) for p in val_pairs}
    train_ids_set = {resolve_image_id(p[0]) for p in train_pairs}

    overlap_test_val = test_ids_set.intersection(val_ids_set)
    if overlap_test_val:
        raise ValueError(f"Data leakage detected! Test and validation sets share overlapping IDs: {overlap_test_val}")
    overlap_train_val = train_ids_set.intersection(val_ids_set)
    if overlap_train_val:
        raise ValueError(f"Data leakage detected! Train and validation sets share overlapping IDs: {overlap_train_val}")
    overlap_train_test = train_ids_set.intersection(test_ids_set)
    if overlap_train_test:
        raise ValueError(f"Data leakage detected! Train and test sets share overlapping IDs: {overlap_train_test}")

    print("\n--- Starting Data Transfer ---")
    copy_files(train_pairs, "train", output_dir)
    copy_files(val_pairs, "val", output_dir)
    copy_files(test_pairs, "test", output_dir)

    split_counts = {
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
    }

    # Extract and save validation, test, and train image ID arrays (Label Studio IDs)
    val_ids = sorted(
        list(val_ids_set),
        key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)),
    )
    test_ids = sorted(
        list(test_ids_set),
        key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)),
    )
    train_ids = sorted(
        list(train_ids_set),
        key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)),
    )

    val_json_path = Path(output_dir) / "val_image_ids.json"
    test_json_path = Path(output_dir) / "test_image_ids.json"
    train_json_path = Path(output_dir) / "train_image_ids.json"

    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_ids, f, indent=2)

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2)

    # Save stage mapping if available in candidate exports
    try:
        from src.data.loaders.dataloader import load_stage_mapping
        stage_map = load_stage_mapping(export_json_path=export_json_path)
        if stage_map:
            stage_json_path = Path(output_dir) / "stage_mapping.json"
            with open(stage_json_path, "w", encoding="utf-8") as f:
                json.dump(stage_map, f, indent=2)
            print(f"--> Saved stage mapping ({len(stage_map)} entries) to {stage_json_path.name}")
    except Exception as e:
        pass

    split_counts["train_ids"] = train_ids
    split_counts["val_ids"] = val_ids
    split_counts["test_ids"] = test_ids
    split_counts["train_ids_path"] = str(train_json_path)
    split_counts["val_ids_path"] = str(val_json_path)
    split_counts["test_ids_path"] = str(test_json_path)

    print("\n--- Split Complete ---")
    print(f"Total dataset size: {len(paired_files)}")
    print(f"Train set: {split_counts['train']} pairs (Label Studio IDs saved to {train_json_path.name})")
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
    parser.add_argument(
        "--use-test-cohort",
        dest="use_test_cohort",
        action="store_true",
        default=True,
        help="Strictly isolate images marked as test cohort in API/export for test evaluation split (default: True).",
    )
    parser.add_argument(
        "--no-test-cohort",
        dest="use_test_cohort",
        action="store_false",
        help="Disable automatic test cohort isolation for test split.",
    )
    parser.add_argument(
        "--use-fixed-cohorts",
        dest="use_fixed_cohorts",
        action="store_true",
        default=True,
        help="Use fixed cohorts directly from CVM Backend API/export without random splitting (default: True).",
    )
    parser.add_argument(
        "--no-fixed-cohorts",
        dest="use_fixed_cohorts",
        action="store_false",
        help="Disable fixed API cohorts and perform ratio-based splitting.",
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
        use_test_cohort=args.use_test_cohort,
        use_fixed_cohorts=args.use_fixed_cohorts,
    )


if __name__ == "__main__":
    main()


