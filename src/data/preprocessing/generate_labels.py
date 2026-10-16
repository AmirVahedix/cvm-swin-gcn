import json
import os
import shutil
import argparse
import numpy as np
import cv2
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS


def prepare_empty_directory(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def generate_gaussian_heatmap(shape, center, sigma=4.0):
    h, w = shape
    x, y = center
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    heatmap = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma**2))
    return heatmap.astype(np.float32)


def process_single_record(record, images_dir, output_dir, sigma):
    # Skip any record marked as review_needed
    if record.get("review_needed") or record.get("reviewNeeded"):
        return "SKIP_REVIEW_NEEDED"

    clean_filename = None
    if "filename" in record and record["filename"]:
        clean_filename = record["filename"]
    else:
        raw_path = record.get("file_upload") or record.get("data", {}).get("img") or record.get("data", {}).get("image")
        if not raw_path:
            return "SKIP_NO_PATH"

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

    img_path = os.path.join(images_dir, clean_filename)

    if not os.path.exists(img_path):
        return "MISSING_IMAGE"

    img = cv2.imread(img_path)
    if img is None:
        return "INVALID_IMAGE"

    h, w = img.shape[:2]

    # Initialize both heatmaps and GCN coordinates arrays using NUM_LANDMARKS
    heatmaps = np.zeros((NUM_LANDMARKS, h, w), dtype=np.float16)
    coords = np.full((NUM_LANDMARKS, 2), -1.0, dtype=np.float32)
    valid_keypoints = 0

    # 1. Check for CVM Backend API format (landmarks list)
    if "landmarks" in record and isinstance(record["landmarks"], list):
        for lm in record["landmarks"]:
            label_name = lm.get("id")
            if not label_name or label_name not in LANDMARK_CLASSES:
                continue

            channel_idx = LANDMARK_CLASSES.index(label_name)
            norm_x = float(lm.get("x", 0.0))
            norm_y = float(lm.get("y", 0.0))

            # Handle both normalized [0, 1] and percentage [0, 100]
            if norm_x > 1.0 or norm_y > 1.0:
                norm_x /= 100.0
                norm_y /= 100.0

            abs_x = norm_x * float(w)
            abs_y = norm_y * float(h)

            heatmap_layer = generate_gaussian_heatmap((h, w), (abs_x, abs_y), sigma)
            heatmaps[channel_idx] = heatmap_layer.astype(np.float16)
            coords[channel_idx] = [norm_x, norm_y]
            valid_keypoints += 1

    # 2. Check for Label Studio format (annotations -> result)
    elif "annotations" in record and isinstance(record["annotations"], list):
        annotations = record.get("annotations", [])
        if not annotations:
            return "SKIP_NO_ANNOTATIONS"

        result_list = annotations[0].get("result", [])
        for item in result_list:
            if item.get("type") == "keypointlabels":
                val = item.get("value", {})
                labels = val.get("keypointlabels", [])

                if not labels:
                    continue

                label_name = labels[0]
                if label_name not in LANDMARK_CLASSES:
                    continue

                channel_idx = LANDMARK_CLASSES.index(label_name)

                orig_w = float(item.get("original_width", w))
                orig_h = float(item.get("original_height", h))
                abs_x = (float(val.get("x", 0.0)) * orig_w) / 100.0
                abs_y = (float(val.get("y", 0.0)) * orig_h) / 100.0

                heatmap_layer = generate_gaussian_heatmap((h, w), (abs_x, abs_y), sigma)
                heatmaps[channel_idx] = heatmap_layer.astype(np.float16)

                norm_x = float(val.get("x", 0.0)) / 100.0
                norm_y = float(val.get("y", 0.0)) / 100.0
                coords[channel_idx] = [norm_x, norm_y]

                valid_keypoints += 1

    if valid_keypoints == 0:
        return "SKIP_NO_VALID_KEYPOINTS"

    base_name = os.path.splitext(clean_filename)[0]
    npz_output_path = os.path.join(output_dir, f"{base_name}.npz")

    # Save both arrays into the compressed archive
    np.savez_compressed(npz_output_path, heatmaps=heatmaps, coords=coords)
    return "SUCCESS"


def generate_labels(
    json_path: str = "data/exports/export.json",
    images_dir: str = "data/images",
    output_dir: str = "data/labels",
    sigma: float = 4.0,
    max_workers: int | None = None,
) -> int:
    """
    Processes export JSON (CVM Backend API or Label Studio) into NPZ heatmap masks and GCN coordinates.
    Automatically skips records marked as review_needed.

    Returns:
        int: Number of successfully generated label files (.npz).
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Export JSON file not found at: {json_path}")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")

    prepare_empty_directory(output_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict) and "data" in raw_data:
        data = raw_data["data"]
    elif isinstance(raw_data, list):
        data = raw_data
    else:
        data = []

    print(f"Found {len(data)} records in {os.path.basename(json_path)}")

    processed_count = 0
    missing_images = 0
    skipped_review = 0

    worker_func = partial(
        process_single_record, images_dir=images_dir, output_dir=output_dir, sigma=sigma
    )

    workers = max_workers or os.cpu_count()
    print(f"Starting multiprocessing pool with {workers} parallel workers...\n")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_func, record) for record in data]

        for future in tqdm(
            as_completed(futures),
            total=len(data),
            desc="Generating Heatmaps & GCN Labels",
            unit="img",
        ):
            result = future.result()

            if result == "SUCCESS":
                processed_count += 1
            elif result == "MISSING_IMAGE":
                missing_images += 1
            elif result == "SKIP_REVIEW_NEEDED":
                skipped_review += 1

    print("\n" + "-" * 30)
    print("Export Complete!")
    print(f"Successfully generated: {processed_count} .npz files.")
    if skipped_review > 0:
        print(f"Skipped review_needed images: {skipped_review}")
    if missing_images > 0:
        print(f"Missing images in directory: {missing_images}")

    return processed_count


# Backwards compatibility alias
process_label_studio_export = generate_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process keypoint exports (CVM API or Label Studio) into NPZ heatmap masks and GCN coordinates."
    )

    parser.add_argument(
        "--json_path",
        type=str,
        default="data/exports/export.json",
        help="Path to JSON export file",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/images",
        help="Path to images directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/labels",
        help="Path to output directory for .npz label files",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Gaussian heatmap sigma parameter (default: 4.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes",
    )

    args = parser.parse_args()

    generate_labels(
        json_path=args.json_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        sigma=args.sigma,
        max_workers=args.workers,
    )
