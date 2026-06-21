import json
import os
import shutil  # <-- Added for wiping directories
import argparse
import numpy as np
import cv2
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

LANDMARK_CLASSES = [
    "C2_PI",
    "C2_IC",
    "C2_AI",
    "C3_PS",
    "C3_AS",
    "C3_PI",
    "C3_IC",
    "C3_AI",
    "C4_PS",
    "C4_AS",
    "C4_PI",
    "C4_IC",
    "C4_AI",
]


def prepare_empty_directory(path):
    """
    Ensures the target directory is completely empty.
    If it exists, it is deleted along with its contents, then recreated.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def generate_gaussian_heatmap(shape, center, sigma=3.0):
    """
    Generates a 2D Gaussian heatmap array.
    """
    h, w = shape
    x, y = center

    # Create a meshgrid
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate the Gaussian
    heatmap = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma**2))

    return heatmap.astype(np.float32)


def process_single_record(record, images_dir, output_dir, sigma):
    """
    Isolated function to process a single record.
    Returns a status string to be aggregated by the main process.
    """
    # 1. Extract raw path
    raw_path = record.get("file_upload") or record.get("data", {}).get("img")
    if not raw_path:
        return "SKIP_NO_PATH"

    # 2. Parse the URL
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

    # Verify image exists
    if not os.path.exists(img_path):
        return "MISSING_IMAGE"

    # Load image strictly to get dimensions (H, W)
    img = cv2.imread(img_path)
    if img is None:
        return "INVALID_IMAGE"

    h, w = img.shape[:2]

    # 3. Initialize the 13-channel numpy array
    heatmaps = np.zeros((13, h, w), dtype=np.float16)

    # 4. Parse annotations
    annotations = record.get("annotations", [])
    if not annotations:
        return "SKIP_NO_ANNOTATIONS"

    result_list = annotations[0].get("result", [])
    valid_keypoints = 0

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
            orig_w = item.get("original_width", w)
            orig_h = item.get("original_height", h)

            abs_x = int((val.get("x", 0) * orig_w) / 100.0)
            abs_y = int((val.get("y", 0) * orig_h) / 100.0)

            # Generate and assign the heatmap
            heatmap_layer = generate_gaussian_heatmap((h, w), (abs_x, abs_y), sigma)
            heatmaps[channel_idx] = heatmap_layer.astype(np.float16)
            valid_keypoints += 1

    if valid_keypoints == 0:
        return "SKIP_NO_VALID_KEYPOINTS"

    # 5. Save to .npz
    base_name = os.path.splitext(clean_filename)[0]
    npz_output_path = os.path.join(output_dir, f"{base_name}.npz")

    # Save as compressed
    np.savez_compressed(npz_output_path, heatmaps=heatmaps)
    return "SUCCESS"


def process_label_studio_export(
    json_path, images_dir, output_dir, sigma, max_workers=None
):
    # <-- Replaced the old directory logic here
    prepare_empty_directory(output_dir)

    # Load the Label Studio export
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} records in {os.path.basename(json_path)}")

    processed_count = 0
    missing_images = 0

    # Create a partial function with the static arguments locked in
    worker_func = partial(
        process_single_record, images_dir=images_dir, output_dir=output_dir, sigma=sigma
    )

    # Determine core count (default to all available logical cores if max_workers is None)
    workers = max_workers or os.cpu_count()
    print(f"Starting multiprocessing pool with {workers} parallel workers...\n")

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(worker_func, record) for record in data]

        # Use as_completed to update the progress bar the moment any process finishes
        for future in tqdm(
            as_completed(futures),
            total=len(data),
            desc="Generating Heatmaps",
            unit="img",
        ):
            result = future.result()

            if result == "SUCCESS":
                processed_count += 1
            elif result == "MISSING_IMAGE":
                missing_images += 1

    # Final summary output
    print("\n" + "-" * 30)
    print("Export Complete!")
    print(f"Successfully generated: {processed_count} .npz files.")
    if missing_images > 0:
        print(f"Missing images in directory: {missing_images}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Label Studio keypoint exports into NPZ heatmap masks in parallel."
    )

    parser.add_argument(
        "--json_path",
        type=str,
        default="data/exports/export.json",
        help="Path to the Label Studio export JSON file.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/images",
        help="Directory containing the source images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/heatmaps",
        help="Directory where the .npz heatmaps will be saved.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Spread of the Gaussian heatmap. Adjust based on your image resolution.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of CPU cores to use. Defaults to all available cores.",
    )

    args = parser.parse_args()

    process_label_studio_export(
        json_path=args.json_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        sigma=args.sigma,
        max_workers=args.workers,
    )
