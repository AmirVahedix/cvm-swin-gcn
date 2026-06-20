import json
import os
import argparse
import numpy as np
import cv2
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm

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


def create_directory(path):
    if not os.path.exists(path):
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


def process_label_studio_export(json_path, images_dir, output_dir, sigma):
    create_directory(output_dir)

    # Load the Label Studio export
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} records in {os.path.basename(json_path)}")

    processed_count = 0
    missing_images = 0

    # Wrap the data loop with tqdm for a visual progress bar
    for record in tqdm(data, desc="Generating Heatmaps", unit="img"):
        # 1. Extract raw path from either file_upload or data.image
        raw_path = record.get("file_upload") or record.get("data", {}).get("img")

        if not raw_path:
            # We skip printing here to avoid messing up the progress bar UI
            continue

        # 2. Parse the URL to handle Local Storage parameters (e.g., ?d=cvm-images/0000.jpg)
        parsed_url = urlparse(raw_path)
        query_params = parse_qs(parsed_url.query)

        if "d" in query_params:
            # Extracts '0000.jpg' from 'cvm-images/0000.jpg'
            filename = os.path.basename(query_params["d"][0])
        else:
            # Fallback for standard Label Studio file uploads
            filename = os.path.basename(parsed_url.path)

        # Clean up Label Studio's unique ID prefix if it exists (for standard uploads)
        if "-" in filename and len(filename.split("-")[0]) == 8:
            clean_filename = "-".join(filename.split("-")[1:])
        else:
            clean_filename = filename

        img_path = os.path.join(images_dir, clean_filename)

        # Verify image exists to get its shape
        if not os.path.exists(img_path):
            missing_images += 1
            continue

        # Load image strictly to get dimensions (H, W)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 3. Initialize the 13-channel numpy array (Shape: 13 x H x W)
        heatmaps = np.zeros((13, h, w), dtype=np.float16)

        # 4. Parse annotations
        annotations = record.get("annotations", [])
        if not annotations:
            continue

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

                # Label studio x, y are percentages. Convert to absolute pixels.
                orig_w = item.get("original_width", w)
                orig_h = item.get("original_height", h)

                abs_x = int((val.get("x", 0) * orig_w) / 100.0)
                abs_y = int((val.get("y", 0) * orig_h) / 100.0)

                # Generate and assign the heatmap
                heatmap_layer = generate_gaussian_heatmap((h, w), (abs_x, abs_y), sigma)
                heatmaps[channel_idx] = heatmap_layer.astype(np.float16)

        # 5. Save to .npz
        base_name = os.path.splitext(clean_filename)[0]
        npz_output_path = os.path.join(output_dir, f"{base_name}.npz")

        # Save as compressed to minimize storage size
        np.savez_compressed(npz_output_path, heatmaps=heatmaps)
        processed_count += 1

    # Final summary output
    print("\n" + "-" * 30)
    print("Export Complete!")
    print(f"Successfully generated: {processed_count} .npz files.")
    if missing_images > 0:
        print(f"Missing images in directory: {missing_images}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Label Studio keypoint exports into NPZ heatmap masks."
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
        default="data/npz_exports",
        help="Directory where the .npz heatmaps will be saved.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Spread of the Gaussian heatmap. Adjust based on your image resolution.",
    )

    args = parser.parse_args()

    process_label_studio_export(
        json_path=args.json_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        sigma=args.sigma,
    )
