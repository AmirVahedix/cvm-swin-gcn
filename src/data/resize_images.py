import json
import os
import cv2
import argparse
from tqdm import tqdm


def process_and_resize_dataset(
    img_dir, json_path, out_img_dir, out_json_path, target_size=640
):
    """
    Reads Label Studio JSON, crops (2400, 1935) images from the top to make them square,
    resizes to (640, 640) without distortion, and translates the JSON coordinates.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    print(f"📄 Loading JSON from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    processed_count = 0
    skipped_count = 0

    for task in tqdm(
        tasks, desc="Processing Images & Labels", unit="img", colour="blue"
    ):
        img_url = task["data"]["img"]

        # Replicate your download script's filename cleaning logic
        base_name = os.path.splitext(os.path.basename(img_url))[0]
        clean_name = (
            img_url.split("-", 1)[1] if "-" in base_name else os.path.basename(img_url)
        )
        img_ext = os.path.splitext(clean_name)[1]
        clean_base_name = os.path.splitext(clean_name)[0]
        img_filename = f"{clean_base_name}{img_ext}"

        img_path = os.path.join(img_dir, img_filename)
        out_path = os.path.join(out_img_dir, img_filename)

        if not os.path.exists(img_path):
            tqdm.write(f"⚠️ Missing image: {img_filename}")
            skipped_count += 1
            continue

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            tqdm.write(f"❌ Failed to read: {img_filename}")
            skipped_count += 1
            continue

        h, w = img.shape[:2]

        # Scenario 1: Image is already target size (640x640)
        if h == target_size and w == target_size:
            cv2.imwrite(out_path, img)
            update_annotations(task, h, w, target_size, crop_y=0)
            processed_count += 1
            continue

        # Scenario 2: Image is portrait, e.g., (2400, 1935)
        if h > w:
            # 1. Crop from the top to make it square
            crop_y = h - w
            cropped_img = img[crop_y:, :]  # Image is now W x W

            # 2. Resize the square to 640x640 (No distortion)
            resized_img = cv2.resize(
                cropped_img, (target_size, target_size), interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(out_path, resized_img)

            # 3. Update the JSON coordinates
            update_annotations(task, h, w, target_size, crop_y)
            processed_count += 1
        else:
            # Fallback for unexpected landscape images (ignores cropping top, just resizes)
            tqdm.write(
                f"⚠️ Warning: {img_filename} is not portrait ({w}x{h}). Force resizing."
            )
            resized_img = cv2.resize(
                img, (target_size, target_size), interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(out_path, resized_img)
            update_annotations(task, h, w, target_size, crop_y=0)
            processed_count += 1

    print(f"💾 Saving updated JSON export to: {out_json_path}")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4)

    print(f"\n🚀 Done! Processed {processed_count} images. Skipped {skipped_count}.")


def update_annotations(task, orig_h, orig_w, target_size, crop_y):
    """
    Updates the keypoint coordinates inside the Label Studio task dictionary.
    """
    # The new base dimension for percentage calculation after cropping to square
    new_orig_h = orig_h - crop_y

    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "keypointlabels":
                val = result.get("value", {})

                if crop_y > 0:
                    # 1. Convert Y percentage to absolute original pixels
                    abs_y = (val["y"] * orig_h) / 100.0

                    # 2. Apply the top crop shift
                    new_abs_y = abs_y - crop_y

                    # 3. Convert back to percentage relative to the new square dimension
                    # Note: We don't touch X, because cropping from top doesn't change width,
                    # and the subsequent resize to 640x640 keeps the percentages identical.
                    val["y"] = (new_abs_y / new_orig_h) * 100.0

                # 4. Update the Label Studio metadata to reflect the new 640x640 reality
                result["original_width"] = target_size
                result["original_height"] = target_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop, resize, and adjust JSON coordinates for Cephalometric landmarks."
    )
    parser.add_argument(
        "--img_dir",
        default="data/raw/images",
        help="Directory containing the raw downloaded images.",
    )
    parser.add_argument(
        "--json_path",
        default="data/raw/exports/export.json",
        help="Path to the original Label Studio JSON export.",
    )
    parser.add_argument(
        "--out_img_dir",
        default="data/images",
        help="Directory to save the resulting 640x640 images.",
    )
    parser.add_argument(
        "--out_json_path",
        default="data/exports/export.json",
        help="Path to save the modified JSON export.",
    )

    args = parser.parse_args()

    process_and_resize_dataset(
        img_dir=args.img_dir,
        json_path=args.json_path,
        out_img_dir=args.out_img_dir,
        out_json_path=args.out_json_path,
    )
