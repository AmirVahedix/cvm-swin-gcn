import json
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm


def resize_images(img_dir, json_path, out_img_dir, out_json_path, target_size=1024):
    """
    Reads Label Studio JSON, resizes images of any size/aspect ratio to target_size x target_size
    without cropping or distortion using letterboxing padding, and translates JSON annotation coordinates.
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

        # Replicate filename cleaning logic
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

        # Calculate uniform scale factor to preserve aspect ratio without cropping
        scale = min(target_size / w, target_size / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Calculate symmetric letterbox padding
        pad_x_int = (target_size - new_w) // 2
        pad_y_int = (target_size - new_h) // 2

        if len(img.shape) == 3:
            canvas = np.zeros((target_size, target_size, img.shape[2]), dtype=img.dtype)
        else:
            canvas = np.zeros((target_size, target_size), dtype=img.dtype)

        canvas[pad_y_int : pad_y_int + new_h, pad_x_int : pad_x_int + new_w] = (
            resized_img
        )
        cv2.imwrite(out_path, canvas)

        # Exact scaling and offset matching where resized_img is placed on canvas
        scale_x = float(new_w) / float(w)
        scale_y = float(new_h) / float(h)
        pad_x = float(pad_x_int)
        pad_y = float(pad_y_int)

        # Update JSON coordinates to account for scaling and letterbox padding
        update_annotations(task, h, w, target_size, scale_x, scale_y, pad_x, pad_y)
        processed_count += 1

    print(f"💾 Saving updated JSON export to: {out_json_path}")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4)

    print(f"\n🚀 Done! Processed {processed_count} images. Skipped {skipped_count}.")


def update_annotations(task, orig_h, orig_w, target_size, scale_x, scale_y, pad_x, pad_y):
    """
    Updates annotation coordinates inside the Label Studio task dictionary.
    Supports keypointlabels, rectanglelabels, and polygonlabels.
    Guarantees exact sub-pixel coordinate alignment without drift from letterboxing.
    """
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            label_type = result.get("type")
            val = result.get("value", {})

            item_orig_w = float(result.get("original_width") or orig_w)
            item_orig_h = float(result.get("original_height") or orig_h)

            if label_type == "keypointlabels":
                abs_x = (float(val.get("x", 0.0)) * item_orig_w) / 100.0
                abs_y = (float(val.get("y", 0.0)) * item_orig_h) / 100.0

                new_abs_x = abs_x * scale_x + pad_x
                new_abs_y = abs_y * scale_y + pad_y

                val["x"] = (new_abs_x / float(target_size)) * 100.0
                val["y"] = (new_abs_y / float(target_size)) * 100.0

                result["original_width"] = target_size
                result["original_height"] = target_size

            elif label_type == "rectanglelabels":
                abs_x = (float(val.get("x", 0.0)) * item_orig_w) / 100.0
                abs_y = (float(val.get("y", 0.0)) * item_orig_h) / 100.0
                abs_w = (float(val.get("width", 0.0)) * item_orig_w) / 100.0
                abs_h = (float(val.get("height", 0.0)) * item_orig_h) / 100.0

                new_abs_x = abs_x * scale_x + pad_x
                new_abs_y = abs_y * scale_y + pad_y
                new_abs_w = abs_w * scale_x
                new_abs_h = abs_h * scale_y

                val["x"] = (new_abs_x / float(target_size)) * 100.0
                val["y"] = (new_abs_y / float(target_size)) * 100.0
                val["width"] = (new_abs_w / float(target_size)) * 100.0
                val["height"] = (new_abs_h / float(target_size)) * 100.0

                result["original_width"] = target_size
                result["original_height"] = target_size

            elif label_type == "polygonlabels":
                points = val.get("points", [])
                new_points = []
                for pt in points:
                    px, py = float(pt[0]), float(pt[1])
                    abs_x = (px * item_orig_w) / 100.0
                    abs_y = (py * item_orig_h) / 100.0

                    new_px = ((abs_x * scale_x + pad_x) / float(target_size)) * 100.0
                    new_py = ((abs_y * scale_y + pad_y) / float(target_size)) * 100.0
                    new_points.append([new_px, new_py])

                val["points"] = new_points
                result["original_width"] = target_size
                result["original_height"] = target_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resize images of any aspect ratio with letterboxing and adjust JSON coordinates."
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
        help="Directory to save the resulting resized images.",
    )
    parser.add_argument(
        "--out_json_path",
        default="data/exports/export.json",
        help="Path to save the modified JSON export.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=1024,
        help="Target square image dimension (default: 1024).",
    )

    args = parser.parse_args()

    resize_images(
        img_dir=args.img_dir,
        json_path=args.json_path,
        out_img_dir=args.out_img_dir,
        out_json_path=args.out_json_path,
        target_size=args.target_size,
    )
