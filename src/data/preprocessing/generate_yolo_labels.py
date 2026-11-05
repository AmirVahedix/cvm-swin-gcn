#!/usr/bin/env python3
"""
Generate YOLO Object Detection Labels from Anatomical Landmarks.

Iterates over dataset splits (train, val, test) and converts the 13 ground truth
vertebra landmarks (C2-C4) into a single-class ('cervical_spine', class_id 0)
YOLO-format bounding box with padding margins, matching compute_roi_bbox logic.

YOLO BBox format:
    <class_id> <x_center> <y_center> <width> <height>
    where coordinates and dimensions are normalized to [0, 1].
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.utils.roi_utils import compute_roi_bbox, ROIBoundingBox


def landmark_coords_to_yolo_bbox(
    coords: np.ndarray,
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.22,
    bottom_margin_ratio: Optional[float] = 0.28,
    min_size: int = 180,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Computes a padded isotropic bounding box from landmark coordinates and formats
    it as normalized YOLO coordinates (x_center, y_center, width, height) in [0, 1].

    Returns:
        (x_center, y_center, width, height) all in [0, 1], or None if coords invalid.
    """
    roi: ROIBoundingBox = compute_roi_bbox(
        coords=coords,
        img_w=img_w,
        img_h=img_h,
        margin_ratio=margin_ratio,
        bottom_margin_ratio=bottom_margin_ratio,
        min_size=min_size,
        use_anatomical_prior=True,
    )

    if roi.is_fallback:
        # Fallback anchor can still be used as an approximate box, but let's check
        pass

    # Clip box coordinates to image canvas to compute valid YOLO target
    x1 = max(0.0, float(roi.u1))
    y1 = max(0.0, float(roi.v1))
    x2 = min(float(img_w), float(roi.u2))
    y2 = min(float(img_h), float(roi.v2))

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return None

    # Normalized center and dimensions
    cx = (x1 + x2) / 2.0 / float(img_w)
    cy = (y1 + y2) / 2.0 / float(img_h)
    nw = box_w / float(img_w)
    nh = box_h / float(img_h)

    # Strictly clamp to [0, 1]
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    nw = float(np.clip(nw, 0.0, 1.0))
    nh = float(np.clip(nh, 0.0, 1.0))

    return cx, cy, nw, nh


def generate_yolo_dataset(
    dataset_root: str = "dataset",
    output_yolo_dir: str = "dataset/yolo",
    margin_ratio: float = 0.22,
    bottom_margin_ratio: Optional[float] = 0.28,
    min_size: int = 180,
    link_images: bool = True,
) -> Dict[str, int]:
    """
    Generates YOLO dataset from existing dataset/{train,val,test} directories.
    Creates:
        output_yolo_dir/
            images/train, images/val, images/test
            labels/train, labels/val, labels/test
            cvm_roi.yaml
    """
    dataset_path = Path(dataset_root)
    yolo_path = Path(output_yolo_dir)
    splits = ["train", "val", "test"]

    counts = {}

    for split in splits:
        img_src_dir = dataset_path / split / "images"
        lbl_src_dir = dataset_path / split / "labels"

        img_dst_dir = yolo_path / "images" / split
        lbl_dst_dir = yolo_path / "labels" / split

        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

        if not lbl_src_dir.exists():
            print(f"⚠️ Warning: Split directory not found: {lbl_src_dir}")
            counts[split] = 0
            continue

        label_files = sorted(lbl_src_dir.glob("*.npz"))
        valid_count = 0

        for npz_file in tqdm(label_files, desc=f"Generating YOLO {split} labels"):
            stem = npz_file.stem
            # Find matching image
            matching_imgs = list(img_src_dir.glob(f"{stem}.*"))
            if not matching_imgs:
                continue
            src_img_file = matching_imgs[0]

            try:
                data = np.load(npz_file)
                coords = data["coords"]  # [13, 2] in normalized or pixel scale
            except Exception as e:
                print(f"⚠️ Error reading {npz_file}: {e}")
                continue

            # Load image dimensions
            img = cv2.imread(str(src_img_file))
            if img is None:
                continue
            h, w = img.shape[:2]

            bbox = landmark_coords_to_yolo_bbox(
                coords=coords,
                img_w=w,
                img_h=h,
                margin_ratio=margin_ratio,
                bottom_margin_ratio=bottom_margin_ratio,
                min_size=min_size,
            )

            if bbox is None:
                continue

            cx, cy, nw, nh = bbox
            yolo_label_path = lbl_dst_dir / f"{stem}.txt"
            with open(yolo_label_path, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            # Destination image
            dst_img_file = img_dst_dir / src_img_file.name
            if not dst_img_file.exists():
                if link_images:
                    try:
                        os.symlink(os.path.abspath(src_img_file), dst_img_file)
                    except OSError:
                        shutil.copy2(src_img_file, dst_img_file)
                else:
                    shutil.copy2(src_img_file, dst_img_file)

            valid_count += 1

        counts[split] = valid_count
        print(f"✅ Split '{split}': Generated {valid_count} YOLO annotations in {lbl_dst_dir}")

    # Write cvm_roi.yaml
    yaml_data = {
        "path": str(yolo_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "cervical_spine",
        },
    }

    yaml_file = yolo_path / "cvm_roi.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"\n📄 Saved YOLO dataset config: {yaml_file}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLO labels from anatomical landmarks")
    parser.add_argument("--dataset-root", default="dataset", help="Root of dataset splits")
    parser.add_argument("--output-yolo-dir", default="dataset/yolo", help="Output directory for YOLO format")
    parser.add_argument("--margin-ratio", type=float, default=0.22, help="Padding margin ratio")
    parser.add_argument("--bottom-margin-ratio", type=float, default=0.28, help="Bottom margin ratio")
    parser.add_argument("--min-size", type=int, default=180, help="Minimum ROI square size")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking")

    args = parser.parse_args()

    generate_yolo_dataset(
        dataset_root=args.dataset_root,
        output_yolo_dir=args.output_yolo_dir,
        margin_ratio=args.margin_ratio,
        bottom_margin_ratio=args.bottom_margin_ratio,
        min_size=args.min_size,
        link_images=not args.copy_images,
    )
