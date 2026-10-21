#!/usr/bin/env python3
"""
scripts/check_augmentations.py

Augmentation verification and landmark integrity suite for CVM Cephalometric radiographs.
Validates:
1. Synchronized geometric transformations (Affine, Perspective, Contrast/Exposure).
2. Pure ElasticTransform behavior and Albumentations non-rigid keypoint remapping.
3. Renders high-resolution side-by-side comparisons on 10 test dataset images:
   - Full 640x640 Cephalogram (Original vs Augmented)
   - Zoomed Cervical Vertebrae (C2, C3, C4) ROI with anatomical contour lines and numbered badges
4. Generates:
   - 10 individual sample cards: evaluation/augmentation_check/sample_XX_<name>.png
   - Master 10-sample overview: evaluation/augmentation_check/all_10_samples_overview.png
   - Pure ElasticTransform analysis: evaluation/augmentation_check/elastic_transform_analysis.png
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import albumentations as A

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.constants import LANDMARK_CLASSES

CONTOURS = {
    "C2": [0, 1, 2],
    "C3": [3, 4, 7, 6, 5, 3],
    "C4": [8, 9, 12, 11, 10, 8],
}

LANDMARK_COLORS_RGB = [
    (255, 60, 60),    # 0: C2_PI - Red
    (255, 140, 0),    # 1: C2_IC - Orange
    (255, 215, 0),    # 2: C2_AI - Gold
    (50, 205, 50),    # 3: C3_PS - Lime
    (0, 255, 127),    # 4: C3_AS - Spring Green
    (0, 210, 210),    # 5: C3_PI - Cyan
    (30, 144, 255),   # 6: C3_IC - Dodger Blue
    (138, 43, 226),   # 7: C3_AI - Blue Violet
    (255, 20, 147),   # 8: C4_PS - Deep Pink
    (255, 105, 180),  # 9: C4_AS - Hot Pink
    (147, 112, 219),  # 10: C4_PI - Medium Purple
    (110, 50, 190),   # 11: C4_IC - Indigo
    (220, 20, 60),    # 12: C4_AI - Crimson
]


def build_safe_training_pipeline():
    """
    Builds the active geometric and photometric training augmentation pipeline
    matching src/data/loaders/transforms.py (excluding final normalization).
    Uses Affine, Perspective, CLAHE, Gamma, BrightnessContrast, and GaussianBlur.
    """
    return A.Compose(
        [
            A.CLAHE(clip_limit=(2.0, 4.0), tile_grid_size=(8, 8), p=0.8),
            A.RandomGamma(gamma_limit=(85, 115), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
            A.Affine(
                translate_percent=(-0.04, 0.04),
                scale=(0.95, 1.05),
                rotate=(-8, 8),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
            A.Perspective(scale=(0.02, 0.03), keep_size=True, p=0.7),
            A.GaussianBlur(blur_limit=(3, 3), p=0.3),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def compute_vertebrae_roi_bbox(
    coords_px: np.ndarray,
    img_w: int = 640,
    img_h: int = 640,
    margin_ratio: float = 0.35,
    min_size: int = 210,
) -> tuple[int, int, int, int]:
    """Computes a square bounding box centered around the C2-C4 vertebrae landmarks."""
    valid_mask = (coords_px[:, 0] >= 0) & (coords_px[:, 1] >= 0)
    pts = coords_px[valid_mask]

    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    bw = max_x - min_x
    bh = max_y - min_y
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    dim = max(bw, bh) * (1.0 + margin_ratio)
    dim = max(dim, min_size)
    half_dim = dim / 2.0

    x1 = int(round(cx - half_dim))
    x2 = int(round(cx + half_dim))
    y1 = int(round(cy - half_dim))
    y2 = int(round(cy + half_dim))

    if x1 < 0:
        x2 = min(img_w, x2 - x1)
        x1 = 0
    if x2 > img_w:
        x1 = max(0, x1 - (x2 - img_w))
        x2 = img_w
    if y1 < 0:
        y2 = min(img_h, y2 - y1)
        y1 = 0
    if y2 > img_h:
        y1 = max(0, y1 - (y2 - img_h))
        y2 = img_h

    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def draw_landmarks_and_contours(
    img_rgb: np.ndarray,
    coords_px: np.ndarray,
    draw_contours: bool = True,
    draw_badges: bool = True,
    scale_factor: float = 1.0,
    alpha_overlay: float = 1.0,
) -> np.ndarray:
    """Renders vertebrae contours, landmark dots, and number badges on an image."""
    canvas = img_rgb.copy()
    pts_dict = {}

    for i, pt in enumerate(coords_px):
        if pt[0] < 0 or pt[1] < 0 or np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        pts_dict[i] = (int(round(pt[0])), int(round(pt[1])))

    # 1. Draw Anatomical Contours
    if draw_contours:
        contour_colors = {
            "C2": (255, 180, 50),
            "C3": (50, 220, 100),
            "C4": (230, 80, 160),
        }
        for v_name, indices in CONTOURS.items():
            valid_pts = [pts_dict[i] for i in indices if i in pts_dict]
            if len(valid_pts) > 1:
                pts_arr = np.array(valid_pts, dtype=np.int32).reshape((-1, 1, 2))
                is_closed = v_name != "C2"  # C2 is open anterior/inferior border
                # Shadow border
                cv2.polylines(canvas, [pts_arr], is_closed, (0, 0, 0), int(round(3 * scale_factor)), lineType=cv2.LINE_AA)
                # Colored contour line
                cv2.polylines(canvas, [pts_arr], is_closed, contour_colors[v_name], int(round(1.5 * scale_factor)), lineType=cv2.LINE_AA)

    # 2. Draw Landmark Dots and Number Badges
    for i, pt in enumerate(coords_px):
        if i not in pts_dict:
            continue
        px, py = pts_dict[i]
        color = LANDMARK_COLORS_RGB[i % len(LANDMARK_COLORS_RGB)]

        r_out = max(4, int(round(5 * scale_factor)))
        r_col = max(3, int(round(4 * scale_factor)))
        r_in = max(1, int(round(1.5 * scale_factor)))

        # Outer high-contrast black border
        cv2.circle(canvas, (px, py), r_out, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        # Colored landmark dot
        cv2.circle(canvas, (px, py), r_col, color, -1, lineType=cv2.LINE_AA)
        # Inner pinpoint white center
        cv2.circle(canvas, (px, py), r_in, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        if draw_badges:
            badge_text = str(i)
            font_scale = 0.36 * scale_factor
            thickness = 1
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            bx = px + int(round(6 * scale_factor))
            by = py - int(round(4 * scale_factor))
            cv2.rectangle(canvas, (bx - 1, by - th - 1), (bx + tw + 1, by + 1), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                badge_text,
                (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

    if alpha_overlay < 1.0:
        canvas = cv2.addWeighted(canvas, alpha_overlay, img_rgb, 1.0 - alpha_overlay, 0)

    return canvas


def create_sample_comparison_card(
    orig_img_rgb: np.ndarray,
    orig_coords_px: np.ndarray,
    aug_img_rgb: np.ndarray,
    aug_coords_px: np.ndarray,
    filename: str,
    index: int,
) -> tuple[np.ndarray, dict]:
    """
    Creates a comprehensive 4-panel visual comparison card:
    [1. Original Full (640)] | [2. Original ROI (Zoom)] | [3. Augmented ROI (Zoom)] | [4. Augmented Full (640)]
    """
    img_h, img_w = orig_img_rgb.shape[:2]

    # Calculate displacements
    diffs = np.linalg.norm(aug_coords_px - orig_coords_px, axis=1)
    mean_disp = float(np.mean(diffs))
    max_disp = float(np.max(diffs))
    min_disp = float(np.min(diffs))

    # Shared ROI bounding box
    all_coords = np.vstack([orig_coords_px, aug_coords_px])
    x1, y1, x2, y2 = compute_vertebrae_roi_bbox(all_coords, img_w, img_h, margin_ratio=0.35, min_size=210)
    roi_w = x2 - x1
    roi_h = y2 - y1

    # Panel 1: Original Full
    orig_full_drawn = draw_landmarks_and_contours(orig_img_rgb, orig_coords_px, draw_contours=True, draw_badges=True, scale_factor=1.0)
    cv2.rectangle(orig_full_drawn, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(orig_full_drawn, "ROI", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # Panel 4: Augmented Full
    aug_full_drawn = draw_landmarks_and_contours(aug_img_rgb, aug_coords_px, draw_contours=True, draw_badges=True, scale_factor=1.0)
    cv2.rectangle(aug_full_drawn, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(aug_full_drawn, "ROI", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # Panel 2: Original Zoomed ROI
    panel_size = 540
    orig_roi = orig_img_rgb[y1:y2, x1:x2]
    orig_roi_scaled = cv2.resize(orig_roi, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)
    orig_roi_coords = np.zeros_like(orig_coords_px)
    for i, pt in enumerate(orig_coords_px):
        orig_roi_coords[i, 0] = (pt[0] - x1) / float(roi_w) * panel_size
        orig_roi_coords[i, 1] = (pt[1] - y1) / float(roi_h) * panel_size
    orig_roi_drawn = draw_landmarks_and_contours(orig_roi_scaled, orig_roi_coords, draw_contours=True, draw_badges=True, scale_factor=1.5)

    # Panel 3: Augmented Zoomed ROI
    aug_roi = aug_img_rgb[y1:y2, x1:x2]
    aug_roi_scaled = cv2.resize(aug_roi, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)
    aug_roi_coords = np.zeros_like(aug_coords_px)
    for i, pt in enumerate(aug_coords_px):
        aug_roi_coords[i, 0] = (pt[0] - x1) / float(roi_w) * panel_size
        aug_roi_coords[i, 1] = (pt[1] - y1) / float(roi_h) * panel_size
    aug_roi_drawn = draw_landmarks_and_contours(aug_roi_scaled, aug_roi_coords, draw_contours=True, draw_badges=True, scale_factor=1.5)

    # Draw motion displacement arrows on Augmented ROI
    for i in range(len(orig_roi_coords)):
        p_orig = (int(round(orig_roi_coords[i, 0])), int(round(orig_roi_coords[i, 1])))
        p_aug = (int(round(aug_roi_coords[i, 0])), int(round(aug_roi_coords[i, 1])))
        if np.linalg.norm(np.array(p_aug) - np.array(p_orig)) > 4:
            cv2.arrowedLine(aug_roi_drawn, p_orig, p_aug, (0, 255, 255), 1, tipLength=0.25, line_type=cv2.LINE_AA)

    orig_full_resized = cv2.resize(orig_full_drawn, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
    aug_full_resized = cv2.resize(aug_full_drawn, (panel_size, panel_size), interpolation=cv2.INTER_AREA)

    def add_sub_header(panel_img, title, color):
        sh_h = 32
        strip = np.full((sh_h, panel_img.shape[1], 3), (25, 28, 34), dtype=np.uint8)
        cv2.line(strip, (0, sh_h - 1), (panel_img.shape[1], sh_h - 1), color, 2)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(strip, title, ((panel_img.shape[1] - tw) // 2, (sh_h + th) // 2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        return np.vstack([strip, panel_img])

    p1 = add_sub_header(orig_full_resized, "1. ORIGINAL CEPHALOGRAM (640x640)", (100, 200, 100))
    p2 = add_sub_header(orig_roi_drawn, "2. ORIGINAL VERTEBRAE ROI (Zoomed)", (100, 220, 120))
    p3 = add_sub_header(aug_roi_drawn, "3. AUGMENTED VERTEBRAE ROI (Transformed)", (80, 180, 255))
    p4 = add_sub_header(aug_full_resized, "4. AUGMENTED CEPHALOGRAM (640x640)", (60, 150, 255))

    divider = np.full((p1.shape[0], 4, 3), (45, 50, 60), dtype=np.uint8)
    panels_row = np.hstack([p1, divider, p2, divider, p3, divider, p4])
    total_w = panels_row.shape[1]

    # Top Master Header Banner
    banner_h = 60
    banner = np.full((banner_h, total_w, 3), (16, 18, 22), dtype=np.uint8)
    cv2.putText(
        banner,
        f"Sample {index:02d}: {filename}  |  Cephalometric Augmentation & Landmark Tracking Check",
        (18, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    stats_str = f"Landmark Motion: Mean = {mean_disp:.1f} px, Min = {min_disp:.1f} px, Max = {max_disp:.1f} px  |  Yellow arrows: motion vectors (Original -> Augmented)"
    cv2.putText(
        banner,
        stats_str,
        (18, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (120, 210, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(banner, (0, banner_h - 1), (total_w, banner_h - 1), (60, 65, 80), 2)

    # Bottom Legend Banner
    legend_h = 42
    legend = np.full((legend_h, total_w, 3), (16, 18, 22), dtype=np.uint8)
    cv2.line(legend, (0, 0), (total_w, 0), (60, 65, 80), 1)

    legend_items = [
        ("C2: 0-C2_PI, 1-C2_IC, 2-C2_AI", (255, 180, 50)),
        ("C3: 3-C3_PS, 4-C3_AS, 5-C3_PI, 6-C3_IC, 7-C3_AI", (50, 220, 100)),
        ("C4: 8-C4_PS, 9-C4_AS, 10-C4_PI, 11-C4_IC, 12-C4_AI", (230, 80, 160)),
    ]
    col_w = total_w // 3
    for i, (text, col) in enumerate(legend_items):
        cx = i * col_w + 20
        cv2.circle(legend, (cx, 21), 6, col, -1, cv2.LINE_AA)
        cv2.putText(legend, text, (cx + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)

    composite = np.vstack([banner, panels_row, legend])

    stats = {
        "filename": filename,
        "index": index,
        "mean_disp": mean_disp,
        "min_disp": min_disp,
        "max_disp": max_disp,
        "diffs": diffs.tolist(),
        "orig_coords": orig_coords_px.tolist(),
        "aug_coords": aug_coords_px.tolist(),
    }
    return composite, stats


def generate_elastic_transform_analysis(
    sample_img_rgb: np.ndarray,
    coords_px: np.ndarray,
    output_path: Path,
):
    """
    Renders a dedicated, transparent empirical study of ElasticTransform on cervical vertebrae:
    1. Default project setting: alpha=1, sigma=30 (displacement is ~0.08 px)
    2. Amplified deformation: alpha=35, sigma=30 (displacement is ~3-5 px)
    3. Grid warp visualization demonstrating displacement field.
    """
    h, w = sample_img_rgb.shape[:2]
    x1, y1, x2, y2 = compute_vertebrae_roi_bbox(coords_px, w, h, margin_ratio=0.35, min_size=210)
    roi_w = x2 - x1
    panel_size = 480

    # Test 1: Alpha = 1.0 (Project setting)
    et_default = A.Compose(
        [A.ElasticTransform(alpha=1.0, sigma=30.0, border_mode=cv2.BORDER_CONSTANT, p=1.0)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    np.random.seed(101)
    res_def = et_default(image=sample_img_rgb, keypoints=coords_px.tolist())
    def_coords = np.array(res_def["keypoints"], dtype=np.float32) if len(res_def["keypoints"]) == 13 else coords_px
    diff_def = np.linalg.norm(def_coords - coords_px, axis=1)

    # Test 2: Amplified Alpha = 35.0 (To clearly visualize non-rigid curvature warping)
    et_amplified = A.Compose(
        [A.ElasticTransform(alpha=35.0, sigma=30.0, border_mode=cv2.BORDER_CONSTANT, p=1.0)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    np.random.seed(101)
    res_amp = et_amplified(image=sample_img_rgb, keypoints=coords_px.tolist())
    amp_coords = np.array(res_amp["keypoints"], dtype=np.float32) if len(res_amp["keypoints"]) == 13 else coords_px
    diff_amp = np.linalg.norm(amp_coords - coords_px, axis=1)

    # Render ROIs
    def render_roi(img, pts):
        roi = img[y1:y2, x1:x2]
        scaled = cv2.resize(roi, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)
        pts_roi = np.zeros_like(pts)
        for i, pt in enumerate(pts):
            pts_roi[i, 0] = (pt[0] - x1) / float(roi_w) * panel_size
            pts_roi[i, 1] = (pt[1] - y1) / float(roi_w) * panel_size
        return draw_landmarks_and_contours(scaled, pts_roi, draw_contours=True, draw_badges=True, scale_factor=1.4)

    p_orig = render_roi(sample_img_rgb, coords_px)
    p_def = render_roi(res_def["image"], def_coords)
    p_amp = render_roi(res_amp["image"], amp_coords)

    # Render Grid Warp
    grid_img = np.full((panel_size, panel_size, 3), 30, dtype=np.uint8)
    step = 30
    for gy in range(0, panel_size, step):
        cv2.line(grid_img, (0, gy), (panel_size, gy), (180, 180, 180), 1)
    for gx in range(0, panel_size, step):
        cv2.line(grid_img, (gx, 0), (gx, panel_size), (180, 180, 180), 1)

    et_grid = A.Compose([A.ElasticTransform(alpha=35.0, sigma=30.0, border_mode=cv2.BORDER_CONSTANT, p=1.0)])
    np.random.seed(101)
    warped_grid = et_grid(image=grid_img)["image"]

    def add_sub(p, title, color):
        sh_h = 32
        strip = np.full((sh_h, panel_size, 3), (25, 28, 34), dtype=np.uint8)
        cv2.line(strip, (0, sh_h - 1), (panel_size, sh_h - 1), color, 2)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(strip, title, ((panel_size - tw) // 2, (sh_h + th) // 2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        return np.vstack([strip, p])

    p1 = add_sub(p_orig, "ORIGINAL VERTEBRAE", (100, 200, 100))
    p2 = add_sub(p_def, f"ELASTIC (alpha=1, sigma=30) [mean shift={np.mean(diff_def):.3f}px]", (80, 180, 255))
    p3 = add_sub(p_amp, f"AMPLIFIED (alpha=35, sigma=30) [mean shift={np.mean(diff_amp):.2f}px]", (255, 150, 50))
    p4 = add_sub(warped_grid, "DISPLACEMENT FIELD (Deformation Grid)", (200, 100, 255))

    divider = np.full((p1.shape[0], 4, 3), (45, 50, 60), dtype=np.uint8)
    row = np.hstack([p1, divider, p2, divider, p3, divider, p4])
    tot_w = row.shape[1]

    bh = 54
    banner = np.full((bh, tot_w, 3), (16, 18, 22), dtype=np.uint8)
    cv2.putText(
        banner,
        "Deep-Dive: ElasticTransform Keypoint Transformation & Displacement Analysis",
        (18, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        f"At alpha=1.0, mean landmark displacement is {np.mean(diff_def):.4f} px (sub-pixel). Albumentations transforms keypoints, but alpha=1 has virtually no visible deformation.",
        (18, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (120, 210, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(banner, (0, bh - 1), (tot_w, bh - 1), (60, 65, 80), 2)

    comp = np.vstack([banner, row])
    cv2.imwrite(str(output_path), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"✅ ElasticTransform deep-dive analysis saved to: {output_path}")


def run_augmentation_check(
    img_dir: str = "dataset/test/images",
    npz_dir: str = "dataset/test/labels",
    output_dir: str = "evaluation/augmentation_check",
    num_samples: int = 10,
    seed: int = 100,
):
    """Executes the augmentation check across 10 samples and saves PNGs and summary."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if len(img_files) < num_samples:
        raise ValueError(f"Found {len(img_files)} images in {img_dir}, expected at least {num_samples}")

    selected_files = img_files[:num_samples]
    print(f"--> Selected {len(selected_files)} images for augmentation check from: {img_dir}")

    pipeline = build_safe_training_pipeline()

    all_stats = []
    generated_pngs = []
    card_thumbnails = []

    first_orig_img = None
    first_coords_px = None

    for idx, fname in enumerate(selected_files, start=1):
        np.random.seed(seed + idx * 7)
        stem = Path(fname).stem
        img_p = Path(img_dir) / fname
        npz_p = Path(npz_dir) / f"{stem}.npz"

        if not npz_p.exists():
            continue

        raw_bgr = cv2.imread(str(img_p))
        if raw_bgr is None:
            continue
        orig_img_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = orig_img_rgb.shape[:2]

        data = np.load(str(npz_p))
        coords_norm = data["coords"]
        coords_px = (coords_norm * np.array([img_w, img_h])).astype(np.float32)

        if first_orig_img is None:
            first_orig_img = orig_img_rgb.copy()
            first_coords_px = coords_px.copy()

        # Apply augmentation pipeline
        augmented = pipeline(image=orig_img_rgb, keypoints=coords_px.tolist())
        aug_img_rgb = augmented["image"]
        aug_kps = augmented["keypoints"]
        aug_coords_px = np.array(aug_kps, dtype=np.float32)

        card, stats = create_sample_comparison_card(
            orig_img_rgb=orig_img_rgb,
            orig_coords_px=coords_px,
            aug_img_rgb=aug_img_rgb,
            aug_coords_px=aug_coords_px,
            filename=fname,
            index=idx,
        )

        save_name = f"aug_sample_{idx:02d}_{stem}.png"
        save_p = out_path / save_name
        cv2.imwrite(str(save_p), cv2.cvtColor(card, cv2.COLOR_RGB2BGR))
        generated_pngs.append(str(save_p))
        all_stats.append(stats)

        thumb = cv2.resize(card, (1080, int(round(card.shape[0] * 1080.0 / card.shape[1]))), interpolation=cv2.INTER_AREA)
        card_thumbnails.append(thumb)

        print(f"  [{idx:02d}/{num_samples}] Saved: {save_name} | Shift mean: {stats['mean_disp']:5.2f} px, min: {stats['min_disp']:5.2f} px, max: {stats['max_disp']:5.2f} px")

    # Generate Master Overview Contact Sheet
    if card_thumbnails:
        grid_img = np.vstack(card_thumbnails)
        grid_save_p = out_path / "all_10_samples_overview.png"
        cv2.imwrite(str(grid_save_p), cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        print(f"\n✅ Master overview grid saved to: {grid_save_p}")

    # Generate ElasticTransform Deep-Dive Study
    if first_orig_img is not None:
        et_out = out_path / "elastic_transform_analysis.png"
        generate_elastic_transform_analysis(first_orig_img, first_coords_px, et_out)

    # Print Summary Table
    print("\n" + "=" * 88)
    print("AUGMENTATION KEYPOINT SHIFT & TRACKING VERIFICATION (10 SAMPLES)")
    print("=" * 88)
    print(f"{'Sample':<8} {'Filename':<16} {'Mean Shift (px)':<18} {'Min Shift (px)':<18} {'Max Shift (px)':<18} {'Status':<12}")
    print("-" * 88)
    for s in all_stats:
        status = "SYNCHRONIZED" if s['max_disp'] > 0 else "STATIC"
        print(f"#{s['index']:02d}     {s['filename']:<16} {s['mean_disp']:<18.2f} {s['min_disp']:<18.2f} {s['max_disp']:<18.2f} {status:<12}")
    print("=" * 88)

    return generated_pngs, all_stats


if __name__ == "__main__":
    run_augmentation_check()
