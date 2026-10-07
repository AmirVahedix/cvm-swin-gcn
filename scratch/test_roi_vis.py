import os
import cv2
import numpy as np
import torch
from pathlib import Path

from src.eval import load_model, LANDMARK_COLORS_BGR
from src.data import LANDMARK_CLASSES, get_test_dataloader

VERTEBRA_CONTOURS = [
    ("C2", [0, 1, 2], False),
    ("C3", [3, 4, 7, 6, 5], True),
    ("C4", [8, 9, 12, 11, 10], True),
]


def compute_roi_bbox(
    gt_coords: np.ndarray | None,
    pred_coords: np.ndarray | None,
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.35,
    min_size: int = 180,
) -> tuple[int, int, int, int]:
    """
    Computes a square bounding box centered around valid landmarks with generous margin.
    """
    valid_pts = []
    if gt_coords is not None:
        for pt in gt_coords:
            if pt[0] >= 0 and pt[1] >= 0 and not np.isnan(pt[0]) and not np.isnan(pt[1]):
                valid_pts.append(pt)
    if pred_coords is not None:
        for pt in pred_coords:
            if pt[0] >= 0 and pt[1] >= 0 and not np.isnan(pt[0]) and not np.isnan(pt[1]):
                valid_pts.append(pt)

    if not valid_pts:
        half = min(img_w, img_h) // 4
        cx, cy = img_w // 2, img_h // 2
        return max(0, cx - half), max(0, cy - half), min(img_w, cx + half), min(img_h, cy + half)

    pts = np.array(valid_pts)
    xs = pts[:, 0] * img_w
    ys = pts[:, 1] * img_h

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    box_w = max_x - min_x
    box_h = max_y - min_y
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    dim = max(box_w, box_h) * (1.0 + 2.0 * margin_ratio)
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


def render_panel(
    roi_rgb: np.ndarray,
    coords: np.ndarray,
    img_w: int,
    img_h: int,
    x1: int,
    y1: int,
    roi_w: int,
    roi_h: int,
    panel_size: int = 560,
    title: str = "Ground Truth",
    title_color: tuple[int, int, int] = (40, 180, 80),
) -> np.ndarray:
    """
    Renders one panel (Ground Truth or Prediction) at high resolution.
    """
    panel = cv2.resize(roi_rgb, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)

    # Convert coordinates to panel pixel space
    pts_dict = {}
    for i, pt in enumerate(coords):
        if pt[0] < 0 or pt[1] < 0 or np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        px_full = pt[0] * img_w
        py_full = pt[1] * img_h
        px_p = int(round((px_full - x1) / float(roi_w) * panel_size))
        py_p = int(round((py_full - y1) / float(roi_h) * panel_size))
        pts_dict[i] = (px_p, py_p)

    # 1. Draw anatomical contour lines
    overlay = panel.copy()
    contour_color = (255, 255, 255)
    for v_name, indices, is_closed in VERTEBRA_CONTOURS:
        chain = [pts_dict[idx] for idx in indices if idx in pts_dict]
        if len(chain) >= 2:
            pts_arr = np.array(chain, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts_arr], isClosed=is_closed, color=contour_color, thickness=1, lineType=cv2.LINE_AA)

    # Blend contour lines with 60% opacity
    panel = cv2.addWeighted(overlay, 0.45, panel, 0.55, 0)

    # 2. Draw landmark points and number badges
    for i, pt in enumerate(coords):
        if i not in pts_dict:
            continue
        px, py = pts_dict[i]
        color_bgr = LANDMARK_COLORS_BGR[i % len(LANDMARK_COLORS_BGR)]
        # LANDMARK_COLORS_BGR is BGR, panel is RGB:
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        # Outer black ring
        cv2.circle(panel, (px, py), 7, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        # Colored circle
        cv2.circle(panel, (px, py), 5, color_rgb, -1, lineType=cv2.LINE_AA)
        # Inner white center dot
        cv2.circle(panel, (px, py), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Draw number badge
        badge_text = f"{i:02d}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)

        # Offset text slightly
        # Put text to the left or right depending on landmark position
        tx = px + 9 if px < panel_size - 40 else px - tw - 11
        ty = py - 4 if py > 20 else py + th + 4

        # Pill background for number
        cv2.rectangle(panel, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (15, 15, 15), -1)
        cv2.rectangle(panel, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), color_rgb, 1)
        cv2.putText(panel, badge_text, (tx, ty), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    # 3. Add panel title banner at top
    banner_h = 36
    banner = np.zeros((banner_h, panel_size, 3), dtype=np.uint8)
    banner[:] = (25, 28, 32)
    # Bottom border
    cv2.line(banner, (0, banner_h - 1), (panel_size, banner_h - 1), title_color, 2)
    
    # Title text
    (ttw, tth), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(
        banner,
        title,
        ((panel_size - ttw) // 2, (banner_h + tth) // 2 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        title_color,
        2,
        lineType=cv2.LINE_AA,
    )

    panel_with_header = np.vstack([banner, panel])
    return panel_with_header


def create_side_by_side_roi_image(
    image_rgb: np.ndarray,
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    sample_name: str = "",
    pixel_spacing: float = 0.1,
    panel_size: int = 560,
) -> np.ndarray:
    """
    Creates a publication-quality side-by-side ROI comparison image.
    Left: Ground Truth | Right: Prediction
    Includes:
    - Top header with sample name and error metrics
    - Cropped high-resolution vertebrae ROI
    - Clean bottom legend with all landmarks
    """
    img_h, img_w = image_rgb.shape[:2]

    # Compute ROI bounding box
    x1, y1, x2, y2 = compute_roi_bbox(gt_coords, pred_coords, img_w, img_h, margin_ratio=0.35, min_size=180)
    roi_rgb = image_rgb[y1:y2, x1:x2]
    roi_w = x2 - x1
    roi_h = y2 - y1

    # Compute sample-level error
    valid_mask = (gt_coords[:, 0] >= 0) & (gt_coords[:, 1] >= 0)
    if np.any(valid_mask):
        gt_px = gt_coords[valid_mask] * np.array([img_w, img_h])
        pred_px = pred_coords[valid_mask] * np.array([img_w, img_h])
        radial_errors_px = np.sqrt(np.sum((gt_px - pred_px) ** 2, axis=1))
        mre_px = float(np.mean(radial_errors_px))
        mre_mm = mre_px * pixel_spacing
        metric_str = f"Mean Radial Error: {mre_px:.2f} px ({mre_mm:.2f} mm)"
    else:
        metric_str = "Mean Radial Error: N/A"

    # Render Left Panel (Ground Truth)
    panel_gt = render_panel(
        roi_rgb=roi_rgb,
        coords=gt_coords,
        img_w=img_w,
        img_h=img_h,
        x1=x1,
        y1=y1,
        roi_w=roi_w,
        roi_h=roi_h,
        panel_size=panel_size,
        title="GROUND TRUTH",
        title_color=(76, 175, 80),  # Material Green
    )

    # Render Right Panel (Prediction)
    panel_pred = render_panel(
        roi_rgb=roi_rgb,
        coords=pred_coords,
        img_w=img_w,
        img_h=img_h,
        x1=x1,
        y1=y1,
        roi_w=roi_w,
        roi_h=roi_h,
        panel_size=panel_size,
        title="MODEL PREDICTION",
        title_color=(33, 150, 243),  # Material Blue
    )

    # Divider bar between panels
    divider_w = 6
    divider = np.full((panel_gt.shape[0], divider_w, 3), 40, dtype=np.uint8)

    # Combine side by side
    panels_combined = np.hstack([panel_gt, divider, panel_pred])
    total_w = panels_combined.shape[1]

    # Top Header
    header_h = 44
    header = np.zeros((header_h, total_w, 3), dtype=np.uint8)
    header[:] = (18, 20, 24)

    header_left = f"Sample: {sample_name}" if sample_name else "CVM Vertebrae Evaluation"
    cv2.putText(header, header_left, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, lineType=cv2.LINE_AA)
    
    (mw, _), _ = cv2.getTextSize(metric_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(header, metric_str, (total_w - mw - 16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 200, 100), 1, lineType=cv2.LINE_AA)
    cv2.line(header, (0, header_h - 1), (total_w, header_h - 1), (50, 55, 65), 1)

    # Bottom Legend
    # 13 landmarks split into 3 rows or columns: C2 (3), C3 (5), C4 (5)
    legend_h = 80
    legend = np.zeros((legend_h, total_w, 3), dtype=np.uint8)
    legend[:] = (18, 20, 24)
    cv2.line(legend, (0, 0), (total_w, 0), (50, 55, 65), 1)

    # Groups: C2: 0..2, C3: 3..7, C4: 8..12
    groups = [
        ("C2", [0, 1, 2]),
        ("C3", [3, 4, 5, 6, 7]),
        ("C4", [8, 9, 10, 11, 12]),
    ]

    col_w = total_w // 3
    for col_idx, (grp_name, indices) in enumerate(groups):
        start_x = col_idx * col_w + 16
        # Group title
        cv2.putText(legend, f"[{grp_name}]", (start_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, lineType=cv2.LINE_AA)
        
        # Landmarks in this group
        for row_i, idx in enumerate(indices):
            color_bgr = LANDMARK_COLORS_BGR[idx % len(LANDMARK_COLORS_BGR)]
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            name = LANDMARK_CLASSES[idx]
            
            # Position: 2 mini-columns inside C3 and C4 if needed, or row placement
            if len(indices) <= 3:
                lx = start_x + 60 + (row_i * 90)
                ly = 20
            else:
                if row_i < 3:
                    lx = start_x + 50 + (row_i * 90)
                    ly = 20
                else:
                    lx = start_x + 50 + ((row_i - 3) * 90)
                    ly = 50

            cv2.circle(legend, (lx, ly - 4), 5, color_rgb, -1, lineType=cv2.LINE_AA)
            cv2.circle(legend, (lx, ly - 4), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(legend, f"{idx:02d}:{name.split('_')[-1]}", (lx + 9, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1, lineType=cv2.LINE_AA)

    final_image = np.vstack([header, panels_combined, legend])
    return final_image


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model("artifacts/best.pth", device=device, img_size=640)
    loader = get_test_dataloader(
        test_img_dir="dataset/test/images",
        test_npz_dir="dataset/test/labels",
        batch_size=1,
        img_size=640
    )
    batch = next(iter(loader))
    images = batch["image"].to(device)
    with torch.no_grad():
        _, pred_coords = model(images)

    fname = batch["filename"][0]
    gt_coords = batch["coords"][0].cpu().numpy()
    pred_coords = pred_coords[0].cpu().numpy()

    orig_img_path = Path("dataset/test/images") / fname
    raw_bgr = cv2.imread(str(orig_img_path))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    spacing = float(batch["pixel_spacing"][0]) if "pixel_spacing" in batch else 0.1

    vis = create_side_by_side_roi_image(
        image_rgb=raw_rgb,
        gt_coords=gt_coords,
        pred_coords=pred_coords,
        sample_name=fname,
        pixel_spacing=spacing,
        panel_size=560,
    )

    out_path = Path("scratch/test_roi_0515.png")
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Generated test ROI image: {out_path} ({vis.shape})")

if __name__ == "__main__":
    main()
