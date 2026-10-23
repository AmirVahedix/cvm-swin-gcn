#!/usr/bin/env python3
"""
Stage 2 ROI Model Evaluation Suite on Ground-Truth Cervical Vertebrae ROIs (Oracle Evaluation).

Evaluates the high-resolution CephalometricSwinGCN refiner strictly on ground-truth cervical
vertebrae ROIs (C2-C4) with asymmetric bottom margin, subpixel parabolic peak extraction,
and exact inverse half-pixel re-projection to native radiograph pixel space.

Produces:
  - Full comprehensive clinical & statistical metric report (summary_report.txt)
  - Raw per-landmark evaluation records (predictions_landmarks.csv)
  - Serialized metric payload (metrics.json)
  - High-resolution Ground Truth vs Stage 2 side-by-side visualization panels
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow

from src.data.constants import (
    LANDMARK_CLASSES,
    NUM_LANDMARKS,
)
from src.data.loaders.dataset import get_image_dimensions_map
from src.inference import resolve_device
from src.models.model import CephalometricSwinGCN
from src.utils.cvm_calculator import evaluate_landmarks_dict
from src.utils.metrics import (
    C3_AI_IDX,
    C3_PI_IDX,
    compute_normalized_errors,
    compute_reference_length,
    compute_test_comprehensive_metrics,
    format_summary_report,
)
from src.utils.roi_utils import (
    ROIBoundingBox,
    canvas_to_native_coords,
    compute_roi_bbox,
    crop_roi_with_padding,
    extract_subpixel_peaks,
    inverse_transform_coords,
    map_bbox_to_native,
    transform_coords_to_roi,
)

# 13 Distinct High-Contrast BGR Colors for 13 CVM Landmarks
LANDMARK_COLORS_BGR = [
    (0, 0, 255),      # 0: C2_PI - Red
    (0, 140, 255),    # 1: C2_IC - Orange
    (0, 215, 255),    # 2: C2_AI - Yellow
    (0, 255, 128),    # 3: C3_PS - Spring Green
    (0, 255, 0),      # 4: C3_AS - Green
    (128, 255, 0),    # 5: C3_PI - Lime
    (255, 255, 0),    # 6: C3_IC - Cyan
    (255, 191, 0),    # 7: C3_AI - Azure
    (255, 0, 0),      # 8: C4_PS - Blue
    (255, 0, 255),    # 9: C4_AS - Magenta
    (255, 0, 128),    # 10: C4_PI - Rose
    (128, 128, 255),  # 11: C4_IC - Light Coral
    (0, 255, 255),    # 12: C4_AI - Mint
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_normalized_tensor(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalizes an RGB uint8 image [H, W, 3] using ImageNet statistics into a float tensor [1, 3, H, W]."""
    norm = ((img_rgb.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)


def load_stage2_model(weights_path: str, img_size: int, device: torch.device) -> CephalometricSwinGCN:
    """Loads a single CephalometricSwinGCN checkpoint for Stage 2 ROI refinement."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    model = CephalometricSwinGCN(num_landmarks=NUM_LANDMARKS, pretrained=False, img_size=img_size)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("stage2."):
            k = k[len("stage2."):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def render_roi_panel_with_contours(
    roi_rgb: np.ndarray,
    coords_norm_roi: np.ndarray,
    panel_size: int = 560,
    title: str = "PREDICTION",
    title_color: Tuple[int, int, int] = (33, 150, 243),
) -> np.ndarray:
    """Renders a single high-resolution ROI panel with vertebra landmark dots, contours, and title banner."""
    panel = cv2.resize(roi_rgb, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)

    pts = {}
    for i, pt in enumerate(coords_norm_roi):
        if pt[0] < 0 or pt[1] < 0 or np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        px = int(np.clip(round(pt[0] * panel_size), 0, panel_size - 1))
        py = int(np.clip(round(pt[1] * panel_size), 0, panel_size - 1))
        pts[i] = (px, py)

    c2_conns = [(0, 1), (1, 2)]
    c3_conns = [(3, 4), (4, 7), (7, 6), (6, 5), (5, 3)]
    c4_conns = [(8, 9), (9, 12), (12, 11), (11, 10), (10, 8)]

    for u, v in c2_conns + c3_conns + c4_conns:
        if u in pts and v in pts:
            cv2.line(panel, pts[u], pts[v], (180, 180, 180), 1, lineType=cv2.LINE_AA)

    for i in sorted(pts.keys()):
        px, py = pts[i]
        color_bgr = LANDMARK_COLORS_BGR[i % len(LANDMARK_COLORS_BGR)]
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        cv2.circle(panel, (px, py), 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (px, py), 5, color_rgb, -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (px, py), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    banner_h = 36
    banner = np.zeros((banner_h, panel_size, 3), dtype=np.uint8)
    banner[:] = (25, 28, 32)
    cv2.line(banner, (0, banner_h - 1), (panel_size, banner_h - 1), title_color, 2)
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
    return np.vstack([banner, panel])


def create_two_way_comparison_panel(
    roi_rgb: np.ndarray,
    gt_coords_roi: np.ndarray,
    pred_coords_roi: np.ndarray,
    sample_name: str,
    norm_err: float,
    panel_size: int = 560,
) -> np.ndarray:
    """Renders a 2-panel comparison: Ground Truth vs Stage 2 Fine ROI Prediction."""
    p_gt = render_roi_panel_with_contours(
        roi_rgb, gt_coords_roi, panel_size=panel_size, title="GROUND TRUTH", title_color=(76, 175, 80)
    )
    p_pred = render_roi_panel_with_contours(
        roi_rgb, pred_coords_roi, panel_size=panel_size, title=f"STAGE 2 SWIN ({norm_err:.2f}% L_ref)", title_color=(33, 150, 243)
    )

    divider = np.full((p_gt.shape[0], 6, 3), 40, dtype=np.uint8)
    combined = np.hstack([p_gt, divider, p_pred])

    header_h = 44
    header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
    header[:] = (18, 20, 24)

    cv2.putText(
        header,
        f"Sample: {sample_name} | Mean Normalized Error: {norm_err:.2f}% L_ref",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        1,
        lineType=cv2.LINE_AA,
    )

    return np.vstack([header, combined])


def run_stage2_roi_evaluation(
    stage2_weights: str = "artifacts/stage2_best.pth",
    img_dir: str = "dataset/test/images",
    npz_dir: str = "dataset/test/labels",
    output_dir: str = "evaluation/stage2_roi",
    split_name: str = "test",
    run_name: Optional[str] = None,
    stage2_img_size: int = 448,
    canvas_size: int = 640,
    margin_ratio: float = 0.18,
    bottom_margin_ratio: Optional[float] = 0.28,
    num_visualizations: int = 10,
    device_str: Optional[str] = None,
    native_img_dir: Optional[str] = "data/raw/images",
    log_to_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    tracking_username: Optional[str] = None,
    tracking_password: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Evaluates Stage 2 CephalometricSwinGCN strictly on Ground-Truth calculated ROIs across a dataset split.
    """
    load_dotenv()
    device = resolve_device(device_str) if device_str else resolve_device("auto")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_run_name = run_name or f"stage2_gt_roi_{split_name}_{timestamp}"

    run_dir = Path(output_dir) / actual_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"STARTING STAGE 2 GT-ROI EVALUATION ({split_name.upper()} SPLIT) | Device: {device}")
    print(f"Stage 2 Weights: '{stage2_weights}' ({stage2_img_size}x{stage2_img_size})")
    print(f"ROI Margin Ratio: {margin_ratio:.2f} | Bottom Margin: {bottom_margin_ratio if bottom_margin_ratio is not None else margin_ratio:.2f}")
    print(f"Landmark Extraction: Sub-Pixel Heatmap Peaks (extract_subpixel_peaks)")
    print(f"Image Directory: '{img_dir}' | Labels Directory: '{npz_dir}'")
    print(f"Native Radiograph Directory: '{native_img_dir}'")
    print("=" * 88)

    model = load_stage2_model(stage2_weights, stage2_img_size, device)

    img_files = sorted([
        f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    if not img_files:
        raise FileNotFoundError(f"No valid image files found in '{img_dir}'")

    dims_map = get_image_dimensions_map()

    all_filenames = []
    all_orig_dims = []
    gt_native_list = []
    pred_native_list = []
    gt_stages = []
    pred_stages = []
    gt_morph_list = []
    pred_morph_list = []
    latencies_ms = []
    vis_saved = 0

    print(f"\n--> Running Stage 2 GT-ROI inference on {len(img_files)} evaluation samples...")

    for fname in tqdm(img_files, desc=f"Evaluating Stage 2 ({split_name})"):
        stem = Path(fname).stem
        npz_path = Path(npz_dir) / f"{stem}.npz"
        if not npz_path.exists():
            continue

        label_data = np.load(npz_path)
        gt_canvas_norm = label_data["coords"].astype(np.float32)  # [13, 2] in [0, 1]

        orig_w = dims_map.get(fname, {}).get("orig_w") or dims_map.get(f"{stem}.jpg", {}).get("orig_w") or dims_map.get(f"{stem}.png", {}).get("orig_w")
        orig_h = dims_map.get(fname, {}).get("orig_h") or dims_map.get(f"{stem}.jpg", {}).get("orig_h") or dims_map.get(f"{stem}.png", {}).get("orig_h")

        img_path = Path(img_dir) / fname
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue
        if orig_w is None or orig_h is None:
            orig_h, orig_w = raw_bgr.shape[:2]

        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        orig_w = int(orig_w)
        orig_h = int(orig_h)

        # Check native image presence for true uncompressed high-resolution cropping
        native_rgb = None
        if native_img_dir and os.path.exists(native_img_dir):
            for ext in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
                cand = Path(native_img_dir) / f"{stem}{ext}"
                if cand.exists():
                    nbgr = cv2.imread(str(cand))
                    if nbgr is not None:
                        native_rgb = cv2.cvtColor(nbgr, cv2.COLOR_BGR2RGB)
                    break

        canvas_img = cv2.resize(raw_rgb, (canvas_size, canvas_size))

        # Compute Ground-Truth ROI bounding box with bottom headroom for C4
        gt_bbox = compute_roi_bbox(
            coords=gt_canvas_norm,
            img_w=canvas_size,
            img_h=canvas_size,
            margin_ratio=margin_ratio,
            bottom_margin_ratio=bottom_margin_ratio,
        )

        # Crop patch from native high-res image if available, otherwise from canvas
        if native_rgb is not None and max(orig_w, orig_h) > canvas_size:
            native_b = map_bbox_to_native(gt_bbox, orig_w=orig_w, orig_h=orig_h, canvas_size=canvas_size)
            roi_patch, _ = crop_roi_with_padding(
                image=native_rgb,
                bbox=native_b,
                target_size=stage2_img_size,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            roi_patch, _ = crop_roi_with_padding(
                image=canvas_img,
                bbox=gt_bbox,
                target_size=stage2_img_size,
                interpolation=cv2.INTER_LINEAR,
            )

        tensor_in = to_normalized_tensor(roi_patch, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            pred_hm, _ = model(tensor_in)
            pred_coords_roi_t = extract_subpixel_peaks(pred_hm)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        pred_coords_roi = pred_coords_roi_t.squeeze(0).cpu().numpy()

        # Inversely reproject predicted coordinates back to full [0, 1] canvas space
        pred_coords_canvas = inverse_transform_coords(
            roi_coords=pred_coords_roi,
            bbox=gt_bbox,
            target_size=stage2_img_size,
            output_normalized=True,
        )

        # Inversely reproject to native image resolution (pixels)
        pred_coords_native = canvas_to_native_coords(
            pred_coords_canvas, orig_w=orig_w, orig_h=orig_h, canvas_size=canvas_size
        )
        gt_coords_native = canvas_to_native_coords(
            gt_canvas_norm, orig_w=orig_w, orig_h=orig_h, canvas_size=canvas_size
        )

        # Morphometrics and Staging
        pred_dict = {LANDMARK_CLASSES[i]: pred_coords_native[i].tolist() for i in range(NUM_LANDMARKS)}
        gt_dict = {LANDMARK_CLASSES[i]: gt_coords_native[i].tolist() for i in range(NUM_LANDMARKS)}

        pred_cvm = evaluate_landmarks_dict(pred_dict)
        gt_cvm = evaluate_landmarks_dict(gt_dict)

        def _parse_stage(cvm_res: Dict[str, Any]) -> int:
            st = cvm_res.get("stage_number") or cvm_res.get("stage")
            if st is None:
                return -1
            s = str(st).replace("CS", "").strip()
            try:
                iv = int(s)
                return iv if 1 <= iv <= 6 else -1
            except (ValueError, TypeError):
                return -1

        all_filenames.append(fname)
        all_orig_dims.append({"orig_w": orig_w, "orig_h": orig_h})
        gt_native_list.append(gt_coords_native)
        pred_native_list.append(pred_coords_native)
        gt_stages.append(_parse_stage(gt_cvm))
        pred_stages.append(_parse_stage(pred_cvm))
        gt_morph_list.append(gt_cvm)
        pred_morph_list.append(pred_cvm)

        # Visualization
        if vis_saved < num_visualizations:
            gt_coords_roi = transform_coords_to_roi(
                coords=gt_canvas_norm,
                bbox=gt_bbox,
                target_size=stage2_img_size,
                is_normalized_input=True,
            )
            sample_err = compute_normalized_errors(pred_coords_native, gt_coords_native)
            sample_mean_err = float(np.mean(sample_err["normalized_error_pct"]))

            vis_panel = create_two_way_comparison_panel(
                roi_rgb=roi_patch,
                gt_coords_roi=gt_coords_roi,
                pred_coords_roi=pred_coords_roi,
                sample_name=fname,
                norm_err=sample_mean_err,
                panel_size=560,
            )
            out_vis_path = vis_dir / f"vis_{vis_saved + 1:03d}_{stem}.png"
            cv2.imwrite(str(out_vis_path), cv2.cvtColor(vis_panel, cv2.COLOR_RGB2BGR))
            vis_saved += 1

    # Format morphometrics structure
    def build_morph_pack(cvm_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "concavities": {
                "C2": [c.get("percentages", {}).get("C2", 0.0) for c in cvm_list],
                "C3": [c.get("percentages", {}).get("C3", 0.0) for c in cvm_list],
                "C4": [c.get("percentages", {}).get("C4", 0.0) for c in cvm_list],
            },
            "shape_indices": {
                "C3": [c.get("shapes", {}).get("C3", {}).get("shape_index", c.get("shapes", {}).get("C3", {}).get("ratio", 1.0)) for c in cvm_list],
                "C4": [c.get("shapes", {}).get("C4", {}).get("shape_index", c.get("shapes", {}).get("C4", {}).get("ratio", 1.0)) for c in cvm_list],
            },
            "notches": {
                "C2": [c.get("notches", {}).get("C2", False) for c in cvm_list],
                "C3": [c.get("notches", {}).get("C3", False) for c in cvm_list],
                "C4": [c.get("notches", {}).get("C4", False) for c in cvm_list],
            },
            "shapes": {
                "C3": [c.get("shapes", {}).get("C3", {}).get("shape", "Unknown") for c in cvm_list],
                "C4": [c.get("shapes", {}).get("C4", {}).get("shape", "Unknown") for c in cvm_list],
            },
        }

    gt_morph_pack = build_morph_pack(gt_morph_list)
    pred_morph_pack = build_morph_pack(pred_morph_list)

    gt_arr = np.array(gt_native_list)
    pred_arr = np.array(pred_native_list)

    print(f"\n--> Computing comprehensive statistical & clinical metrics for {split_name.upper()} split...")
    metrics = compute_test_comprehensive_metrics(
        pred_coords=pred_arr,
        gt_coords=gt_arr,
        sample_ids=all_filenames,
        orig_dims=all_orig_dims,
        img_size=canvas_size,
        pred_stages=pred_stages,
        gt_stages=gt_stages,
        pred_morphometrics=pred_morph_pack,
        gt_morphometrics=gt_morph_pack,
        n_bootstrap=500,
    )

    latency_summary = {
        "mean_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "median_ms": float(np.median(latencies_ms)) if latencies_ms else 0.0,
        "iqr_ms": float(np.percentile(latencies_ms, 75.0) - np.percentile(latencies_ms, 25.0)) if latencies_ms else 0.0,
        "p90_ms": float(np.percentile(latencies_ms, 90.0)) if latencies_ms else 0.0,
        "fps": float(1000.0 / np.mean(latencies_ms)) if latencies_ms and np.mean(latencies_ms) > 0 else 0.0,
        "total_images": len(latencies_ms),
    }

    report_title = f"STAGE 2 SWIN (GT-ROI ORACLE EVALUATION) — {split_name.upper()} SET ({actual_run_name})"
    report_text = format_summary_report(metrics, title=report_title)

    latency_section = (
        "\n" + "=" * 88 + "\n"
        + "                         SECTION E: INFERENCE LATENCY & PROFILING                       \n"
        + "=" * 88 + "\n"
        + f" Total Samples Evaluated    : {len(all_filenames)}\n"
        + f" Mean Inference Latency     : {latency_summary['mean_ms']:.1f} ms / image ({latency_summary['fps']:.1f} FPS)\n"
        + f" Median Latency (IQR)       : {latency_summary['median_ms']:.1f} ms ({latency_summary['iqr_ms']:.1f} ms)\n"
        + f" 90th Percentile (P90)      : {latency_summary['p90_ms']:.1f} ms\n"
        + "=" * 88 + "\n"
    )

    full_report = report_text + latency_section

    # 1. Save summary report
    report_file = run_dir / f"summary_report_{split_name}.txt" if split_name != "test" else run_dir / "summary_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(full_report)

    # 2. Save JSON metrics
    combined_payload = {
        "timestamp": timestamp,
        "run_name": actual_run_name,
        "split": split_name,
        "stage2_weights": stage2_weights,
        "stage2_img_size": stage2_img_size,
        "margin_ratio": margin_ratio,
        "bottom_margin_ratio": bottom_margin_ratio,
        "inference_latency": latency_summary,
        "metrics": metrics,
    }
    metrics_json_file = run_dir / f"metrics_{split_name}.json" if split_name != "test" else run_dir / "metrics.json"
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(combined_payload, f, indent=2)

    # 3. Save Predictions Landmarks CSV
    csv_file = run_dir / f"predictions_landmarks_{split_name}.csv" if split_name != "test" else run_dir / "predictions_landmarks.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "sample_id", "landmark_index", "landmark_name",
            "gt_x_px", "gt_y_px", "pred_x_px", "pred_y_px",
            "radial_err_px", "norm_err_pct",
        ]
        writer.writerow(header)

        for s_idx, fname in enumerate(all_filenames):
            ref_l = math.hypot(
                gt_arr[s_idx, C3_AI_IDX, 0] - gt_arr[s_idx, C3_PI_IDX, 0],
                gt_arr[s_idx, C3_AI_IDX, 1] - gt_arr[s_idx, C3_PI_IDX, 1],
            )
            for l_idx in range(NUM_LANDMARKS):
                g_x, g_y = gt_arr[s_idx, l_idx]
                if g_x < 0 or g_y < 0:
                    continue
                p_x, p_y = pred_arr[s_idx, l_idx]
                r_err = math.hypot(p_x - g_x, p_y - g_y)
                writer.writerow([
                    fname,
                    l_idx,
                    LANDMARK_CLASSES[l_idx],
                    f"{g_x:.2f}", f"{g_y:.2f}",
                    f"{p_x:.2f}", f"{p_y:.2f}",
                    f"{r_err:.2f}",
                    f"{(r_err / ref_l * 100.0):.2f}",
                ])

    print(f"\n--> Successfully saved {split_name.upper()} evaluation artifacts to '{run_dir}':")
    print(f"    • Report: {report_file.name}")
    print(f"    • Metrics JSON: {metrics_json_file.name}")
    print(f"    • Landmarks CSV: {csv_file.name}")
    print(f"    • Visualizations: {vis_saved} panels in '{vis_dir.name}'")

    if log_to_mlflow:
        try:
            user = tracking_username or os.getenv("MLFLOW_TRACKING_USERNAME")
            pwd = tracking_password or os.getenv("MLFLOW_TRACKING_PASSWORD")
            if user:
                os.environ["MLFLOW_TRACKING_USERNAME"] = user
            if pwd:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd

            uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
            if uri:
                mlflow.set_tracking_uri(uri)
            exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or "cvm-stage2-gt-roi-eval"
            mlflow.set_experiment(exp_name)

            with mlflow.start_run(run_name=f"eval_{split_name}_{actual_run_name}"):
                mlflow.log_params({
                    "stage2_weights": stage2_weights,
                    "split": split_name,
                    "stage2_img_size": stage2_img_size,
                    "margin_ratio": margin_ratio,
                    "bottom_margin_ratio": bottom_margin_ratio,
                    "samples": len(all_filenames),
                })
                gen = metrics["general"]
                stg = metrics.get("staging", {})
                mlflow.log_metrics({
                    f"{split_name}_mean_norm_err": gen["mean"],
                    f"{split_name}_median_norm_err": gen["median"],
                    f"{split_name}_share_within_2_5pct": gen["share_within_2_5pct"],
                    f"{split_name}_share_within_5pct": gen["share_within_5pct"],
                    f"{split_name}_share_within_10pct": gen["share_within_10pct"],
                    f"{split_name}_p90": gen["p90"],
                    f"{split_name}_staging_acc": stg.get("overall_accuracy", 0.0),
                    f"{split_name}_quadratic_kappa": stg.get("quadratic_weighted_kappa", 0.0),
                    f"{split_name}_icc_2_1": stg.get("icc_2_1", 0.0),
                    f"{split_name}_latency_mean_ms": latency_summary["mean_ms"],
                })
                mlflow.log_artifact(str(report_file), artifact_path=f"evaluation/{split_name}")
                mlflow.log_artifact(str(metrics_json_file), artifact_path=f"evaluation/{split_name}")
                mlflow.log_artifact(str(csv_file), artifact_path=f"evaluation/{split_name}")
                if vis_saved > 0:
                    mlflow.log_artifacts(str(vis_dir), artifact_path=f"evaluation/{split_name}/visualizations")
        except Exception as ml_err:
            print(f"⚠️ Warning: MLflow logging failed: {ml_err}")

    return metrics, str(run_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 CephalometricSwinGCN on Ground-Truth Cervical Spine ROIs")
    parser.add_argument("--weights", type=str, default="artifacts/stage2_best.pth", help="Stage 2 checkpoint path")
    parser.add_argument("--img-dir", type=str, default="dataset/test/images", help="Images directory")
    parser.add_argument("--npz-dir", type=str, default="dataset/test/labels", help="Labels (.npz) directory")
    parser.add_argument("--output-dir", type=str, default="evaluation/stage2_roi", help="Output directory")
    parser.add_argument("--split", type=str, default="test", help="Cohort split name (test, val, train)")
    parser.add_argument("--run-name", type=str, default=None, help="Evaluation run name")
    parser.add_argument("--target-size", type=int, default=448, help="Stage 2 ROI resolution (default: 448)")
    parser.add_argument("--canvas-size", type=int, default=640, help="Canvas resolution (default: 640)")
    parser.add_argument("--margin-ratio", type=float, default=0.18, help="ROI margin ratio (default: 0.18)")
    parser.add_argument("--bottom-margin-ratio", type=float, default=0.28, help="Bottom margin ratio for C4 (default: 0.28)")
    parser.add_argument("--visualize", type=int, default=10, help="Number of visualization panels to save")
    parser.add_argument("--native-img-dir", type=str, default="data/raw/images", help="Native high-resolution image directory")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--device", type=str, default=None, help="Device to use (auto, cuda, mps, cpu)")

    args = parser.parse_args()

    run_stage2_roi_evaluation(
        stage2_weights=args.weights,
        img_dir=args.img_dir,
        npz_dir=args.npz_dir,
        output_dir=args.output_dir,
        split_name=args.split,
        run_name=args.run_name,
        stage2_img_size=args.target_size,
        canvas_size=args.canvas_size,
        margin_ratio=args.margin_ratio,
        bottom_margin_ratio=args.bottom_margin_ratio,
        num_visualizations=args.visualize,
        device_str=args.device,
        native_img_dir=args.native_img_dir,
        log_to_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
