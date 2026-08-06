import os
import sys
import json
import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import LANDMARK_CLASSES, NUM_LANDMARKS, get_test_dataloader
from src.models.model import CephalometricSwinGCN


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
    (255, 0, 128),    # 9: C4_AS - Violet
    (255, 0, 255),    # 10: C4_PI - Magenta
    (128, 0, 255),    # 11: C4_IC - Purple
    (255, 255, 255),  # 12: C4_AI - White
]


def draw_legend_box(vis_img: np.ndarray, num_landmarks: int):
    """
    Draws a clean, rectangular guide box in the top-right section of the image,
    displaying each landmark's distinct color alongside its corresponding label.
    """
    h, w = vis_img.shape[:2]
    total_landmarks = min(num_landmarks, len(LANDMARK_CLASSES))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1
    row_height = 18
    padding_x = 10
    padding_y = 8
    dot_radius = 4

    # Calculate max text width for tight guide box sizing
    max_text_w = 0
    for i in range(total_landmarks):
        label_text = f"{i}:{LANDMARK_CLASSES[i]}"
        (text_w, _), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        if text_w > max_text_w:
            max_text_w = text_w

    box_w = padding_x * 2 + dot_radius * 2 + 8 + max_text_w
    box_h = padding_y * 2 + total_landmarks * row_height

    margin_right = 12
    margin_top = 12

    box_x2 = w - margin_right
    box_x1 = max(0, box_x2 - box_w)
    box_y1 = margin_top
    box_y2 = min(h, box_y1 + box_h)

    # Semi-transparent dark background for the guide box
    sub_img = vis_img[box_y1:box_y2, box_x1:box_x2]
    if sub_img.size > 0:
        dark_rect = np.zeros_like(sub_img, dtype=np.uint8)
        blended = cv2.addWeighted(sub_img, 0.2, dark_rect, 0.8, 0)
        vis_img[box_y1:box_y2, box_x1:box_x2] = blended

    # White outline border for the guide box
    cv2.rectangle(vis_img, (box_x1, box_y1), (box_x2, box_y2), (220, 220, 220), 1)

    # Render entries inside the guide box
    for i in range(total_landmarks):
        color_bgr = LANDMARK_COLORS_BGR[i % len(LANDMARK_COLORS_BGR)]
        label_text = f"{i}:{LANDMARK_CLASSES[i]}"

        center_y = box_y1 + padding_y + (i * row_height) + (row_height // 2)
        dot_x = box_x1 + padding_x + dot_radius
        text_x = dot_x + dot_radius + 6

        # Draw colored dot
        cv2.circle(vis_img, (dot_x, center_y), dot_radius, color_bgr, -1)
        cv2.circle(vis_img, (dot_x, center_y), dot_radius + 1, (0, 0, 0), 1)

        # Draw corresponding text label
        cv2.putText(
            vis_img,
            label_text,
            (text_x, center_y + 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Instantiates CephalometricSwinGCN and loads checkpoint weights safely.
    """
    print(f"Loading checkpoint weights from: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at '{weights_path}'")

    model = CephalometricSwinGCN(num_landmarks=NUM_LANDMARKS, pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        # Full saved model object
        model = checkpoint
        model.to(device)
        model.eval()
        return model

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_metrics(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    img_size: int = 640,
    threshold_px: float = 2.5,
) -> tuple[dict, list[dict]]:
    """
    Computes overall summary metrics and per-landmark metrics breakdown.

    Args:
        pred_coords: Array of shape (N, 13, 2) in normalized [0, 1] range.
        gt_coords: Array of shape (N, 13, 2) in normalized [0, 1] range (-1.0 for missing).
        img_size: Image dimension in pixels (e.g. 640).
        threshold_px: Tolerance threshold in pixels for binary detection metrics.

    Returns:
        summary_metrics: Dict of overall dataset evaluation metrics.
        landmark_metrics: List of dicts with metrics per landmark.
    """
    # Convert normalized coordinates to absolute pixel scale
    pred_px = pred_coords * img_size
    gt_px = gt_coords * img_size

    # Valid mask for GT coordinates (excluding missing landmarks marked as -1.0)
    valid_mask = (gt_coords[:, :, 0] >= 0) & (gt_coords[:, :, 1] >= 0)  # Shape: (N, 13)

    dx = pred_px[:, :, 0] - gt_px[:, :, 0]
    dy = pred_px[:, :, 1] - gt_px[:, :, 1]
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    radial_errors = np.sqrt(dx**2 + dy**2)  # Shape: (N, 13)

    valid_radial_errors = radial_errors[valid_mask]
    valid_abs_dx = abs_dx[valid_mask]
    valid_abs_dy = abs_dy[valid_mask]

    if len(valid_radial_errors) == 0:
        raise ValueError("No valid ground truth landmarks found for metric calculation.")

    # 1. Regression Metrics
    mae_x = float(np.mean(valid_abs_dx))
    mae_y = float(np.mean(valid_abs_dy))
    mae = float(np.mean((valid_abs_dx + valid_abs_dy) / 2.0))
    rmse = float(np.sqrt(np.mean(dx[valid_mask] ** 2 + dy[valid_mask] ** 2)))
    mre = float(np.mean(valid_radial_errors))
    medre = float(np.median(valid_radial_errors))
    sdre = float(np.std(valid_radial_errors))
    min_err = float(np.min(valid_radial_errors))
    max_err = float(np.max(valid_radial_errors))

    # 2. SDR (Successful Detection Rate) at various radial thresholds in pixels
    sdr_thresholds = [2.0, 2.5, 3.0, 4.0, 5.0, 10.0]
    sdr_dict = {}
    for th in sdr_thresholds:
        sdr_val = float(np.mean(valid_radial_errors <= th) * 100.0)
        sdr_dict[f"sdr_{th}px"] = sdr_val

    total_valid = len(valid_radial_errors)

    summary_metrics = {
        "total_samples": int(pred_coords.shape[0]),
        "total_valid_landmarks": total_valid,
        "mae_pixels": mae,
        "mae_x_pixels": mae_x,
        "mae_y_pixels": mae_y,
        "rmse_pixels": rmse,
        "mre_pixels": mre,
        "medre_pixels": medre,
        "sdre_pixels": sdre,
        "min_error_pixels": min_err,
        "max_error_pixels": max_err,
        **sdr_dict,
    }

    # 3. Per-landmark Breakdown
    landmark_metrics = []
    for i, name in enumerate(LANDMARK_CLASSES):
        l_mask = valid_mask[:, i]
        if not np.any(l_mask):
            continue

        l_radial = radial_errors[:, i][l_mask]
        l_dx = dx[:, i][l_mask]
        l_dy = dy[:, i][l_mask]
        l_abs_dx = abs_dx[:, i][l_mask]
        l_abs_dy = abs_dy[:, i][l_mask]

        l_mae = float(np.mean((l_abs_dx + l_abs_dy) / 2.0))
        l_rmse = float(np.sqrt(np.mean(l_dx**2 + l_dy**2)))
        l_mre = float(np.mean(l_radial))
        l_medre = float(np.median(l_radial))
        l_sdre = float(np.std(l_radial))

        l_sdr2_5 = float(np.mean(l_radial <= 2.5) * 100.0)
        l_sdr4_0 = float(np.mean(l_radial <= 4.0) * 100.0)

        landmark_metrics.append({
            "id": i,
            "name": name,
            "count": int(np.sum(l_mask)),
            "mae_px": l_mae,
            "rmse_px": l_rmse,
            "mre_px": l_mre,
            "medre_px": l_medre,
            "sdre_px": l_sdre,
            "sdr_2.5px": l_sdr2_5,
            "sdr_4.0px": l_sdr4_0,
        })

    return summary_metrics, landmark_metrics, valid_radial_errors


def format_metrics_table(
    summary: dict,
    landmarks: list[dict],
    run_name: str,
    weights_path: str,
) -> str:
    """
    Formats the evaluation results into ASCII terminal tables.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f" SWIN-GCN MODEL EVALUATION REPORT | RUN: {run_name}")
    lines.append("=" * 80)
    lines.append(f"Checkpoint Weights: {weights_path}")
    lines.append(f"Total Test Images: {summary['total_samples']} | Total Valid Landmarks: {summary['total_valid_landmarks']}")
    lines.append("-" * 80)

    # 1. Overall Summary Table
    lines.append("\n" + "+" + "-" * 42 + "+" + "-" * 35 + "+")
    lines.append(f"| {'EVALUATION METRIC':<40} | {'VALUE':<33} |")
    lines.append("+" + "-" * 42 + "+" + "-" * 35 + "+")
    
    summary_rows = [
        ("MAE (Pixels)", f"{summary['mae_pixels']:.4f} px"),
        ("RMSE (Pixels)", f"{summary['rmse_pixels']:.4f} px"),
        ("MRE (Mean Radial Error)", f"{summary['mre_pixels']:.4f} px"),
        ("MedRE (Median Radial Error)", f"{summary['medre_pixels']:.4f} px"),
        ("SDRE (Std Dev Radial Error)", f"{summary['sdre_pixels']:.4f} px"),
        ("Min Radial Error", f"{summary['min_error_pixels']:.4f} px"),
        ("Max Radial Error", f"{summary['max_error_pixels']:.4f} px"),
        ("SDR @ 2.0 px", f"{summary['sdr_2.0px']:.2f} %"),
        ("SDR @ 2.5 px", f"{summary['sdr_2.5px']:.2f} %"),
        ("SDR @ 3.0 px", f"{summary['sdr_3.0px']:.2f} %"),
        ("SDR @ 4.0 px", f"{summary['sdr_4.0px']:.2f} %"),
        ("SDR @ 5.0 px", f"{summary['sdr_5.0px']:.2f} %"),
        ("SDR @ 10.0 px", f"{summary['sdr_10.0px']:.2f} %"),
    ]

    for label, val in summary_rows:
        lines.append(f"| {label:<40} | {val:<33} |")
    lines.append("+" + "-" * 42 + "+" + "-" * 35 + "+")

    # 2. Per-Landmark Breakdown Table
    lines.append("\n" + "+" + "-" * 88 + "+")
    lines.append(f"| {'PER-LANDMARK METRICS BREAKDOWN':^86} |")
    lines.append("+" + "-" * 4 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+")
    lines.append(f"| {'ID':<2} | {'Landmark':<10} | {'MAE (px)':<8} | {'RMSE (px)':<9} | {'MRE (px)':<8} | {'MedRE(px)':<9} | {'SDR@2.5px':<9} | {'SDR@4.0px':<9} |")
    lines.append("+" + "-" * 4 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+")

    for lm in landmarks:
        lines.append(
            f"| {lm['id']:<2} | {lm['name']:<10} | {lm['mae_px']:<8.2f} | {lm['rmse_px']:<9.2f} | {lm['mre_px']:<8.2f} | {lm['medre_px']:<9.2f} | {lm['sdr_2.5px']:<8.1f}% | {lm['sdr_4.0px']:<8.1f}% |"
        )
    lines.append("+" + "-" * 4 + "+" + "-" * 12 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+")

    return "\n".join(lines)


def generate_evaluation_charts(
    summary_metrics: dict,
    landmark_metrics: list[dict],
    valid_radial_errors: np.ndarray,
    save_dir: Path | str | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, str]:
    """
    Generates PNG diagram charts for all evaluation metrics and logs them as artifacts to MLflow.
    """
    use_temp = save_dir is None
    temp_dir_obj = None
    if use_temp:
        temp_dir_obj = tempfile.TemporaryDirectory()
        target_dir = Path(temp_dir_obj.name)
    else:
        target_dir = Path(save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    generated_charts = {}

    try:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.family": "sans-serif"})

        group_colors = {"C2": "#e63946", "C3": "#2a9d8f", "C4": "#457b9d"}
        lm_names = [lm["name"] for lm in landmark_metrics]
        lm_indices = np.arange(len(landmark_metrics))
        width = 0.25

        # 1. Landmark Errors Bar Chart (MAE, RMSE, MRE)
        fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        mre_vals = [lm["mre_px"] for lm in landmark_metrics]
        mae_vals = [lm["mae_px"] for lm in landmark_metrics]
        rmse_vals = [lm["rmse_px"] for lm in landmark_metrics]

        ax1.bar(lm_indices - width, mre_vals, width, label="MRE (px)", color="#e76f51", alpha=0.9)
        ax1.bar(lm_indices, mae_vals, width, label="MAE (px)", color="#2a9d8f", alpha=0.9)
        ax1.bar(lm_indices + width, rmse_vals, width, label="RMSE (px)", color="#457b9d", alpha=0.9)

        ax1.set_xticks(lm_indices)
        ax1.set_xticklabels(lm_names, rotation=45, ha="right")
        ax1.set_ylabel("Error (Pixels)")
        ax1.set_title("Per-Landmark Errors Breakdown (MRE / MAE / RMSE)", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p1 = target_dir / "chart_landmark_errors.png"
        fig1.savefig(p1, bbox_inches="tight")
        plt.close(fig1)
        generated_charts["landmark_errors"] = str(p1)

        # 2. Per-Landmark SDR Bar Chart (SDR@2.5px vs SDR@4.0px)
        fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=300)
        sdr2_5 = [lm["sdr_2.5px"] for lm in landmark_metrics]
        sdr4_0 = [lm["sdr_4.0px"] for lm in landmark_metrics]

        ax2.bar(lm_indices - width / 2, sdr2_5, width, label="SDR @ 2.5px (%)", color="#3a86ff", alpha=0.9)
        ax2.bar(lm_indices + width / 2, sdr4_0, width, label="SDR @ 4.0px (%)", color="#8338ec", alpha=0.9)

        ax2.set_xticks(lm_indices)
        ax2.set_xticklabels(lm_names, rotation=45, ha="right")
        ax2.set_ylabel("Detection Rate (%)")
        ax2.set_ylim(0, 105)
        ax2.set_title("Per-Landmark SDR Comparison (SDR@2.5px vs SDR@4.0px)", fontsize=12, fontweight="bold")
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p2 = target_dir / "chart_landmark_sdr.png"
        fig2.savefig(p2, bbox_inches="tight")
        plt.close(fig2)
        generated_charts["landmark_sdr"] = str(p2)

        # 3. SDR Threshold Curve (Cumulative Detection Rate)
        fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=300)
        thresholds = [2.0, 2.5, 3.0, 4.0, 5.0, 10.0]
        sdr_values = [summary_metrics.get(f"sdr_{th}px", 0.0) for th in thresholds]

        ax3.plot(thresholds, sdr_values, marker="o", linewidth=2.5, color="#ff006e", markersize=8)
        for th, val in zip(thresholds, sdr_values):
            ax3.annotate(f"{val:.1f}%", (th, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

        ax3.set_xlabel("Radius Threshold (Pixels)")
        ax3.set_ylabel("Successful Detection Rate (%)")
        ax3.set_ylim(0, 108)
        ax3.set_title("Cumulative SDR Curve Across Distance Thresholds", fontsize=12, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p3 = target_dir / "chart_sdr_thresholds.png"
        fig3.savefig(p3, bbox_inches="tight")
        plt.close(fig3)
        generated_charts["sdr_thresholds"] = str(p3)

        # 4. Radial Error Distribution Histogram
        fig4, ax4 = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax4.hist(valid_radial_errors, bins=30, color="#fb8500", edgecolor="black", alpha=0.7, density=True, label="Radial Error (px)")

        mean_err = summary_metrics.get("mre_pixels", np.mean(valid_radial_errors))
        med_err = summary_metrics.get("medre_pixels", np.median(valid_radial_errors))

        ax4.axvline(mean_err, color="red", linestyle="--", linewidth=2, label=f"Mean (MRE): {mean_err:.2f}px")
        ax4.axvline(med_err, color="green", linestyle="-.", linewidth=2, label=f"Median (MedRE): {med_err:.2f}px")

        ax4.set_xlabel("Radial Error (Pixels)")
        ax4.set_ylabel("Density")
        ax4.set_title("Radial Error Distribution Across All Test Samples", fontsize=12, fontweight="bold")
        ax4.legend(loc="upper right")
        ax4.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p4 = target_dir / "chart_radial_error_distribution.png"
        fig4.savefig(p4, bbox_inches="tight")
        plt.close(fig4)
        generated_charts["radial_error_distribution"] = str(p4)

        # 5. Evaluation Dashboard (2x2 Grid)
        fig5, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

        # Subplot (0,0): Landmark Errors
        axes[0, 0].bar(lm_indices - width, mre_vals, width, label="MRE", color="#e76f51")
        axes[0, 0].bar(lm_indices, mae_vals, width, label="MAE", color="#2a9d8f")
        axes[0, 0].bar(lm_indices + width, rmse_vals, width, label="RMSE", color="#457b9d")
        axes[0, 0].set_xticks(lm_indices)
        axes[0, 0].set_xticklabels(lm_names, rotation=45, ha="right", fontsize=8)
        axes[0, 0].set_title("Landmark Errors Breakdown", fontsize=10, fontweight="bold")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        # Subplot (0,1): Per-Landmark SDR
        axes[0, 1].bar(lm_indices - width / 2, sdr2_5, width, label="SDR@2.5px", color="#3a86ff")
        axes[0, 1].bar(lm_indices + width / 2, sdr4_0, width, label="SDR@4.0px", color="#8338ec")
        axes[0, 1].set_xticks(lm_indices)
        axes[0, 1].set_xticklabels(lm_names, rotation=45, ha="right", fontsize=8)
        axes[0, 1].set_title("Per-Landmark SDR", fontsize=10, fontweight="bold")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        # Subplot (1,0): SDR Curve
        axes[1, 0].plot(thresholds, sdr_values, marker="o", color="#ff006e", linewidth=2)
        axes[1, 0].set_title("Cumulative SDR Curve", fontsize=10, fontweight="bold")
        axes[1, 0].set_xlabel("Threshold (px)", fontsize=9)
        axes[1, 0].set_ylabel("SDR (%)", fontsize=9)
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        # Subplot (1,1): Error Distribution
        axes[1, 1].hist(valid_radial_errors, bins=25, color="#fb8500", alpha=0.7, density=True)
        axes[1, 1].axvline(mean_err, color="red", linestyle="--", label=f"MRE: {mean_err:.2f}")
        axes[1, 1].set_title("Radial Error Distribution", fontsize=10, fontweight="bold")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("SWIN-GCN EVALUATION METRICS DASHBOARD", fontsize=14, fontweight="bold")
        plt.tight_layout()
        p5 = target_dir / "chart_evaluation_dashboard.png"
        fig5.savefig(p5, bbox_inches="tight")
        plt.close(fig5)
        generated_charts["evaluation_dashboard"] = str(p5)

        # Log artifacts to MLflow if tracking is active
        if log_to_mlflow:
            try:
                active_run = mlflow.active_run()
                if active_run is not None:
                    mlflow.log_artifacts(str(target_dir), artifact_path="evaluation/charts")
                    print(f"--> Successfully logged evaluation metric PNG charts to MLflow artifact path 'evaluation/charts'")
            except Exception as ml_err:
                print(f"⚠️ Warning: Could not log evaluation charts to MLflow: {ml_err}")

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    return generated_charts


def draw_landmarks_on_image(
    image: np.ndarray,
    pred_coords: np.ndarray,
    gt_coords: np.ndarray | None = None,
    img_size: int = 640,
) -> np.ndarray:
    """
    Renders predicted landmarks (and optionally ground truth) onto an RGB image.
    Each landmark is drawn in a distinct color, and a rectangular guide box (legend)
    is rendered on the top right section of the image.
    """
    vis_img = image.copy()
    h, w = vis_img.shape[:2]

    # Draw Ground Truth landmarks first (Yellow circle markers)
    if gt_coords is not None:
        for i in range(len(gt_coords)):
            x_norm, y_norm = gt_coords[i]
            if x_norm < 0 or y_norm < 0:
                continue
            abs_x = int(round(x_norm * w))
            abs_y = int(round(y_norm * h))
            cv2.circle(vis_img, (abs_x, abs_y), 6, (255, 255, 0), 1)  # Yellow outline circle
            cv2.drawMarker(vis_img, (abs_x, abs_y), (255, 255, 0), cv2.MARKER_CROSS, 8, 1)

    # Draw Predicted landmarks with distinct colors
    for i in range(len(pred_coords)):
        x_norm, y_norm = pred_coords[i]
        if x_norm < 0 or y_norm < 0:
            continue
        abs_x = int(round(x_norm * w))
        abs_y = int(round(y_norm * h))

        color_bgr = LANDMARK_COLORS_BGR[i % len(LANDMARK_COLORS_BGR)]

        # Draw filled landmark circle with black border
        cv2.circle(vis_img, (abs_x, abs_y), 4, color_bgr, -1)
        cv2.circle(vis_img, (abs_x, abs_y), 5, (0, 0, 0), 1)

    # Add rectangular guide box on top-right section
    draw_legend_box(vis_img, len(pred_coords))

    return vis_img


def visualize_batch_and_save(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    save_dir: Path,
    num_samples: int = 8,
    img_size: int = 640,
    test_img_dir: str = "dataset/test/images",
):
    """
    Runs model on a batch of test data and saves visualized PNG images into save_dir.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    with torch.no_grad():
        for batch in test_loader:
            images_tensor = batch["image"].to(device)
            gt_coords_batch = batch["coords"].cpu().numpy()
            filenames = batch.get("filename", [f"sample_{i}.png" for i in range(len(images_tensor))])

            _, pred_coords_batch = model(images_tensor)
            pred_coords_batch = pred_coords_batch.cpu().numpy()

            for idx in range(len(images_tensor)):
                if saved_count >= num_samples:
                    break

                img_name = filenames[idx] if isinstance(filenames, (list, tuple)) else filenames
                raw_img_path = os.path.join(test_img_dir, img_name)

                if os.path.exists(raw_img_path):
                    image_bgr = cv2.imread(raw_img_path)
                else:
                    # Fallback to un-normalizing tensor image
                    img_np = images_tensor[idx].cpu().numpy().transpose(1, 2, 0)
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255.0).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                    image_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                rendered_img = draw_landmarks_on_image(
                    image=image_bgr,
                    pred_coords=pred_coords_batch[idx],
                    gt_coords=gt_coords_batch[idx],
                    img_size=img_size,
                )

                out_path = save_dir / f"eval_{saved_count + 1:03d}_{img_name}"
                cv2.imwrite(str(out_path), rendered_img)
                saved_count += 1

            if saved_count >= num_samples:
                break

    print(f"--> Saved {saved_count} visualization PNGs to '{save_dir}'")


def run_evaluation(
    weights_path: str = "./artifacts/best.pth",
    test_img_dir: str = "dataset/test/images",
    test_npz_dir: str = "dataset/test/labels",
    output_dir: str = "evaluation",
    run_name: str | None = None,
    batch_size: int = 8,
    img_size: int = 640,
    num_samples: int = 8,
    threshold_px: float = 2.5,
    device_str: str | None = None,
) -> tuple[dict, str]:
    """
    Main programmatic evaluation routine.

    Returns:
        summary_metrics: Dictionary of metric values.
        run_folder: Path string of the run subfolder.
    """
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n" + "=" * 80)
    print(f"STARTING MODEL EVALUATION | Device: {device}")
    print(f"=" * 80)

    # 1. Setup output run folder
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    run_folder = Path(output_dir) / run_name
    run_folder.mkdir(parents=True, exist_ok=True)
    vis_folder = run_folder / "visualizations"

    # 2. Load model & test dataset
    model = load_model(weights_path, device)
    test_loader = get_test_dataloader(
        test_img_dir=test_img_dir,
        test_npz_dir=test_npz_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=0,  # Safety for cross-platform evaluation
    )

    # 3. Aggregate all predictions and ground truth targets
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            gt_coords = batch["coords"].cpu().numpy()
            _, pred_coords = model(images)

            all_preds.append(pred_coords.cpu().numpy())
            all_gts.append(gt_coords)

    pred_array = np.concatenate(all_preds, axis=0)  # (N, 13, 2)
    gt_array = np.concatenate(all_gts, axis=0)      # (N, 13, 2)

    # 4. Compute metrics
    summary_metrics, landmark_metrics, valid_radial_errors = compute_metrics(
        pred_coords=pred_array,
        gt_coords=gt_array,
        img_size=img_size,
        threshold_px=threshold_px,
    )

    # 4b. Generate & log metric PNG charts to MLflow
    generate_evaluation_charts(
        summary_metrics=summary_metrics,
        landmark_metrics=landmark_metrics,
        valid_radial_errors=valid_radial_errors,
        save_dir=run_folder / "charts",
        log_to_mlflow=True,
    )

    # 5. Format & print table
    table_text = format_metrics_table(
        summary=summary_metrics,
        landmarks=landmark_metrics,
        run_name=run_name,
        weights_path=weights_path,
    )
    print("\n" + table_text + "\n")

    # 6. Save JSON metrics and summary report TXT
    metrics_export = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "weights_path": weights_path,
        "summary": summary_metrics,
        "landmarks": landmark_metrics,
    }

    json_path = run_folder / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_export, f, indent=4)
    print(f"--> Saved evaluation metrics JSON to '{json_path}'")

    txt_path = run_folder / "summary_report.txt"
    with open(txt_path, "w") as f:
        f.write(table_text + "\n")
    print(f"--> Saved summary report text to '{txt_path}'")

    # 7. Render & save visualization PNGs
    visualize_batch_and_save(
        test_loader=test_loader,
        model=model,
        device=device,
        save_dir=vis_folder,
        num_samples=num_samples,
        img_size=img_size,
        test_img_dir=test_img_dir,
    )

    print(f"\nEvaluation run '{run_name}' completed successfully.")
    print(f"All artifacts saved in directory: '{run_folder}'\n" + "=" * 80)

    return summary_metrics, str(run_folder)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Swin-GCN landmark detection model on test dataset."
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default="./artifacts/best.pth",
        help="Path to trained PyTorch model checkpoint (.pth or .pt).",
    )
    parser.add_argument(
        "--test-img-dir",
        type=str,
        default="dataset/test/images",
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--test-npz-dir",
        type=str,
        default="dataset/test/labels",
        help="Directory containing test label .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Base folder to store evaluation run subfolders.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom subfolder name for this evaluation run (defaults to run_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for test evaluation.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Target image dimension in pixels.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of test image visualizations to save as PNG.",
    )
    parser.add_argument(
        "--threshold-px",
        type=float,
        default=2.5,
        help="Radial error tolerance threshold in pixels for detection metrics.",
    )

    args = parser.parse_args()

    run_evaluation(
        weights_path=args.weights,
        test_img_dir=args.test_img_dir,
        test_npz_dir=args.test_npz_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_samples=args.num_samples,
        threshold_px=args.threshold_px,
    )


if __name__ == "__main__":
    main()
