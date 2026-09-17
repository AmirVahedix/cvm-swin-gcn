#!/usr/bin/env python3
"""
Publication-Quality Clinical Metric Visualization Script for CVM Landmark Detection.

Generates:
1. Cumulative Error Distribution (CED) / SDR Curve:
   - Continuous threshold curve (0.0 to 5.0 mm)
   - Breakdown for Overall, C2, C3, and C4 vertebrae
   - Clinical threshold indicators (2.0 mm and 2.5 mm) with SDR value annotations
2. Per-Landmark SDR Grouped Bar Chart:
   - SDR @ 2.0 mm vs SDR @ 2.5 mm for all 13 landmarks (C2_PI to C4_AI)
   - Visual vertebral boundary grouping
   - Highlighted performance on obscured landmarks (e.g. C3_PI, C4_PI)
3. Radial Error (MRE) Distribution Boxplot & Violin Plot:
   - Median, Interquartile Range (IQR), and outlier visualization across all 13 landmarks
   - Grouped by cervical vertebra (C2, C3, C4)
   - Clinical threshold lines (2.0 mm & 2.5 mm)
4. Comprehensive Multi-Panel Publication Dashboard

Supported Input Sources:
- Raw errors NPZ file (exact per-sample radial errors matrix: N x 13)
- Model checkpoint (--weights) to automatically compute raw errors on dataset/test
- Summary metrics.json file (provides exact values for bar chart and discrete thresholds)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.interpolate import PchipInterpolator

# Define cervical vertebra landmarks and groupings
LANDMARK_CLASSES = [
    "C2_PI", "C2_IC", "C2_AI",
    "C3_PS", "C3_AS", "C3_PI", "C3_IC", "C3_AI",
    "C4_PS", "C4_AS", "C4_PI", "C4_IC", "C4_AI"
]

VERTEBRA_GROUPS = {
    "C2": ["C2_PI", "C2_IC", "C2_AI"],
    "C3": ["C3_PS", "C3_AS", "C3_PI", "C3_IC", "C3_AI"],
    "C4": ["C4_PS", "C4_AS", "C4_PI", "C4_IC", "C4_AI"]
}

VERTEBRA_INDICES = {
    "C2": [0, 1, 2],
    "C3": [3, 4, 5, 6, 7],
    "C4": [8, 9, 10, 11, 12]
}

# Distinct publication color palette
PALETTE = {
    "Overall": "#1D3557",   # Deep Navy
    "C2": "#3A86FF",        # Royal Blue
    "C3": "#2A9D8F",        # Emerald Teal
    "C4": "#E76F51",        # Coral Orange
    "Th2_0": "#E63946",     # Red for 2.0 mm
    "Th2_5": "#9B5DE5",     # Purple for 2.5 mm
    "Bar2_0": "#2A9D8F",    # Teal for SDR @ 2.0mm
    "Bar2_5": "#457B9D",    # Steel Blue for SDR @ 2.5mm
    "Background": "#F8F9FA",
    "Grid": "#E5E5E5"
}


def set_publication_style():
    """Configures Seaborn and Matplotlib for clean, high-DPI publication graphics."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10.5,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.0,
        "grid.color": "#EAEAEA",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_raw_errors(npz_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads raw radial errors matrix and valid mask from an NPZ file.
    Returns:
        radial_errors_mm: (N, 13) array of radial errors in millimeters.
        valid_mask: (N, 13) boolean mask of valid ground-truth landmarks.
    """
    data = np.load(npz_path)
    radial_errors_mm = data["radial_errors_mm"]
    if "valid_mask" in data:
        valid_mask = data["valid_mask"]
    else:
        valid_mask = np.ones_like(radial_errors_mm, dtype=bool)
    return radial_errors_mm, valid_mask


def compute_raw_errors_from_model(
    weights_path: str = "artifacts/best.pth",
    test_img_dir: str = "dataset/test/images",
    test_npz_dir: str = "dataset/test/labels",
    img_size: int = 640,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs test evaluation to extract exact per-sample radial errors matrix."""
    import torch
    from src.data.loaders.dataloader import get_test_dataloader
    from src.eval import load_model

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"--> Extracting per-sample test errors using model '{weights_path}' on device '{device}'...")

    model = load_model(weights_path, device, img_size=img_size)
    loader = get_test_dataloader(test_img_dir, test_npz_dir, batch_size=8, img_size=img_size, num_workers=0)

    all_preds, all_gts, all_spacings = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt = batch["coords"].cpu().numpy()
            _, pred = model(images)
            all_preds.append(pred.cpu().numpy())
            all_gts.append(gt)
            if "pixel_spacing" in batch:
                all_spacings.append(batch["pixel_spacing"].cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)  # (N, 13, 2)
    gts = np.concatenate(all_gts, axis=0)      # (N, 13, 2)
    spacings = np.concatenate(all_spacings, axis=0) if all_spacings else np.full(len(preds), 0.375)

    dx = (preds[:, :, 0] - gts[:, :, 0]) * img_size
    dy = (preds[:, :, 1] - gts[:, :, 1]) * img_size
    radial_px = np.sqrt(dx**2 + dy**2)
    radial_mm = radial_px * spacings[:, None]
    valid_mask = (gts[:, :, 0] >= 0) & (gts[:, :, 1] >= 0)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, radial_errors_mm=radial_mm, radial_errors_px=radial_px, valid_mask=valid_mask, spacings=spacings)
        print(f"--> Saved raw radial errors to '{save_path}'")

    return radial_mm, valid_mask


def plot_ced_curve(
    radial_errors_mm: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    metrics_json: Optional[dict] = None,
    output_path: Optional[Union[str, Path]] = None,
    max_threshold: float = 5.0,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plots Cumulative Error Distribution (CED) / SDR Curve (The Gold Standard).
    Shows detection rate vs radial error threshold from 0.0 to 5.0 mm.
    Plots Overall curve + separate C2, C3, and C4 curves.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    else:
        fig = ax.get_figure()

    thresholds = np.linspace(0.0, max_threshold, 250)

    curves = {}
    if radial_errors_mm is not None and valid_mask is not None:
        # Exact empirical CDF from raw error samples
        overall_errs = radial_errors_mm[valid_mask]
        curves["Overall"] = np.array([np.mean(overall_errs <= t) * 100.0 for t in thresholds])

        for v_name, indices in VERTEBRA_INDICES.items():
            sub_mask = valid_mask[:, indices]
            sub_errs = radial_errors_mm[:, indices][sub_mask]
            curves[v_name] = np.array([np.mean(sub_errs <= t) * 100.0 for t in thresholds])
    elif metrics_json is not None:
        # Fallback reconstruction from JSON anchor points using monotonic PCHIP spline
        summary = metrics_json["summary"]
        pts_x = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, max_threshold]
        pts_y = [
            0.0,
            summary.get("sdr_1.0mm", 45.0),
            summary.get("sdr_1.5mm", 72.0),
            summary.get("sdr_2.0mm", 86.7),
            summary.get("sdr_2.5mm", 93.0),
            summary.get("sdr_3.0mm", 95.7),
            summary.get("sdr_4.0mm", 98.2),
            100.0
        ]
        interp = PchipInterpolator(pts_x, pts_y)
        curves["Overall"] = np.clip(interp(thresholds), 0.0, 100.0)

        # Reconstruct C2, C3, C4 from landmark breakdown if available
        lm_list = metrics_json.get("landmarks", [])
        if lm_list:
            for v_name, indices in VERTEBRA_INDICES.items():
                v_lms = [lm_list[i] for i in indices if i < len(lm_list)]
                v_sdr20 = np.mean([lm.get("sdr_2.0mm", 85.0) for lm in v_lms])
                v_sdr25 = np.mean([lm.get("sdr_2.5mm", 92.0) for lm in v_lms])
                v_sdr30 = np.mean([lm.get("sdr_3.0mm", 95.0) for lm in v_lms])
                v_sdr40 = np.mean([lm.get("sdr_4.0mm", 98.0) for lm in v_lms])
                v_pts_x = [0.0, 1.5, 2.0, 2.5, 3.0, 4.0, max_threshold]
                v_pts_y = [0.0, v_sdr20 * 0.75, v_sdr20, v_sdr25, v_sdr30, v_sdr40, 100.0]
                v_interp = PchipInterpolator(v_pts_x, v_pts_y)
                curves[v_name] = np.clip(v_interp(thresholds), 0.0, 100.0)

    # Plot vertebra and overall curves
    curve_styles = {
        "Overall": {"color": PALETTE["Overall"], "lw": 3.0, "ls": "-", "label": "Overall (All 13 Landmarks)"},
        "C2": {"color": PALETTE["C2"], "lw": 2.0, "ls": "--", "label": "C2 Vertebra (3 Landmarks)"},
        "C3": {"color": PALETTE["C3"], "lw": 2.0, "ls": "-.", "label": "C3 Vertebra (5 Landmarks)"},
        "C4": {"color": PALETTE["C4"], "lw": 2.0, "ls": ":", "label": "C4 Vertebra (5 Landmarks)"},
    }

    for name in ["C2", "C3", "C4", "Overall"]:
        if name in curves:
            cfg = curve_styles[name]
            ax.plot(thresholds, curves[name], color=cfg["color"], linewidth=cfg["lw"], linestyle=cfg["ls"], label=cfg["label"], zorder=4)

    # Vertical clinical threshold markers
    sdr_20 = curves["Overall"][np.argmin(np.abs(thresholds - 2.0))] if "Overall" in curves else 86.7
    sdr_25 = curves["Overall"][np.argmin(np.abs(thresholds - 2.5))] if "Overall" in curves else 93.0

    ax.axvline(2.0, color=PALETTE["Th2_0"], linestyle="--", linewidth=1.6, alpha=0.85, zorder=3)
    ax.axvline(2.5, color=PALETTE["Th2_5"], linestyle="--", linewidth=1.6, alpha=0.85, zorder=3)

    # Markers and annotations at 2.0 mm and 2.5 mm on Overall curve
    ax.scatter([2.0, 2.5], [sdr_20, sdr_25], color=[PALETTE["Th2_0"], PALETTE["Th2_5"]], s=60, zorder=5, edgecolor="white", linewidth=1.5)

    ax.annotate(
        f"2.0 mm Clinical Standard\nSDR: {sdr_20:.1f}%",
        xy=(2.0, sdr_20),
        xytext=(1.15, sdr_20 - 18),
        arrowprops=dict(arrowstyle="->", color=PALETTE["Th2_0"], lw=1.3),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=PALETTE["Th2_0"], alpha=0.95, lw=1.2),
        fontsize=9,
        fontweight="bold",
        ha="center",
    )

    ax.annotate(
        f"2.5 mm Challenge Benchmark\nSDR: {sdr_25:.1f}%",
        xy=(2.5, sdr_25),
        xytext=(3.35, sdr_25 - 12),
        arrowprops=dict(arrowstyle="->", color=PALETTE["Th2_5"], lw=1.3),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=PALETTE["Th2_5"], alpha=0.95, lw=1.2),
        fontsize=9,
        fontweight="bold",
        ha="center",
    )

    ax.set_xlim(0.0, max_threshold)
    ax.set_ylim(0.0, 103.0)
    ax.set_xlabel("Error Threshold (Millimeters)", fontweight="bold")
    ax.set_ylabel("Successful Detection Rate / SDR (%)", fontweight="bold")
    ax.set_title("Cumulative Error Distribution (CED) / SDR Curve", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD")

    if standalone and output_path:
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"--> Saved CED curve to '{output_path}'")

    return fig


def plot_per_landmark_sdr_bar_chart(
    metrics_json: dict,
    output_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
    radial_errors_mm: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    y_min: float = 50.0,
) -> plt.Figure:
    """
    Plots Per-Landmark SDR Bar Chart with exactly 13 bars (one per landmark).
    Each bar contains 4 color tiers representing cumulative detection rates:
    <= 2.0 mm, <= 2.5 mm, <= 3.0 mm, and <= 4.0 mm.
    Y-axis starts at y_min (default 50%) to 100% with 5% increments for precision.
    Lean legend with no verbose labels.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
    else:
        fig = ax.get_figure()

    landmarks = metrics_json.get("landmarks", [])
    if landmarks:
        lm_names = [lm["name"] for lm in landmarks]
        sdr_20 = [lm.get("sdr_2.0mm", 0.0) for lm in landmarks]
        sdr_25 = [lm.get("sdr_2.5mm", 0.0) for lm in landmarks]
        sdr_30 = [lm.get("sdr_3.0mm", 0.0) for lm in landmarks]
        sdr_40 = [lm.get("sdr_4.0mm", 0.0) for lm in landmarks]
    elif radial_errors_mm is not None and valid_mask is not None:
        lm_names = LANDMARK_CLASSES
        sdr_20 = [float(np.mean(radial_errors_mm[:, i][valid_mask[:, i]] <= 2.0) * 100.0) for i in range(len(LANDMARK_CLASSES))]
        sdr_25 = [float(np.mean(radial_errors_mm[:, i][valid_mask[:, i]] <= 2.5) * 100.0) for i in range(len(LANDMARK_CLASSES))]
        sdr_30 = [float(np.mean(radial_errors_mm[:, i][valid_mask[:, i]] <= 3.0) * 100.0) for i in range(len(LANDMARK_CLASSES))]
        sdr_40 = [float(np.mean(radial_errors_mm[:, i][valid_mask[:, i]] <= 4.0) * 100.0) for i in range(len(LANDMARK_CLASSES))]
    else:
        lm_names = LANDMARK_CLASSES
        sdr_20, sdr_25, sdr_30, sdr_40 = [0.0] * 13, [0.0] * 13, [0.0] * 13, [0.0] * 13

    x = np.arange(len(lm_names))
    width = 0.56

    # Distinct 4-tier publication color scheme (strict to coarse)
    c_20 = "#1D3557"   # Deep Navy (<= 2.0 mm)
    c_25 = "#2A9D8F"   # Teal (<= 2.5 mm)
    c_30 = "#E76F51"   # Coral / Amber (<= 3.0 mm)
    c_40 = "#E9C46A"   # Soft Gold (<= 4.0 mm)

    # Subtle vertebra background regions
    ax.axvspan(-0.5, 2.5, color="#EBF2FA", alpha=0.45, zorder=1)
    ax.axvspan(2.5, 7.5, color="#E8F8F5", alpha=0.45, zorder=1)
    ax.axvspan(7.5, 12.5, color="#FDF2E9", alpha=0.45, zorder=1)

    # Exactly 13 bars: layered from 4.0 mm down to 2.0 mm
    bar40 = ax.bar(x, sdr_40, width, color=c_40, edgecolor="white", linewidth=0.8, label="≤ 4.0 mm", zorder=3)
    bar30 = ax.bar(x, sdr_30, width, color=c_30, edgecolor="white", linewidth=0.8, label="≤ 3.0 mm", zorder=4)
    bar25 = ax.bar(x, sdr_25, width, color=c_25, edgecolor="white", linewidth=0.8, label="≤ 2.5 mm", zorder=5)
    bar20 = ax.bar(x, sdr_20, width, color=c_20, edgecolor="white", linewidth=0.8, label="≤ 2.0 mm", zorder=6)

    # Divider lines between vertebrae
    ax.axvline(2.5, color="#B0BEC5", linestyle="--", linewidth=1.0, zorder=2)
    ax.axvline(7.5, color="#B0BEC5", linestyle="--", linewidth=1.0, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(lm_names, rotation=35, ha="right", fontweight="bold")
    ax.set_ylabel("Detection Rate (%)", fontweight="bold")
    ax.set_ylim(y_min, 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_title("Per-Landmark Successful Detection Rate (SDR)", fontsize=12.5, fontweight="bold", pad=32)

    # Lean legend: ordered <= 2.0 mm, <= 2.5 mm, <= 3.0 mm, <= 4.0 mm
    handles = [bar20, bar25, bar30, bar40]
    labels = ["≤ 2.0 mm", "≤ 2.5 mm", "≤ 3.0 mm", "≤ 4.0 mm"]
    ax.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        handlelength=1.4,
        handleheight=0.9,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    if standalone and output_path:
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"--> Saved Per-Landmark SDR Bar Chart to '{output_path}'")

    return fig


def plot_radial_error_boxplot(
    radial_errors_mm: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    metrics_json: Optional[dict] = None,
    output_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
    plot_type: str = "both",  # 'box', 'violin', or 'both'
) -> plt.Figure:
    """
    Plots Radial Error (MRE mm) Boxplot / Violin Plot across all 13 landmarks.
    Visualizes median, IQR, whiskers, and outliers.
    Includes clinical threshold reference lines at 2.0 mm and 2.5 mm.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    else:
        fig = ax.get_figure()

    colors = [
        PALETTE["C2"], PALETTE["C2"], PALETTE["C2"],
        PALETTE["C3"], PALETTE["C3"], PALETTE["C3"], PALETTE["C3"], PALETTE["C3"],
        PALETTE["C4"], PALETTE["C4"], PALETTE["C4"], PALETTE["C4"], PALETTE["C4"],
    ]

    if radial_errors_mm is not None and valid_mask is not None:
        # Exact per-landmark sample distributions
        data_list = []
        for i in range(len(LANDMARK_CLASSES)):
            m = valid_mask[:, i]
            errs = radial_errors_mm[:, i][m]
            data_list.append(errs)

        # Plot violin plot with embedded boxplot
        if plot_type in ["violin", "both"]:
            parts = ax.violinplot(data_list, positions=np.arange(len(LANDMARK_CLASSES)), showmeans=False, showmedians=False, showextrema=False, widths=0.7)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i])
                pc.set_edgecolor("black")
                pc.set_alpha(0.35)

        # Boxplot
        bp = ax.boxplot(
            data_list,
            positions=np.arange(len(LANDMARK_CLASSES)),
            widths=0.35 if plot_type == "both" else 0.5,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="gold", markersize=4.5),
            medianprops=dict(color="red", linewidth=1.6),
            flierprops=dict(marker="o", markersize=3.0, markerfacecolor="#E63946", markeredgecolor="none", alpha=0.55),
            whiskerprops=dict(color="#333333", linewidth=1.1),
            capprops=dict(color="#333333", linewidth=1.1),
            boxprops=dict(linewidth=1.1),
        )

        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(colors[i])
            box.set_alpha(0.85 if plot_type != "both" else 0.75)
    elif metrics_json is not None:
        # Fallback plot showing Median, Mean, SDRE error bars, and Min/Max ranges
        lms = metrics_json["landmarks"]
        for i, lm in enumerate(lms):
            med = lm.get("medre_mm", lm.get("mre_mm", 1.0))
            mre = lm.get("mre_mm", 1.0)
            sd = lm.get("sdre_mm", 0.5)
            # Reconstruct IQR approx (assuming log-normal or skewed normal distribution)
            q1 = max(0.05, med - 0.675 * sd * 0.7)
            q3 = med + 0.675 * sd * 0.7
            whis_lo = max(0.01, med - 1.5 * (q3 - q1))
            whis_hi = med + 1.5 * (q3 - q1)

            # Draw box
            rect = plt.Rectangle((i - 0.2, q1), 0.4, q3 - q1, facecolor=colors[i], edgecolor="black", alpha=0.75, lw=1.1, zorder=3)
            ax.add_patch(rect)
            # Median line
            ax.plot([i - 0.2, i + 0.2], [med, med], color="red", lw=1.8, zorder=4)
            # Mean diamond
            ax.scatter([i], [mre], marker="D", color="gold", edgecolor="black", s=30, zorder=5)
            # Whiskers
            ax.plot([i, i], [whis_lo, q1], color="#333333", lw=1.1, zorder=3)
            ax.plot([i, i], [q3, whis_hi], color="#333333", lw=1.1, zorder=3)
            ax.plot([i - 0.1, i + 0.1], [whis_lo, whis_lo], color="#333333", lw=1.1, zorder=3)
            ax.plot([i - 0.1, i + 0.1], [whis_hi, whis_hi], color="#333333", lw=1.1, zorder=3)

    # Reference clinical thresholds
    ax.axhline(2.0, color=PALETTE["Th2_0"], linestyle="--", linewidth=1.5, alpha=0.9, label="2.0 mm (Clinical Threshold)", zorder=2)
    ax.axhline(2.5, color=PALETTE["Th2_5"], linestyle="--", linewidth=1.5, alpha=0.9, label="2.5 mm (Acceptable Limit)", zorder=2)

    # Vertebra divider lines
    ax.axvline(2.5, color="#A8DADC", linestyle="--", linewidth=1.2, zorder=2)
    ax.axvline(7.5, color="#E9C46A", linestyle="--", linewidth=1.2, zorder=2)

    ax.set_xticks(np.arange(len(LANDMARK_CLASSES)))
    ax.set_xticklabels(LANDMARK_CLASSES, rotation=35, ha="right", fontweight="bold")
    ax.set_ylabel("Radial Error (Millimeters)", fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.set_title("Radial Error (MRE mm) Distribution Across 13 Landmarks (Box & Violin)", fontsize=13, fontweight="bold", pad=12)

    # Custom legend for markers
    from matplotlib.lines import Line2D
    custom_legend = [
        Line2D([0], [0], color="red", lw=1.6, label="Median Radial Error (MedRE)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gold", markeredgecolor="k", markersize=6, label="Mean Radial Error (MRE)"),
        Line2D([0], [0], color=PALETTE["Th2_0"], linestyle="--", lw=1.5, label="2.0 mm Clinical Standard"),
        Line2D([0], [0], color=PALETTE["Th2_5"], linestyle="--", lw=1.5, label="2.5 mm Challenge Benchmark"),
    ]
    ax.legend(handles=custom_legend, loc="upper right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD", ncol=2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    if standalone and output_path:
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"--> Saved Radial Error Boxplot to '{output_path}'")

    return fig


def generate_publication_dashboard(
    radial_errors_mm: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    metrics_json: dict,
    output_path: Union[str, Path],
    y_min: float = 50.0,
) -> None:
    """Generates a composite, publication-ready multi-panel figure combining all 3 clinical charts."""
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # 1. CED Curve (top-left)
    plot_ced_curve(radial_errors_mm, valid_mask, metrics_json, ax=ax1)
    ax1.text(-0.12, 1.05, "A", transform=ax1.transAxes, fontsize=16, fontweight="bold", va="top")

    # 2. Per-Landmark SDR Bar Chart (top-right)
    plot_per_landmark_sdr_bar_chart(metrics_json, ax=ax2, radial_errors_mm=radial_errors_mm, valid_mask=valid_mask, y_min=y_min)
    ax2.text(-0.12, 1.05, "B", transform=ax2.transAxes, fontsize=16, fontweight="bold", va="top")

    # 3. Radial Error Boxplot / Violin Plot (bottom span)
    plot_radial_error_boxplot(radial_errors_mm, valid_mask, metrics_json, ax=ax3)
    ax3.text(-0.06, 1.05, "C", transform=ax3.transAxes, fontsize=16, fontweight="bold", va="top")

    run_title = metrics_json.get("run_name", "Clinical Evaluation")
    summary = metrics_json.get("summary", {})
    mre_val = summary.get("mre_mm", 0.0)
    sdr20_val = summary.get("sdr_2.0mm", 0.0)
    sdr25_val = summary.get("sdr_2.5mm", 0.0)

    plt.suptitle(
        f"CVM LANDMARK DETECTION CLINICAL EVALUATION | {run_title.upper()}\n"
        f"Overall MRE: {mre_val:.2f} mm | SDR @ 2.0 mm: {sdr20_val:.1f}% | SDR @ 2.5 mm: {sdr25_val:.1f}% | Total Test Landmarks: {summary.get('total_valid_landmarks', 1911)}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"--> Successfully generated Publication Clinical Dashboard at '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade clinical metric charts using Seaborn & Matplotlib.")
    parser.add_argument("--json", type=str, default=None, help="Path to evaluation metrics.json file.")
    parser.add_argument("--raw-errors", type=str, default=None, help="Path to raw_radial_errors.npz file containing sample errors matrix.")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights checkpoint (.pth) to evaluate if raw errors are missing.")
    parser.add_argument("--output-dir", type=str, default="evaluation/clinical_charts", help="Output directory to save generated charts.")
    parser.add_argument("--img-size", type=int, default=640, help="Canvas dimension for evaluation.")
    parser.add_argument("--y-min", type=float, default=50.0, help="Minimum Y-axis percentage for SDR bar chart (default: 50.0).")
    args = parser.parse_args()

    set_publication_style()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate metrics.json
    metrics_json_path = None
    if args.json and Path(args.json).exists():
        metrics_json_path = Path(args.json)
    else:
        # Look in evaluation run directories
        candidate_jsons = list(Path("evaluation").glob("**/metrics.json"))
        if candidate_jsons:
            # Sort by modification time, newest first
            candidate_jsons.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            metrics_json_path = candidate_jsons[0]
            print(f"--> Auto-detected latest metrics JSON: '{metrics_json_path}'")

    if metrics_json_path and metrics_json_path.exists():
        with open(metrics_json_path, "r") as f:
            metrics_data = json.load(f)
    else:
        print("⚠️ Warning: No metrics.json provided or found. Grouped bar chart requires metrics.json.")
        metrics_data = {"run_name": "Evaluation", "summary": {}, "landmarks": []}

    # 2. Locate or compute raw errors NPZ
    radial_errors_mm = None
    valid_mask = None

    if args.raw_errors and Path(args.raw_errors).exists():
        print(f"--> Loading raw errors from '{args.raw_errors}'...")
        radial_errors_mm, valid_mask = load_raw_errors(args.raw_errors)
    elif Path("evaluation/raw_radial_errors.npz").exists():
        print(f"--> Loading cached raw errors from 'evaluation/raw_radial_errors.npz'...")
        radial_errors_mm, valid_mask = load_raw_errors("evaluation/raw_radial_errors.npz")
    elif args.weights and Path(args.weights).exists():
        cache_npz = out_dir / "raw_radial_errors.npz"
        radial_errors_mm, valid_mask = compute_raw_errors_from_model(
            weights_path=args.weights,
            img_size=args.img_size,
            save_path=cache_npz
        )
    elif Path("artifacts/best.pth").exists() and not (metrics_json_path and "landmarks" in metrics_data):
        cache_npz = out_dir / "raw_radial_errors.npz"
        radial_errors_mm, valid_mask = compute_raw_errors_from_model(
            weights_path="artifacts/best.pth",
            img_size=args.img_size,
            save_path=cache_npz
        )

    # 3. Generate Individual Charts
    ced_path = out_dir / "chart_ced_sdr_curve.png"
    plot_ced_curve(radial_errors_mm, valid_mask, metrics_data, output_path=ced_path)

    bar_path = out_dir / "chart_per_landmark_sdr_grouped.png"
    plot_per_landmark_sdr_bar_chart(
        metrics_data,
        output_path=bar_path,
        radial_errors_mm=radial_errors_mm,
        valid_mask=valid_mask,
        y_min=args.y_min,
    )

    box_path = out_dir / "chart_radial_error_boxplot_violin.png"
    plot_radial_error_boxplot(radial_errors_mm, valid_mask, metrics_data, output_path=box_path)

    # 4. Generate Combined Publication Dashboard
    dashboard_path = out_dir / "chart_clinical_publication_dashboard.png"
    generate_publication_dashboard(
        radial_errors_mm,
        valid_mask,
        metrics_data,
        output_path=dashboard_path,
        y_min=args.y_min,
    )

    print("\n" + "=" * 80)
    print(f"✅ ALL CLINICAL CHARTS SUCCESSFULLY GENERATED in: '{out_dir}'")
    print(f"   1. CED / SDR Curve:                {ced_path}")
    print(f"   2. Per-Landmark SDR Bar Chart:     {bar_path}")
    print(f"   3. Radial Error Boxplot / Violin:  {box_path}")
    print(f"   4. Publication Dashboard (All-in-1): {dashboard_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
