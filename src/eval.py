import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import LANDMARK_CLASSES, NUM_LANDMARKS, get_test_dataloader
from src.models.model import CephalometricSwinGCN


GROUPS = [
    {"range": range(0, 3), "color": (0, 0, 255), "mpl_color": "red", "label": "C2"},       # C2 (Red)
    {"range": range(3, 8), "color": (0, 255, 0), "mpl_color": "lime", "label": "C3"},      # C3 (Lime/Green)
    {"range": range(8, 13), "color": (255, 255, 0), "mpl_color": "cyan", "label": "C4"},   # C4 (Cyan/Blue)
]


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

    # Normalized variants
    mae_norm = mae / img_size
    rmse_norm = rmse / img_size
    mre_norm = mre / img_size

    # 2. SDR (Successful Detection Rate) at various radial thresholds in pixels
    sdr_thresholds = [2.0, 2.5, 3.0, 4.0, 5.0, 10.0]
    sdr_dict = {}
    for th in sdr_thresholds:
        sdr_val = float(np.mean(valid_radial_errors <= th) * 100.0)
        sdr_dict[f"sdr_{th}px"] = sdr_val

    # 3. Detection / Classification Metrics at specified threshold_px
    tp = int(np.sum(valid_radial_errors <= threshold_px))
    fp = int(np.sum(valid_radial_errors > threshold_px))
    fn = fp  # Missed positive targets
    tn = 0   # Valid target landmarks dataset

    total_valid = len(valid_radial_errors)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = 1.0  # Given all evaluated target instances are valid ground truth targets
    accuracy = tp / total_valid if total_valid > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    summary_metrics = {
        "total_samples": int(pred_coords.shape[0]),
        "total_valid_landmarks": total_valid,
        "mae_pixels": mae,
        "mae_x_pixels": mae_x,
        "mae_y_pixels": mae_y,
        "mae_normalized": mae_norm,
        "rmse_pixels": rmse,
        "rmse_normalized": rmse_norm,
        "mre_pixels": mre,
        "mre_normalized": mre_norm,
        "medre_pixels": medre,
        "sdre_pixels": sdre,
        "min_error_pixels": min_err,
        "max_error_pixels": max_err,
        **sdr_dict,
        "detection_threshold_px": threshold_px,
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "f1_score": float(f1_score),
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        },
    }

    # 4. Per-landmark Breakdown
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

    return summary_metrics, landmark_metrics


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
    lines.append(f"Detection Threshold: {summary['detection_threshold_px']} pixels")
    lines.append("-" * 80)

    # 1. Overall Summary Table
    lines.append("\n" + "+" + "-" * 42 + "+" + "-" * 35 + "+")
    lines.append(f"| {'EVALUATION METRIC':<40} | {'VALUE':<33} |")
    lines.append("+" + "-" * 42 + "+" + "-" * 35 + "+")
    
    summary_rows = [
        ("MAE (Pixels)", f"{summary['mae_pixels']:.4f} px"),
        ("MAE (Normalized)", f"{summary['mae_normalized']:.6f}"),
        ("RMSE (Pixels)", f"{summary['rmse_pixels']:.4f} px"),
        ("RMSE (Normalized)", f"{summary['rmse_normalized']:.6f}"),
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
        (f"Precision (@ {summary['detection_threshold_px']}px)", f"{summary['precision']:.4f}"),
        (f"Recall / Sensitivity (@ {summary['detection_threshold_px']}px)", f"{summary['recall_sensitivity']:.4f}"),
        ("Specificity", f"{summary['specificity']:.4f}"),
        ("Accuracy", f"{summary['accuracy']:.4f}"),
        ("F1-Score", f"{summary['f1_score']:.4f}"),
        ("Confusion Matrix (TP / FP / FN / TN)", f"{summary['confusion_matrix']['TP']} / {summary['confusion_matrix']['FP']} / {summary['confusion_matrix']['FN']} / {summary['confusion_matrix']['TN']}"),
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


def draw_landmarks_on_image(
    image: np.ndarray,
    pred_coords: np.ndarray,
    gt_coords: np.ndarray | None = None,
    show_labels: bool = True,
    img_size: int = 640,
) -> np.ndarray:
    """
    Renders predicted landmarks (and optionally ground truth) onto an RGB image.
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

    # Draw Predicted landmarks per anatomical group
    for group in GROUPS:
        color_bgr = group["color"]
        for i in group["range"]:
            if i >= len(pred_coords):
                continue
            x_norm, y_norm = pred_coords[i]
            abs_x = int(round(x_norm * w))
            abs_y = int(round(y_norm * h))

            # Draw filled landmark circle
            cv2.circle(vis_img, (abs_x, abs_y), 4, color_bgr, -1)
            cv2.circle(vis_img, (abs_x, abs_y), 5, (0, 0, 0), 1)  # Black border

            # Draw labels if enabled
            if show_labels:
                label_text = f"{i}:{LANDMARK_CLASSES[i]}" if i < len(LANDMARK_CLASSES) else str(i)
                text_pos = (abs_x + 6, abs_y + 4)
                
                # Draw text background box for contrast
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(
                    vis_img,
                    (text_pos[0] - 2, text_pos[1] - text_h - 2),
                    (text_pos[0] + text_w + 2, text_pos[1] + baseline),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    vis_img,
                    label_text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    return vis_img


def visualize_batch_and_save(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    save_dir: Path,
    num_samples: int = 8,
    show_labels: bool = True,
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
                    show_labels=show_labels,
                    img_size=img_size,
                )

                out_path = save_dir / f"eval_{saved_count + 1:03d}_{img_name}"
                cv2.imwrite(str(out_path), rendered_img)
                saved_count += 1

            if saved_count >= num_samples:
                break

    print(f"--> Saved {saved_count} visualization PNGs to '{save_dir}' (Labels enabled: {show_labels})")


def run_evaluation(
    weights_path: str = "./artifacts/best.pth",
    test_img_dir: str = "dataset/test/images",
    test_npz_dir: str = "dataset/test/labels",
    output_dir: str = "evaluation",
    run_name: str | None = None,
    batch_size: int = 8,
    img_size: int = 640,
    num_samples: int = 8,
    show_labels: bool = True,
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
    summary_metrics, landmark_metrics = compute_metrics(
        pred_coords=pred_array,
        gt_coords=gt_array,
        img_size=img_size,
        threshold_px=threshold_px,
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
        show_labels=show_labels,
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
        "--no-labels",
        action="store_true",
        help="Disable drawing text labels on predicted landmarks in output PNGs.",
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
        show_labels=not args.no_labels,
        threshold_px=args.threshold_px,
    )


if __name__ == "__main__":
    main()
