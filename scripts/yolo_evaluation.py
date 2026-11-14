#!/usr/bin/env python3
"""
Lightweight YOLO Evaluation Script for Cervical Spine ROI Detection.

Evaluates the Stage 1 YOLO detector (e.g. from MLflow Run ID 9ce919cc321d423fac3aa0929b14872f)
on the 120 test set images using the exact ground-truth ROI generation pipeline
from scripts/train-pipeline.py.

Metrics Computed for Publication:
  1. mAP@50 (Mean Average Precision at IoU >= 0.50)
  2. mAP@50-95 (Mean Average Precision across IoU thresholds 0.50:0.95:0.05)
  3. Precision (Positive Predictive Value)
  4. Recall / Sensitivity (True Positive Rate)
  5. Average Intersection over Union (IoU) [Mean, Std, Median, Min, Max]
  6. ROI Capture Rate / Crop Success Rate (% of crops containing ALL 13 landmarks, plus per-vertebra and per-landmark breakdown)
  7. Inference Latency (ms/image: preprocess, forward inference, postprocess, FPS)

Features for Fresh Cloud GPU Instances:
  - Automatically downloads model weights from MLflow run if missing locally.
  - Automatically prepares dataset splits and downloads images if running on a fresh instance.
  - Automatically logs publication metrics, tables, and visualization artifacts to MLflow.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.constants import LANDMARK_CLASSES
from src.utils.roi_utils import compute_roi_bbox, ROIBoundingBox


def ensure_test_dataset(
    test_img_dir: Path,
    test_npz_dir: Path,
    test_ids_file: Path,
    export_json: Path,
) -> None:
    """
    Checks if the test dataset is available locally.
    If missing (e.g. on a fresh cloud instance), automatically executes the
    exact data download, resize, label generation, and split pipeline.
    """
    load_dotenv()
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_npz_dir.mkdir(parents=True, exist_ok=True)

    existing_imgs = list(test_img_dir.glob("*.*"))
    existing_npzs = list(test_npz_dir.glob("*.npz"))

    if len(existing_imgs) >= 120 and len(existing_npzs) >= 120 and export_json.exists():
        return

    print("=" * 85)
    print("📥 Test dataset not found or incomplete. Preparing dataset from data source...")
    print("=" * 85)

    from src.data.preprocessing.download_data import download_export_and_images
    from src.data.preprocessing.resize_images import resize_images
    from src.data.preprocessing.generate_labels import generate_labels
    from src.data.preprocessing.split_dataset import split_dataset

    # Step 1: Download raw exports & images if needed
    raw_images_dir = Path("data/raw/images")
    raw_exports_dir = Path("data/raw/exports")
    if not raw_images_dir.exists() or len(list(raw_images_dir.glob("*.*"))) == 0:
        print("\n[Step 1/4] Downloading dataset from CVM Backend API...")
        api_url = os.getenv("CVM_API_URL")
        api_token = os.getenv("CVM_API_TOKEN")
        download_export_and_images(
            export_dir=str(raw_exports_dir),
            img_dir=str(raw_images_dir),
            api_url=api_url,
            api_token=api_token,
            skip_review_needed=True,
            skip_excluded=True,
            max_workers=10,
        )

    # Step 2: Resize images & coordinates
    processed_img_dir = Path("data/images")
    if not processed_img_dir.exists() or len(list(processed_img_dir.glob("*.*"))) == 0 or not export_json.exists():
        print("\n[Step 2/4] Resizing images to 640x640 canvas...")
        raw_json_files = sorted(raw_exports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        raw_json_path = str(raw_json_files[0]) if raw_json_files else "data/raw/exports/export.json"
        resize_images(
            img_dir=str(raw_images_dir),
            json_path=raw_json_path,
            out_img_dir=str(processed_img_dir),
            out_json_path=str(export_json),
            target_size=640,
        )

    # Step 3: Generate 2D Gaussian heatmaps and landmark labels
    processed_labels_dir = Path("data/labels")
    if not processed_labels_dir.exists() or len(list(processed_labels_dir.glob("*.npz"))) == 0:
        print("\n[Step 3/4] Generating anatomical landmark labels...")
        generate_labels(
            json_path=str(export_json),
            images_dir=str(processed_img_dir),
            output_dir=str(processed_labels_dir),
            sigma=4.0,
        )

    # Step 4: Split dataset according to fixed test IDs
    print("\n[Step 4/4] Partitioning test split using fixed test_image_ids.json...")
    split_dataset(
        images_dir=str(processed_img_dir),
        labels_dir=str(processed_labels_dir),
        output_dir="./dataset",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        fixed_test_ids_path=str(test_ids_file),
    )
    print("✅ Test dataset successfully prepared!\n")


def resolve_test_images_and_ground_truth(
    test_ids_json: Path,
    export_json: Optional[Path],
    test_img_dir: Path,
    test_npz_dir: Path,
    margin_ratio: float = 0.22,
    bottom_margin_ratio: float = 0.28,
    min_size: int = 180,
) -> List[Dict[str, Any]]:
    """
    Resolves the exact 120 test set items using test_image_ids.json and export.json,
    and computes ground-truth bounding boxes using the exact train-pipeline.py logic.
    """
    if not test_ids_json.exists():
        raise FileNotFoundError(f"Test IDs file not found: {test_ids_json}")

    with open(test_ids_json, "r") as f:
        test_ids_list = json.load(f)
    test_ids_set = set(test_ids_list)

    # Map task IDs to filenames via export.json if present
    task_id_to_stem = {}
    if export_json and export_json.exists():
        with open(export_json, "r") as f:
            export_data = json.load(f)
        for task in export_data:
            tid = task.get("id")
            if tid in test_ids_set:
                img_url = task.get("data", {}).get("img", "")
                if img_url:
                    task_id_to_stem[tid] = Path(img_url).stem

    samples = []
    missing_count = 0

    for tid in test_ids_list:
        target_stem = task_id_to_stem.get(tid)
        if not target_stem:
            target_stem = f"{int(tid):04d}" if str(tid).isdigit() else str(tid)

        img_matches = list(test_img_dir.glob(f"{target_stem}.*"))
        npz_matches = list(test_npz_dir.glob(f"{target_stem}.npz"))

        if not img_matches or not npz_matches:
            img_matches = list(test_img_dir.glob(f"{tid}.*"))
            npz_matches = list(test_npz_dir.glob(f"{tid}.npz"))

        if not img_matches or not npz_matches:
            missing_count += 1
            continue

        img_path = img_matches[0]
        npz_path = npz_matches[0]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            missing_count += 1
            continue
        img_h, img_w = img_bgr.shape[:2]

        data = np.load(npz_path)
        raw_coords = data["coords"]
        is_normalized = (raw_coords.max() <= 1.05 and raw_coords.min() >= -0.05)
        coords_px = raw_coords.copy()
        if is_normalized:
            coords_px[:, 0] *= img_w
            coords_px[:, 1] *= img_h

        # Compute ground truth ROI bounding box using EXACT train-pipeline.py logic
        roi: ROIBoundingBox = compute_roi_bbox(
            coords=raw_coords,
            img_w=img_w,
            img_h=img_h,
            margin_ratio=margin_ratio,
            bottom_margin_ratio=bottom_margin_ratio,
            min_size=min_size,
            use_anatomical_prior=True,
        )

        gx1 = max(0.0, float(roi.u1))
        gy1 = max(0.0, float(roi.v1))
        gx2 = min(float(img_w), float(roi.u2))
        gy2 = min(float(img_h), float(roi.v2))

        box_w = gx2 - gx1
        box_h = gy2 - gy1
        cx = (gx1 + gx2) / 2.0 / float(img_w)
        cy = (gy1 + gy2) / 2.0 / float(img_h)
        nw = box_w / float(img_w)
        nh = box_h / float(img_h)

        samples.append({
            "task_id": tid,
            "stem": target_stem,
            "img_path": img_path,
            "npz_path": npz_path,
            "img_w": img_w,
            "img_h": img_h,
            "coords_px": coords_px,
            "gt_box_px": (gx1, gy1, gx2, gy2),
            "gt_yolo_norm": (cx, cy, nw, nh),
        })

    if missing_count > 0:
        print(f"⚠️ Warning: Could not locate image/npz pairs for {missing_count} test items.")

    return samples


def prepare_yolo_test_yaml(
    samples: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Creates a YOLO dataset directory & yaml referencing exclusively the test set samples."""
    yolo_dir = output_dir / "yolo_test_dataset"
    img_dir = yolo_dir / "images"
    lbl_dir = yolo_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for s in samples:
        src_img: Path = s["img_path"]
        dst_img = img_dir / src_img.name

        if not dst_img.exists():
            try:
                os.symlink(src_img.resolve(), dst_img)
            except OSError:
                import shutil
                shutil.copy2(src_img, dst_img)

        cx, cy, nw, nh = s["gt_yolo_norm"]
        lbl_file = lbl_dir / f"{dst_img.stem}.txt"
        with open(lbl_file, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    yaml_data = {
        "path": str(yolo_dir.resolve()),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": {0: "cervical_spine"},
    }

    yaml_path = yolo_dir / "test_120.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    return yaml_path


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter

    return inter / union if union > 0.0 else 0.0


def run_yolo_evaluation(
    weights_path: str = "artifacts/yolo_stage1.pt",
    run_id: Optional[str] = "9ce919cc321d423fac3aa0929b14872f",
    test_ids_file: str = "dataset/test_image_ids.json",
    export_json: str = "data/exports/export.json",
    test_img_dir: str = "dataset/test/images",
    test_npz_dir: str = "dataset/test/labels",
    output_dir: str = "evaluation/yolo_evaluation",
    conf_threshold: float = 0.25,
    img_size: int = 640,
    device: str = "",
    save_visualizations: int = 10,
    log_to_mlflow: bool = False,
    auto_download_data: bool = True,
) -> Dict[str, Any]:
    """Runs complete publication-grade evaluation of YOLO model on the test set."""
    load_dotenv()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Download weights from MLflow if missing
    if not os.path.exists(weights_path) and run_id:
        print(f"📥 Model checkpoint not found at '{weights_path}'. Fetching from MLflow run '{run_id}'...")
        from scripts.download_checkpoint import download_checkpoint_from_mlflow
        download_checkpoint_from_mlflow(
            run_id=run_id,
            artifact_path="weights/best.pt",
            dest_path=weights_path,
        )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    # 2. Check and auto-prepare dataset if running on a fresh instance
    if auto_download_data:
        ensure_test_dataset(
            test_img_dir=Path(test_img_dir),
            test_npz_dir=Path(test_npz_dir),
            test_ids_file=Path(test_ids_file),
            export_json=Path(export_json),
        )

    from ultralytics import YOLO

    print("=" * 85)
    print(f"🔍 CVM STAGE 1 YOLO EVALUATION (TEST SET)")
    print(f"   Model Weights:     {weights_path}")
    if run_id:
        print(f"   Source Run ID:     {run_id}")
    print(f"   Confidence Thresh: {conf_threshold}")
    print(f"   Input Size:        {img_size}x{img_size}")
    print(f"   Output Directory:  {out_path.resolve()}")
    print("=" * 85)

    # 3. Resolve Test Set Samples with Exact Ground Truth Generation
    print("\n[1/4] Preparing 120 Test Set Samples and Ground Truth ROIs...")
    samples = resolve_test_images_and_ground_truth(
        test_ids_json=Path(test_ids_file),
        export_json=Path(export_json) if os.path.exists(export_json) else None,
        test_img_dir=Path(test_img_dir),
        test_npz_dir=Path(test_npz_dir),
        margin_ratio=0.22,
        bottom_margin_ratio=0.28,
        min_size=180,
    )
    print(f"✅ Resolved {len(samples)} valid test image pairs.")

    # 4. Compute Ultralytics Detection Metrics (mAP50, mAP50-95, Precision, Recall)
    print("\n[2/4] Running Official Ultralytics Validator on Test Set...")
    test_yaml = prepare_yolo_test_yaml(samples, out_path)
    model = YOLO(weights_path)

    val_args = {
        "data": str(test_yaml),
        "split": "test",
        "imgsz": img_size,
        "conf": conf_threshold,
        "verbose": False,
        "plots": False,
    }
    if device:
        val_args["device"] = device

    val_res = model.val(**val_args)

    map50 = float(val_res.box.map50)
    map50_95 = float(val_res.box.map)
    precision = float(val_res.box.mp)
    recall = float(val_res.box.mr)
    f1_score = float(val_res.box.f1[0]) if len(val_res.box.f1) > 0 else (
        2 * precision * recall / (precision + recall + 1e-8)
    )

    val_speed = val_res.speed
    pre_ms = float(val_speed.get("preprocess", 0.0))
    inf_ms = float(val_speed.get("inference", 0.0))
    post_ms = float(val_speed.get("postprocess", 0.0))
    total_val_ms = pre_ms + inf_ms + post_ms

    # 5. Detailed Per-Image Analysis (IoU, ROI Capture Rate, Per-Landmark Containment)
    print("\n[3/4] Evaluating Detailed IoU Overlap & Landmark Capture Rates...")
    ious: List[float] = []
    capture_all_count = 0
    per_landmark_captured = np.zeros(13, dtype=int)
    vertebra_captured = {"C2": 0, "C3": 0, "C4": 0}
    concavity_captured = {"C2_IC": 0, "C3_IC": 0, "C4_IC": 0}
    inferior_corners_captured = {"C3_AI": 0, "C3_PI": 0, "C4_AI": 0, "C4_PI": 0}

    c2_indices = [0, 1, 2]
    c3_indices = [3, 4, 5, 6, 7]
    c4_indices = [8, 9, 10, 11, 12]

    prediction_records = []
    e2e_latencies: List[float] = []

    # Warmup
    dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(source=dummy, imgsz=img_size, verbose=False)

    vis_dir = out_path / "visualizations"
    if save_visualizations > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for idx, s in enumerate(samples):
        img_bgr = cv2.imread(str(s["img_path"]))
        gx1, gy1, gx2, gy2 = s["gt_box_px"]
        coords_px = s["coords_px"]

        t_start = time.perf_counter()
        pred_res = model.predict(
            source=img_bgr,
            conf=conf_threshold,
            imgsz=img_size,
            device=device or None,
            verbose=False,
        )
        t_end = time.perf_counter()
        e2e_latencies.append((t_end - t_start) * 1000.0)

        boxes = pred_res[0].boxes
        if boxes is not None and len(boxes) > 0:
            best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            conf = float(boxes.conf[best_idx].item())
            px1, py1, px2, py2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        else:
            px1, py1, px2, py2 = 0.0, 0.0, 0.0, 0.0
            conf = 0.0

        iou = compute_iou((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
        ious.append(iou)

        in_box = []
        for k in range(13):
            lx, ly = coords_px[k]
            captured = (px1 <= lx <= px2) and (py1 <= ly <= py2)
            if captured:
                per_landmark_captured[k] += 1
            in_box.append(captured)

        all_captured = bool(all(in_box))
        if all_captured:
            capture_all_count += 1

        c2_ok = bool(all(in_box[k] for k in c2_indices))
        c3_ok = bool(all(in_box[k] for k in c3_indices))
        c4_ok = bool(all(in_box[k] for k in c4_indices))
        if c2_ok: vertebra_captured["C2"] += 1
        if c3_ok: vertebra_captured["C3"] += 1
        if c4_ok: vertebra_captured["C4"] += 1

        if in_box[1]: concavity_captured["C2_IC"] += 1
        if in_box[6]: concavity_captured["C3_IC"] += 1
        if in_box[11]: concavity_captured["C4_IC"] += 1

        if in_box[7]: inferior_corners_captured["C3_AI"] += 1
        if in_box[5]: inferior_corners_captured["C3_PI"] += 1
        if in_box[12]: inferior_corners_captured["C4_AI"] += 1
        if in_box[10]: inferior_corners_captured["C4_PI"] += 1

        prediction_records.append({
            "task_id": s["task_id"],
            "stem": s["stem"],
            "confidence": conf,
            "iou": iou,
            "all_13_captured": all_captured,
            "c2_captured": c2_ok,
            "c3_captured": c3_ok,
            "c4_captured": c4_ok,
            "pred_box_xyxy": [round(px1, 1), round(py1, 1), round(px2, 1), round(py2, 1)],
            "gt_box_xyxy": [round(gx1, 1), round(gy1, 1), round(gx2, 1), round(gy2, 1)],
        })

        if save_visualizations > 0 and idx < save_visualizations:
            vis = img_bgr.copy()
            cv2.rectangle(vis, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 200, 0), 2)
            cv2.rectangle(vis, (int(px1), int(py1)), (int(px2), int(py2)), (255, 180, 0), 2)
            for k in range(13):
                lx, ly = int(round(coords_px[k, 0])), int(round(coords_px[k, 1]))
                color = (0, 255, 0) if in_box[k] else (0, 0, 255)
                cv2.circle(vis, (lx, ly), 3, color, -1)

            cv2.putText(
                vis,
                f"IoU: {iou*100:.1f}% | Conf: {conf:.2f} | Captured: {'YES' if all_captured else 'NO'}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imwrite(str(vis_dir / f"test_{s['stem']}_roi_eval.png"), vis)

    N = len(samples)
    iou_arr = np.array(ious)
    mean_iou = float(np.mean(iou_arr))
    sd_iou = float(np.std(iou_arr))
    median_iou = float(np.median(iou_arr))
    min_iou = float(np.min(iou_arr))
    max_iou = float(np.max(iou_arr))
    p25_iou = float(np.percentile(iou_arr, 25))
    p75_iou = float(np.percentile(iou_arr, 75))

    roi_capture_rate = (capture_all_count / N) * 100.0
    mean_latency_ms = float(np.mean(e2e_latencies))
    std_latency_ms = float(np.std(e2e_latencies))
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    # 6. Compile Final Metrics Dictionary
    results_summary = {
        "model": {
            "weights": str(weights_path),
            "run_id": run_id,
            "input_resolution": img_size,
            "conf_threshold": conf_threshold,
            "test_samples": N,
        },
        "detection_metrics": {
            "mAP_50": round(map50 * 100, 2),
            "mAP_50_95": round(map50_95 * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1_score * 100, 2),
        },
        "iou_metrics": {
            "mean_iou_pct": round(mean_iou * 100, 2),
            "std_iou_pct": round(sd_iou * 100, 2),
            "median_iou_pct": round(median_iou * 100, 2),
            "iqr_iou_pct": [round(p25_iou * 100, 2), round(p75_iou * 100, 2)],
            "min_iou_pct": round(min_iou * 100, 2),
            "max_iou_pct": round(max_iou * 100, 2),
        },
        "clinical_roi_capture": {
            "all_13_landmarks_capture_rate_pct": round(roi_capture_rate, 2),
            "all_13_captured_count": capture_all_count,
            "vertebra_capture_rates_pct": {
                "C2": round((vertebra_captured["C2"] / N) * 100, 2),
                "C3": round((vertebra_captured["C3"] / N) * 100, 2),
                "C4": round((vertebra_captured["C4"] / N) * 100, 2),
            },
            "critical_concavities_capture_pct": {
                "C2_IC": round((concavity_captured["C2_IC"] / N) * 100, 2),
                "C3_IC": round((concavity_captured["C3_IC"] / N) * 100, 2),
                "C4_IC": round((concavity_captured["C4_IC"] / N) * 100, 2),
            },
            "critical_inferior_corners_capture_pct": {
                "C3_AI": round((inferior_corners_captured["C3_AI"] / N) * 100, 2),
                "C3_PI": round((inferior_corners_captured["C3_PI"] / N) * 100, 2),
                "C4_AI": round((inferior_corners_captured["C4_AI"] / N) * 100, 2),
                "C4_PI": round((inferior_corners_captured["C4_PI"] / N) * 100, 2),
            },
            "per_landmark_capture_pct": {
                LANDMARK_CLASSES[k]: round((per_landmark_captured[k] / N) * 100, 2)
                for k in range(13)
            },
        },
        "inference_speed": {
            "ultralytics_preprocess_ms": round(pre_ms, 2),
            "ultralytics_inference_ms": round(inf_ms, 2),
            "ultralytics_postprocess_ms": round(post_ms, 2),
            "ultralytics_total_pipeline_ms": round(total_val_ms, 2),
            "e2e_total_latency_mean_ms": round(mean_latency_ms, 2),
            "e2e_total_latency_std_ms": round(std_latency_ms, 2),
            "throughput_fps": round(fps, 1),
        },
    }

    # Save outputs
    json_path = out_path / "test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    csv_path = out_path / "test_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "stem", "confidence", "iou", "all_13_captured", "c2_captured", "c3_captured", "c4_captured", "pred_box_xyxy", "gt_box_xyxy"])
        for r in prediction_records:
            writer.writerow([
                r["task_id"],
                r["stem"],
                f"{r['confidence']:.4f}",
                f"{r['iou']:.4f}",
                r["all_13_captured"],
                r["c2_captured"],
                r["c3_captured"],
                r["c4_captured"],
                ";".join(map(str, r["pred_box_xyxy"])),
                ";".join(map(str, r["gt_box_xyxy"])),
            ])

    report_text = f"""================================================================================
           CVM STAGE 1 YOLO CERVICAL SPINE DETECTOR EVALUATION REPORT
================================================================================
Evaluation Dataset:        Official Test Set (N = {N} lateral cephalograms)
Model Checkpoint:          {weights_path}
Source MLflow Run:         {run_id or 'Local'}
Resolution:                {img_size} x {img_size}
Confidence Threshold:      {conf_threshold}

--------------------------------------------------------------------------------
1. OBJECT DETECTION METRICS (Standard Object Detection Thresholds)
--------------------------------------------------------------------------------
• Mean Average Precision @ 0.50 (mAP@50):        {results_summary['detection_metrics']['mAP_50']:.2f}%
• Mean Average Precision @ 0.50:0.95 (mAP@50-95):  {results_summary['detection_metrics']['mAP_50_95']:.2f}%
• Precision (Positive Predictive Value):          {results_summary['detection_metrics']['precision']:.2f}%
• Recall (Sensitivity / True Positive Rate):      {results_summary['detection_metrics']['recall']:.2f}%
• F1-Score:                                      {results_summary['detection_metrics']['f1_score']:.2f}%

--------------------------------------------------------------------------------
2. BOUNDING BOX OVERLAP & REGRESSION METRICS
--------------------------------------------------------------------------------
• Mean Intersection over Union (IoU):             {results_summary['iou_metrics']['mean_iou_pct']:.2f}% ± {results_summary['iou_metrics']['std_iou_pct']:.2f}%
• Median Intersection over Union (IoU):           {results_summary['iou_metrics']['median_iou_pct']:.2f}%
• Interquartile Range (IQR 25th - 75th):          [{results_summary['iou_metrics']['iqr_iou_pct'][0]:.2f}%, {results_summary['iou_metrics']['iqr_iou_pct'][1]:.2f}%]
• Minimum IoU:                                    {results_summary['iou_metrics']['min_iou_pct']:.2f}%
• Maximum IoU:                                    {results_summary['iou_metrics']['max_iou_pct']:.2f}%

--------------------------------------------------------------------------------
3. CLINICAL ROI CAPTURE RATE (Downstream Swin Safeguard)
--------------------------------------------------------------------------------
• Full ROI Capture Rate (All 13 Landmarks):       {results_summary['clinical_roi_capture']['all_13_landmarks_capture_rate_pct']:.2f}% ({capture_all_count}/{N})
• C2 Vertebra Complete Containment:              {results_summary['clinical_roi_capture']['vertebra_capture_rates_pct']['C2']:.2f}%
• C3 Vertebra Complete Containment:              {results_summary['clinical_roi_capture']['vertebra_capture_rates_pct']['C3']:.2f}%
• C4 Vertebra Complete Containment:              {results_summary['clinical_roi_capture']['vertebra_capture_rates_pct']['C4']:.2f}%

Key Diagnostic Landmarks Containment:
  - C2 Inferior Concavity (C2_IC):               {results_summary['clinical_roi_capture']['critical_concavities_capture_pct']['C2_IC']:.2f}%
  - C3 Inferior Concavity (C3_IC):               {results_summary['clinical_roi_capture']['critical_concavities_capture_pct']['C3_IC']:.2f}%
  - C4 Inferior Concavity (C4_IC):               {results_summary['clinical_roi_capture']['critical_concavities_capture_pct']['C4_IC']:.2f}%
  - C3 Anterior-Inferior Corner (C3_AI):         {results_summary['clinical_roi_capture']['critical_inferior_corners_capture_pct']['C3_AI']:.2f}%
  - C4 Anterior-Inferior Corner (C4_AI):         {results_summary['clinical_roi_capture']['critical_inferior_corners_capture_pct']['C4_AI']:.2f}%

--------------------------------------------------------------------------------
4. INFERENCE LATENCY & THROUGHPUT
--------------------------------------------------------------------------------
• End-to-End Latency per Image:                  {results_summary['inference_speed']['e2e_total_latency_mean_ms']:.2f} ± {results_summary['inference_speed']['e2e_total_latency_std_ms']:.2f} ms
• Model Forward Inference:                       {results_summary['inference_speed']['ultralytics_inference_ms']:.2f} ms
• Pipeline Throughput:                           {results_summary['inference_speed']['throughput_fps']:.1f} frames/sec (FPS)
================================================================================
"""

    report_path = out_path / "summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"✅ Saved metrics JSON:     {json_path}")
    print(f"✅ Saved prediction CSV:   {csv_path}")
    print(f"✅ Saved summary report:   {report_path}")

    # 7. MLflow Logging
    if log_to_mlflow:
        try:
            import mlflow
            uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.amirlogedixyz.ir")
            user = os.getenv("MLFLOW_TRACKING_USERNAME")
            pwd = os.getenv("MLFLOW_TRACKING_PASSWORD")
            if user: os.environ["MLFLOW_TRACKING_USERNAME"] = user
            if pwd: os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd
            mlflow.set_tracking_uri(uri)
            exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "cvm-swin-gcn")
            mlflow.set_experiment(exp_name)

            client = mlflow.tracking.MlflowClient()

            # A) Directly log metrics & artifacts to original training run (if run_id provided)
            if run_id:
                try:
                    print(f"--> Logging test metrics directly to source MLflow Run: {run_id}...")
                    client.log_metric(run_id, "test/mAP50", map50)
                    client.log_metric(run_id, "test/mAP50_95", map50_95)
                    client.log_metric(run_id, "test/precision", precision)
                    client.log_metric(run_id, "test/recall", recall)
                    client.log_metric(run_id, "test/f1_score", f1_score)
                    client.log_metric(run_id, "test/mean_iou", mean_iou)
                    client.log_metric(run_id, "test/median_iou", median_iou)
                    client.log_metric(run_id, "test/roi_capture_rate", roi_capture_rate)
                    client.log_artifact(run_id, str(json_path), artifact_path="test_evaluation")
                    client.log_artifact(run_id, str(csv_path), artifact_path="test_evaluation")
                    client.log_artifact(run_id, str(report_path), artifact_path="test_evaluation")
                    print(f"✅ Logged test artifacts directly under 'test_evaluation/' in run {run_id}")
                except Exception as run_err:
                    print(f"ℹ️ Could not append directly to run {run_id}: {run_err}")

            # B) Create a dedicated visible evaluation run in the experiment
            exp = client.get_experiment_by_name(exp_name)
            exp_id = exp.experiment_id if exp else "1"
            eval_run = client.create_run(
                experiment_id=exp_id,
                tags={
                    "mlflow.runName": f"eval_yolo_test_{run_id[:8] if run_id else 'stage1'}",
                    "parent_run_id": run_id or "",
                    "evaluation_type": "yolo_cervical_roi_test",
                },
            )
            eval_id = eval_run.info.run_id

            client.log_param(eval_id, "weights", weights_path)
            client.log_param(eval_id, "source_run_id", run_id or "none")
            client.log_param(eval_id, "test_samples", N)
            client.log_param(eval_id, "conf_threshold", conf_threshold)

            client.log_metric(eval_id, "mAP50", map50)
            client.log_metric(eval_id, "mAP50_95", map50_95)
            client.log_metric(eval_id, "precision", precision)
            client.log_metric(eval_id, "recall", recall)
            client.log_metric(eval_id, "f1_score", f1_score)
            client.log_metric(eval_id, "mean_iou", mean_iou)
            client.log_metric(eval_id, "median_iou", median_iou)
            client.log_metric(eval_id, "roi_capture_rate_pct", roi_capture_rate)
            client.log_metric(eval_id, "latency_mean_ms", mean_latency_ms)
            client.log_metric(eval_id, "throughput_fps", fps)

            client.log_artifact(eval_id, str(json_path))
            client.log_artifact(eval_id, str(csv_path))
            client.log_artifact(eval_id, str(report_path))
            if vis_dir.exists():
                for vis_f in vis_dir.glob("*.png"):
                    client.log_artifact(eval_id, str(vis_f), artifact_path="visualizations")

            client.set_terminated(eval_id, status="FINISHED")
            print(f"✅ Successfully created dedicated MLflow Evaluation Run: {eval_id}")

        except Exception as ml_err:
            print(f"⚠️ Warning: Could not log to MLflow: {ml_err}")

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 YOLO on 120 CVM Test Set Images")
    parser.add_argument("--weights", default="artifacts/yolo_stage1.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--run-id", default="9ce919cc321d423fac3aa0929b14872f", help="MLflow run ID to fetch weights if missing")
    parser.add_argument("--test-ids", default="dataset/test_image_ids.json", help="Path to test_image_ids.json")
    parser.add_argument("--export-json", default="data/exports/export.json", help="Path to export.json")
    parser.add_argument("--test-img-dir", default="dataset/test/images", help="Test images directory")
    parser.add_argument("--test-npz-dir", default="dataset/test/labels", help="Test labels directory (.npz)")
    parser.add_argument("--output-dir", default="evaluation/yolo_evaluation", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--img-size", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", default="", help="Device: cuda, mps, cpu, or GPU index (e.g. 0)")
    parser.add_argument("--visualize", type=int, default=10, help="Number of visualization overlays to save")
    parser.add_argument("--log-mlflow", action="store_true", help="Log metrics to MLflow")
    parser.add_argument("--skip-auto-download", action="store_true", help="Do not auto-download data if missing")

    args = parser.parse_args()

    # Fallback weights check
    if not os.path.exists(args.weights) and os.path.exists("artifacts/yolo_stage1_9ce919.pt"):
        args.weights = "artifacts/yolo_stage1_9ce919.pt"

    run_yolo_evaluation(
        weights_path=args.weights,
        run_id=args.run_id,
        test_ids_file=args.test_ids,
        export_json=args.export_json,
        test_img_dir=args.test_img_dir,
        test_npz_dir=args.test_npz_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        img_size=args.img_size,
        device=args.device,
        save_visualizations=args.visualize,
        log_to_mlflow=args.log_mlflow,
        auto_download_data=not args.skip_auto_download,
    )
