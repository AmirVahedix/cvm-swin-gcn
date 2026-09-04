#!/usr/bin/env python3
"""
Dataset Validation and Geometric Consistency Verification Script.

Executes the data preprocessing pipeline up to `generate_labels()`, then performs:
1. Label integrity verification (exactly 13 landmarks, 1 of each class, no duplicates/missing/extras).
2. Spatial / distance constraint verification (18 anatomical rules, including lordosis handling for Rule 13).
3. Detailed terminal reporting.
4. Export of violating images and generation of a rich interactive canvas Web UI (index.html).
"""

import sys
import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlparse, parse_qs

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS
from src.data.preprocessing.download_data import download_export_and_images
from src.data.preprocessing.resize_images import resize_images
from src.data.preprocessing.generate_labels import generate_labels

# The 18 Anatomical Geometric Rules
# Direction convention in standard image coordinates:
# (0, 0) is TOP-LEFT.
# x increases to the RIGHT. Therefore "A is left of B"  => x_A < x_B
# y increases DOWNWARDS.     Therefore "A is bottom of B" => y_A > y_B
GEOMETRIC_RULES = [
    # Horizontal constraints (left of)
    {"id": 1,  "type": "horizontal", "a": "C2_PI", "b": "C2_IC", "desc": "C2_PI => must be always at the left of C2_IC"},
    {"id": 2,  "type": "horizontal", "a": "C2_IC", "b": "C2_AI", "desc": "C2_IC => must be always at the left of C2_AI"},
    {"id": 3,  "type": "horizontal", "a": "C3_PI", "b": "C3_IC", "desc": "C3_PI => must be always at the left of C3_IC"},
    {"id": 4,  "type": "horizontal", "a": "C3_IC", "b": "C3_AI", "desc": "C3_IC => must be always at the left of C3_AI"},
    {"id": 5,  "type": "horizontal", "a": "C4_PI", "b": "C4_IC", "desc": "C4_PI => must be always at the left of C4_IC"},
    {"id": 6,  "type": "horizontal", "a": "C4_IC", "b": "C4_AI", "desc": "C4_IC => must be always at the left of C4_AI"},
    {"id": 7,  "type": "horizontal", "a": "C3_PS", "b": "C3_AS", "desc": "C3_PS => must be always at the left of C3_AS"},
    {"id": 8,  "type": "horizontal", "a": "C4_PS", "b": "C4_AS", "desc": "C4_PS => must be always at the left of C4_AS"},

    # Vertical constraints (bottom of)
    {"id": 9,  "type": "vertical",   "a": "C3_AS", "b": "C2_AI", "desc": "C3_AS => must be always at the bottom of C2_AI"},
    {"id": 10, "type": "vertical",   "a": "C3_PS", "b": "C2_PI", "desc": "C3_PS => must be always at the bottom of C2_PI"},
    {"id": 11, "type": "vertical",   "a": "C3_PI", "b": "C3_PS", "desc": "C3_PI => must be always at the bottom of C3_PS"},
    {"id": 12, "type": "vertical",   "a": "C3_AI", "b": "C3_AS", "desc": "C3_AI => must be always at the bottom of C3_AS"},
    {"id": 13, "type": "vertical",   "a": "C4_PS", "b": "C3_PI", "desc": "C4_PS => must be always at the bottom of C3_PI"},
    {"id": 14, "type": "vertical",   "a": "C4_AS", "b": "C3_AI", "desc": "C4_AS => must be always at the bottom of C3_AI"},
    {"id": 15, "type": "vertical",   "a": "C4_PI", "b": "C4_PS", "desc": "C4_PI => must be always at the bottom of C4_PS"},
    {"id": 16, "type": "vertical",   "a": "C4_AI", "b": "C4_AS", "desc": "C4_AI => must be always at the bottom of C4_AS"},
    {"id": 17, "type": "vertical",   "a": "C3_IC", "b": "C2_IC", "desc": "C3_IC => must be always at the bottom of C2_IC"},
    {"id": 18, "type": "vertical",   "a": "C4_IC", "b": "C3_IC", "desc": "C4_IC => must be always at the bottom of C3_IC"},
]

# Distinct palette per landmark
LANDMARK_COLORS = {
    "C2_PI": "#FF5722",  # Deep Orange
    "C2_IC": "#FF9800",  # Orange
    "C2_AI": "#FFC107",  # Amber
    "C3_PS": "#4CAF50",  # Green
    "C3_AS": "#8BC34A",  # Light Green
    "C3_PI": "#009688",  # Teal
    "C3_IC": "#00BCD4",  # Cyan
    "C3_AI": "#03A9F4",  # Light Blue
    "C4_PS": "#3F51B5",  # Indigo
    "C4_AS": "#9C27B0",  # Purple
    "C4_PI": "#E91E63",  # Pink
    "C4_IC": "#F44336",  # Red
    "C4_AI": "#795548",  # Brown
}


def extract_clean_filename(record: dict) -> str:
    """Extracts cleaned filename matching resize_images and generate_labels logic."""
    raw_path = record.get("file_upload") or record.get("data", {}).get("img")
    if not raw_path:
        return ""
    parsed_url = urlparse(raw_path)
    query_params = parse_qs(parsed_url.query)
    if "d" in query_params:
        filename = os.path.basename(query_params["d"][0])
    else:
        filename = os.path.basename(parsed_url.path)

    if "-" in filename and len(filename.split("-")[0]) == 8:
        clean_filename = "-".join(filename.split("-")[1:])
    else:
        clean_filename = filename
    return clean_filename


def extract_landmarks(record: dict, img_w: int = 640, img_h: int = 640) -> Tuple[Dict[str, List[Dict[str, float]]], List[str]]:
    """
    Extracts all keypoints from a Label Studio record.
    Returns:
      landmarks: dict mapping label_name -> list of point dicts with {x, y, norm_x, norm_y}
      all_labels: list of all raw label occurrences in order
    """
    landmarks = {}
    all_labels = []

    annotations = record.get("annotations", [])
    if not annotations:
        return landmarks, all_labels

    for item in annotations[0].get("result", []):
        if item.get("type") == "keypointlabels":
            val = item.get("value", {})
            labels = val.get("keypointlabels", [])
            if not labels:
                continue

            label_name = labels[0]
            all_labels.append(label_name)

            orig_w = item.get("original_width", img_w)
            orig_h = item.get("original_height", img_h)

            norm_x = val.get("x", 0) / 100.0
            norm_y = val.get("y", 0) / 100.0
            px_x = (val.get("x", 0) * orig_w) / 100.0
            px_y = (val.get("y", 0) * orig_h) / 100.0

            point_data = {
                "x": round(px_x, 2),
                "y": round(px_y, 2),
                "norm_x": round(norm_x, 4),
                "norm_y": round(norm_y, 4),
            }

            if label_name not in landmarks:
                landmarks[label_name] = []
            landmarks[label_name].append(point_data)

    return landmarks, all_labels


def validate_single_record(record: dict, rule13_mode: str = "both") -> dict:
    """
    Validates a single task record:
    1. Label uniqueness and completeness (exactly 13 landmarks from LANDMARK_CLASSES).
    2. 18 relative spatial rules.
    """
    clean_filename = extract_clean_filename(record)
    landmarks, all_labels = extract_landmarks(record)

    # 1. Label completeness and uniqueness verification
    label_errors = []
    missing_labels = [cls for cls in LANDMARK_CLASSES if cls not in landmarks]
    duplicate_labels = {cls: len(pts) for cls, pts in landmarks.items() if len(pts) > 1}
    unknown_labels = [lbl for lbl in all_labels if lbl not in LANDMARK_CLASSES]

    if len(all_labels) != NUM_LANDMARKS:
        label_errors.append(f"Total keypoints count is {len(all_labels)} (expected {NUM_LANDMARKS})")
    if missing_labels:
        label_errors.append(f"Missing landmarks: {', '.join(missing_labels)}")
    if duplicate_labels:
        dup_str = ", ".join([f"{k} ({v}x)" for k, v in duplicate_labels.items()])
        label_errors.append(f"Duplicate landmarks: {dup_str}")
    if unknown_labels:
        label_errors.append(f"Unknown landmark classes: {', '.join(unknown_labels)}")

    # 2. Geometric rules verification
    rule_results = []
    rule_violations = []

    # Map each landmark to points list
    for rule in GEOMETRIC_RULES:
        rule_id = rule["id"]
        a_name = rule["a"]
        b_name = rule["b"]
        rule_type = rule["type"]
        desc = rule["desc"]

        pts_a = landmarks.get(a_name, [])
        pts_b = landmarks.get(b_name, [])

        if not pts_a or not pts_b:
            status = "SKIPPED_MISSING"
            passed = False
            delta = None
            msg = f"Cannot evaluate: missing landmark ({a_name if not pts_a else b_name})"
        elif len(pts_a) > 1 or len(pts_b) > 1:
            status = "SKIPPED_DUPLICATE"
            passed = False
            delta = None
            dup_target = a_name if len(pts_a) > 1 else b_name
            msg = f"Cannot evaluate strictly: {dup_target} has {len(landmarks[dup_target])} duplicate annotations"
        else:
            pt_a = pts_a[0]
            pt_b = pts_b[0]

            if rule_type == "horizontal":
                # A left of B => x_a < x_b (diff = x_b - x_a > 0)
                delta = round(pt_b["x"] - pt_a["x"], 2)
                passed = pt_a["x"] < pt_b["x"]
                if not passed:
                    msg = f"VIOLATION: {a_name} (x={pt_a['x']}) is NOT left of {b_name} (x={pt_b['x']}), delta={delta}px"
                else:
                    msg = f"OK: {a_name} is left of {b_name} by {delta}px"
            else:
                # A bottom of B => y_a > y_b (diff = y_a - y_b > 0)
                delta = round(pt_a["y"] - pt_b["y"], 2)
                passed = pt_a["y"] > pt_b["y"]
                if not passed:
                    msg = f"VIOLATION: {a_name} (y={pt_a['y']}) is NOT bottom of {b_name} (y={pt_b['y']}), delta={delta}px"
                else:
                    msg = f"OK: {a_name} is bottom of {b_name} by {delta}px"

        rule_res = {
            "rule_id": rule_id,
            "desc": desc,
            "passed": passed,
            "delta": delta,
            "message": msg,
            "landmark_a": a_name,
            "landmark_b": b_name,
        }
        rule_results.append(rule_res)
        if not passed:
            rule_violations.append(rule_res)

    # 3. Special handling & analysis for Rule 13 (Lordosis / Tilt concern)
    rule13_info = {}
    unique_points = {k: v[0] for k, v in landmarks.items() if len(v) == 1}
    if "C4_PS" in unique_points and "C3_PI" in unique_points and "C3_PS" in unique_points:
        y_c4_ps = unique_points["C4_PS"]["y"]
        y_c3_pi = unique_points["C3_PI"]["y"]
        y_c3_ps = unique_points["C3_PS"]["y"]

        strict_pass = y_c4_ps > y_c3_pi
        strict_delta = round(y_c4_ps - y_c3_pi, 2)
        safer_pass = y_c4_ps > y_c3_ps
        safer_delta = round(y_c4_ps - y_c3_ps, 2)

        # C3 & C4 centroids
        c3_keys = ["C3_PS", "C3_AS", "C3_PI", "C3_IC", "C3_AI"]
        c4_keys = ["C4_PS", "C4_AS", "C4_PI", "C4_IC", "C4_AI"]
        c3_pts = [unique_points[k]["y"] for k in c3_keys if k in unique_points]
        c4_pts = [unique_points[k]["y"] for k in c4_keys if k in unique_points]

        centroid_pass = False
        centroid_delta = None
        if len(c3_pts) == 5 and len(c4_pts) == 5:
            c3_mean_y = sum(c3_pts) / 5.0
            c4_mean_y = sum(c4_pts) / 5.0
            centroid_pass = c4_mean_y > c3_mean_y
            centroid_delta = round(c4_mean_y - c3_mean_y, 2)

        rule13_info = {
            "strict_pass": strict_pass,
            "strict_delta": strict_delta,
            "safer_pass": safer_pass,
            "safer_delta": safer_delta,
            "centroid_pass": centroid_pass,
            "centroid_delta": centroid_delta,
            "is_lordosis_tilt_case": (not strict_pass and safer_pass),
        }

    # Determine overall status
    has_label_error = len(label_errors) > 0

    # Rule 13 consideration in overall geometric validity
    # If rule13_mode == "safer", we don't count strict Rule 13 failure as violation IF safer passes
    effective_violations = []
    for rv in rule_violations:
        if rv["rule_id"] == 13:
            if rule13_mode == "safer":
                if rule13_info.get("safer_pass", False):
                    continue
        effective_violations.append(rv)

    is_valid = (not has_label_error) and (len(effective_violations) == 0)

    # Convert landmarks for serialization
    serializable_landmarks = {}
    for k, v in landmarks.items():
        serializable_landmarks[k] = v[0] if len(v) == 1 else v

    return {
        "filename": clean_filename,
        "is_valid": is_valid,
        "has_label_error": has_label_error,
        "label_errors": label_errors,
        "missing_labels": missing_labels,
        "duplicate_labels": duplicate_labels,
        "unknown_labels": unknown_labels,
        "rule_results": rule_results,
        "rule_violations": rule_violations,
        "effective_violations": effective_violations,
        "rule13_info": rule13_info,
        "landmarks": serializable_landmarks,
    }


def print_terminal_report(results: List[dict], rule13_mode: str = "both") -> None:
    """Prints a structured, high-visibility validation report in the terminal."""
    total_count = len(results)
    valid_count = sum(1 for r in results if r["is_valid"])
    invalid_count = total_count - valid_count

    label_error_count = sum(1 for r in results if r["has_label_error"])
    geo_violation_count = sum(1 for r in results if len(r["rule_violations"]) > 0)
    effective_geo_count = sum(1 for r in results if len(r["effective_violations"]) > 0)

    # Per-rule breakdown
    rule_fail_counts = {r["id"]: 0 for r in GEOMETRIC_RULES}
    rule_fail_samples = {r["id"]: [] for r in GEOMETRIC_RULES}

    for res in results:
        for v in res["rule_violations"]:
            rid = v["rule_id"]
            rule_fail_counts[rid] += 1
            rule_fail_samples[rid].append(res["filename"])

    lordosis_cases = [r for r in results if r.get("rule13_info", {}).get("is_lordosis_tilt_case", False)]

    print("\n" + "=" * 80)
    print("                    DATASET VALIDATION & VERIFICATION REPORT                    ")
    print("=" * 80)
    print(f"Total Samples Analyzed:            {total_count}")
    print(f"Valid Samples (Passed All Checks): {valid_count} ({(valid_count/total_count)*100:.1f}%)")
    print(f"Violating Samples:                 {invalid_count} ({(invalid_count/total_count)*100:.1f}%)")
    print("-" * 80)
    print("BREAKDOWN BY CATEGORY:")
    print(f"  • Label Integrity Violations (Missing/Duplicate/Extra): {label_error_count} samples")
    print(f"  • Geometric Rule Violations (Raw All 18 Rules):         {geo_violation_count} samples")
    if rule13_mode == "safer":
        print(f"  • Geometric Violations with Safer Rule 13 Mode:         {effective_geo_count} samples")
    print("-" * 80)

    # 1. Label Integrity Detail
    if label_error_count > 0:
        print("\n❌ SAMPLES WITH LABEL INTEGRITY ERRORS:")
        for res in results:
            if res["has_label_error"]:
                print(f"  [{res['filename']}]:")
                for err in res["label_errors"]:
                    print(f"    - {err}")

    # 2. Rule 18 Breakdown Table
    print("\n📋 GEOMETRIC RULES COMPLIANCE TABLE:")
    print(f"{'Rule':<6} | {'Direction':<11} | {'Constraint Description':<46} | {'Failures':<8} | {'Status':<6}")
    print("-" * 80)
    for rule in GEOMETRIC_RULES:
        rid = rule["id"]
        fails = rule_fail_counts[rid]
        status = "✅ PASS" if fails == 0 else f"❌ {fails}"
        print(f"#{rid:<5} | {rule['type']:<11} | {rule['desc'][:46]:<46} | {fails:<8} | {status:<6}")

    # 3. Rule 13 Lordosis Analysis
    print("\n" + "-" * 80)
    print("🔍 RULE 13 ANATOMICAL LORDOSIS & EXTENSION ANALYSIS:")
    print("   Rule 13 Strict: 'C4_PS => must be always at the bottom of C3_PI' (y_C4_PS > y_C3_PI)")
    print("   Rule 13 Safer:  'C4_PS => must be always at the bottom of C3_PS' (y_C4_PS > y_C3_PS)")
    print(f"   Strict Rule 13 Failures:                            {rule_fail_counts[13]}")
    print(f"   Lordosis Cases (Strict Fail, but Safer Superior-to-Superior Passes): {len(lordosis_cases)}")
    if lordosis_cases:
        print(f"   Samples exhibiting physiological lordosis/tilt overstep:")
        for lc in lordosis_cases[:10]:
            r13 = lc["rule13_info"]
            print(f"     • {lc['filename']}: C4_PS_y - C3_PI_y = {r13['strict_delta']}px (strict FAIL), C4_PS_y - C3_PS_y = +{r13['safer_delta']}px (safer PASS)")
        if len(lordosis_cases) > 10:
            print(f"     ... and {len(lordosis_cases) - 10} more.")

    # 4. Detailed Violating Samples List
    violating_samples = [r for r in results if not r["is_valid"]]
    if violating_samples:
        print("\n" + "-" * 80)
        print(f"🚨 SAMPLES VIOLATING CONSTRAINTS (Total {len(violating_samples)}):")
        for res in violating_samples[:25]:
            print(f"\n  📁 {res['filename']}:")
            if res["has_label_error"]:
                for err in res["label_errors"]:
                    print(f"     [LABEL ERROR] {err}")
            for v in res["rule_violations"]:
                is_r13_lordosis = (v["rule_id"] == 13 and res.get("rule13_info", {}).get("is_lordosis_tilt_case", False))
                tag = "[RULE 13 LORDOSIS]" if is_r13_lordosis else f"[RULE #{v['rule_id']} VIOLATION]"
                print(f"     {tag} {v['message']}")
        if len(violating_samples) > 25:
            print(f"\n  ... ({len(violating_samples) - 25} more violating samples omitted from console; see index.html for complete inspection)")
    else:
        print("\n🎉 All samples strictly adhere to all constraints!")

    print("\n" + "=" * 80 + "\n")


def generate_web_ui(
    violating_records: List[dict],
    all_records: List[dict],
    images_source_dir: str,
    output_dir: str,
) -> None:
    """
    Copies violating images into output_dir and writes an interactive, zero-dependency index.html.
    Features:
    - Canvas with zoom/pan and high resolution rendering.
    - Responsive, collision-free badge placement near each landmark with leader lines.
    - Distinct vibrant colors per landmark.
    - Violation highlights and interactive rule inspector table.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dest_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dest_dir, exist_ok=True)

    copied_images = []
    for rec in violating_records:
        fname = rec["filename"]
        src_path = os.path.join(images_source_dir, fname)
        dst_path = os.path.join(images_dest_dir, fname)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_images.append(fname)
        else:
            print(f"⚠️ Warning: Violating image '{fname}' not found in {images_source_dir}")

    # Prepare payload for JavaScript
    ui_dataset = []
    for rec in violating_records:
        ui_dataset.append({
            "filename": rec["filename"],
            "image_url": f"images/{rec['filename']}",
            "is_valid": rec["is_valid"],
            "has_label_error": rec["has_label_error"],
            "label_errors": rec["label_errors"],
            "rule_violations": rec["rule_violations"],
            "rule13_info": rec.get("rule13_info", {}),
            "landmarks": rec["landmarks"],
        })

    # Write data.js for clean separation
    data_js_path = os.path.join(output_dir, "data.js")
    with open(data_js_path, "w", encoding="utf-8") as f:
        f.write("window.VALIDATION_DATA = " + json.dumps(ui_dataset, indent=2) + ";\n")
        f.write("window.LANDMARK_COLORS = " + json.dumps(LANDMARK_COLORS, indent=2) + ";\n")
        f.write("window.GEOMETRIC_RULES = " + json.dumps(GEOMETRIC_RULES, indent=2) + ";\n")

    # Generate index.html
    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(INDEX_HTML_CONTENT)

    print(f"✅ Web UI successfully generated at: {os.path.abspath(html_path)}")
    print(f"   Copied {len(copied_images)} violating images into: {os.path.abspath(images_dest_dir)}")


INDEX_HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CVM Cervical Spine - Landmark Validation Inspector</title>
  <style>
    :root {
      --bg-dark: #0f172a;
      --panel-bg: #1e293b;
      --panel-border: #334155;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --danger: #ef4444;
      --warning: #f59e0b;
      --success: #10b981;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background: var(--bg-dark); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }

    /* Left Sidebar: Image List */
    #sidebar {
      width: 320px;
      min-width: 280px;
      background: var(--panel-bg);
      border-right: 1px solid var(--panel-border);
      display: flex;
      flex-direction: column;
      height: 100%;
    }
    .sidebar-header {
      padding: 16px;
      border-bottom: 1px solid var(--panel-border);
    }
    .sidebar-header h2 { font-size: 1.1rem; font-weight: 700; color: var(--accent); margin-bottom: 6px; }
    .sidebar-header p { font-size: 0.8rem; color: var(--text-muted); }
    .search-box {
      width: 100%;
      padding: 8px 12px;
      margin-top: 10px;
      border-radius: 6px;
      border: 1px solid var(--panel-border);
      background: #0b1120;
      color: #fff;
      font-size: 0.85rem;
    }
    .filter-select {
      width: 100%;
      padding: 6px 10px;
      margin-top: 8px;
      border-radius: 6px;
      border: 1px solid var(--panel-border);
      background: #0b1120;
      color: #e2e8f0;
      font-size: 0.8rem;
    }
    #image-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
    }
    .image-item {
      padding: 10px 12px;
      margin-bottom: 6px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid transparent;
      transition: all 0.15s ease;
    }
    .image-item:hover { background: rgba(56, 189, 248, 0.1); border-color: rgba(56, 189, 248, 0.3); }
    .image-item.active { background: rgba(56, 189, 248, 0.2); border-color: var(--accent); font-weight: 600; }
    .item-title { font-size: 0.85rem; }
    .badge-count {
      background: var(--danger);
      color: #fff;
      font-size: 0.75rem;
      padding: 2px 7px;
      border-radius: 12px;
      font-weight: bold;
    }
    .badge-count.lordosis {
      background: var(--warning);
      color: #000;
    }

    /* Central Workspace: Canvas */
    #viewport-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      position: relative;
      background: #020617;
      overflow: hidden;
    }
    #toolbar {
      height: 48px;
      background: rgba(30, 41, 59, 0.85);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      z-index: 10;
    }
    .toolbar-left, .toolbar-right { display: flex; align-items: center; gap: 8px; }
    .btn {
      background: #334155;
      color: var(--text-main);
      border: none;
      padding: 6px 12px;
      border-radius: 5px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s;
    }
    .btn:hover { background: #475569; }
    .btn-primary { background: var(--accent); color: #000; }
    .btn-primary:hover { background: var(--accent-hover); color: #fff; }
    .zoom-text { font-size: 0.85rem; color: var(--text-muted); min-width: 50px; text-align: center; }

    #canvas-wrapper {
      flex: 1;
      position: relative;
      cursor: grab;
    }
    #canvas-wrapper:active { cursor: grabbing; }
    canvas { display: block; width: 100%; height: 100%; }

    /* Right Inspection Panel */
    #inspector {
      width: 360px;
      min-width: 300px;
      background: var(--panel-bg);
      border-left: 1px solid var(--panel-border);
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow-y: auto;
    }
    .panel-section {
      padding: 16px;
      border-bottom: 1px solid var(--panel-border);
    }
    .panel-section h3 { font-size: 0.95rem; font-weight: 600; margin-bottom: 10px; color: var(--accent); }
    .violation-card {
      background: rgba(239, 68, 68, 0.12);
      border-left: 4px solid var(--danger);
      padding: 10px 12px;
      border-radius: 4px;
      margin-bottom: 8px;
      font-size: 0.8rem;
    }
    .violation-card.lordosis {
      background: rgba(245, 158, 11, 0.15);
      border-left-color: var(--warning);
    }
    .violation-title { font-weight: 700; margin-bottom: 4px; display: flex; justify-content: space-between; }
    .violation-detail { color: #cbd5e1; font-family: monospace; font-size: 0.75rem; margin-top: 4px; }

    /* Landmark Legend & Table */
    .legend-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 8px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      background: rgba(0, 0, 0, 0.2);
      padding: 4px 8px;
      border-radius: 4px;
    }
    .legend-color {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .legend-name { font-weight: 600; }
    .legend-coords { margin-left: auto; color: var(--text-muted); font-size: 0.7rem; font-family: monospace; }

    /* Rule Checklist */
    .rule-checklist {
      list-style: none;
      font-size: 0.78rem;
    }
    .rule-checklist li {
      padding: 6px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .rule-pass { color: var(--success); font-weight: 600; }
    .rule-fail { color: var(--danger); font-weight: 600; }
    .rule-warning { color: var(--warning); font-weight: 600; }
  </style>
</head>
<body>

  <!-- Left Sidebar: Violating Images -->
  <aside id="sidebar">
    <div class="sidebar-header">
      <h2>Validation Violations</h2>
      <p id="summary-count">Loading violations...</p>
      <input type="text" id="search-input" class="search-box" placeholder="Search filename (e.g. 0023)...">
      <select id="rule-filter" class="filter-select">
        <option value="ALL">All Violations</option>
        <option value="LABEL_ERROR">Label Count/Uniqueness Errors</option>
        <option value="R13_STRICT">Rule #13 Strict Failures</option>
        <option value="R13_LORDOSIS">Rule #13 Lordosis Only</option>
      </select>
    </div>
    <div id="image-list"></div>
  </aside>

  <!-- Central Canvas -->
  <main id="viewport-container">
    <div id="toolbar">
      <div class="toolbar-left">
        <button class="btn" id="btn-zoom-in">➕ Zoom In</button>
        <button class="btn" id="btn-zoom-out">➖ Zoom Out</button>
        <button class="btn" id="btn-reset">⟲ Fit to Screen</button>
        <span class="zoom-text" id="zoom-level">100%</span>
      </div>
      <div class="toolbar-right">
        <label style="font-size:0.8rem; display:flex; align-items:center; gap:6px; cursor:pointer;">
          <input type="checkbox" id="toggle-badges" checked> Always Show Labels
        </label>
        <label style="font-size:0.8rem; display:flex; align-items:center; gap:6px; cursor:pointer;">
          <input type="checkbox" id="toggle-connections" checked> Vertebrae Polygons
        </label>
      </div>
    </div>
    <div id="canvas-wrapper">
      <canvas id="main-canvas"></canvas>
    </div>
  </main>

  <!-- Right Inspection Panel -->
  <aside id="inspector">
    <div class="panel-section">
      <h3 id="current-filename">Select an image</h3>
      <div id="violation-cards-container"></div>
    </div>

    <div class="panel-section">
      <h3>13 Cervical Landmarks</h3>
      <div class="legend-grid" id="landmark-legend"></div>
    </div>

    <div class="panel-section">
      <h3>18 Geometric Rules Checklist</h3>
      <ul class="rule-checklist" id="rule-checklist"></ul>
    </div>
  </aside>

  <!-- Load injected JSON dataset -->
  <script src="data.js"></script>
  <script>
    // State management
    let currentIndex = 0;
    let filteredData = [];
    let currentImage = new Image();
    let imageLoaded = false;

    // Canvas transformations
    let scale = 1.0;
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let startDragX = 0;
    let startDragY = 0;

    const canvas = document.getElementById("main-canvas");
    const ctx = canvas.getContext("2d");
    const wrapper = document.getElementById("canvas-wrapper");

    // Vertebrae groupings for optional polygon visualization
    const VERTEBRAE_GROUPS = [
      { name: "C2", keys: ["C2_PI", "C2_IC", "C2_AI"], color: "rgba(255, 152, 0, 0.25)" },
      { name: "C3", keys: ["C3_PS", "C3_AS", "C3_AI", "C3_IC", "C3_PI"], color: "rgba(76, 175, 80, 0.25)" },
      { name: "C4", keys: ["C4_PS", "C4_AS", "C4_AI", "C4_IC", "C4_PI"], color: "rgba(156, 39, 176, 0.25)" }
    ];

    function init() {
      if (!window.VALIDATION_DATA || window.VALIDATION_DATA.length === 0) {
        document.getElementById("summary-count").textContent = "0 violating images found!";
        document.getElementById("image-list").innerHTML = "<div style='padding:20px; color:#94a3b8;'>No violations found! All images passed all validation rules.</div>";
        return;
      }

      // Populate rule filter options
      const filterSelect = document.getElementById("rule-filter");
      window.GEOMETRIC_RULES.forEach(r => {
        const opt = document.createElement("option");
        opt.value = `RULE_${r.id}`;
        opt.textContent = `Rule #${r.id}: ${r.desc.substring(0, 32)}...`;
        filterSelect.appendChild(opt);
      });

      filteredData = [...window.VALIDATION_DATA];
      document.getElementById("summary-count").textContent = `${filteredData.length} violating images`;

      renderSidebarList();
      setupEventListeners();
      resizeCanvas();
      selectImage(0);
    }

    function renderSidebarList() {
      const container = document.getElementById("image-list");
      container.innerHTML = "";

      filteredData.forEach((item, idx) => {
        const div = document.createElement("div");
        div.className = `image-item ${idx === currentIndex ? "active" : ""}`;
        
        const isLordosis = item.rule13_info && item.rule13_info.is_lordosis_tilt_case;
        const count = item.rule_violations.length + (item.has_label_error ? 1 : 0);
        const badgeClass = isLordosis ? "badge-count lordosis" : "badge-count";

        div.innerHTML = `
          <span class="item-title">${item.filename}</span>
          <span class="${badgeClass}">${count} fail</span>
        `;
        div.onclick = () => selectImage(idx);
        container.appendChild(div);
      });
    }

    function selectImage(index) {
      if (index < 0 || index >= filteredData.length) return;
      currentIndex = index;

      document.querySelectorAll(".image-item").forEach((el, idx) => {
        el.classList.toggle("active", idx === currentIndex);
      });

      const record = filteredData[currentIndex];
      document.getElementById("current-filename").textContent = record.filename;

      imageLoaded = false;
      currentImage = new Image();
      currentImage.onload = () => {
        imageLoaded = true;
        fitImageToScreen();
        render();
      };
      currentImage.src = record.image_url;

      renderInspector(record);
    }

    function renderInspector(record) {
      // 1. Violations Cards
      const container = document.getElementById("violation-cards-container");
      container.innerHTML = "";

      if (record.has_label_error) {
        const card = document.createElement("div");
        card.className = "violation-card";
        card.innerHTML = `
          <div class="violation-title"><span>⚠️ Label Error</span></div>
          <div class="violation-detail">${record.label_errors.join("<br>")}</div>
        `;
        container.appendChild(card);
      }

      record.rule_violations.forEach(v => {
        const isR13Lordosis = (v.rule_id === 13 && record.rule13_info && record.rule13_info.is_lordosis_tilt_case);
        const card = document.createElement("div");
        card.className = isR13Lordosis ? "violation-card lordosis" : "violation-card";
        card.innerHTML = `
          <div class="violation-title">
            <span>${isR13Lordosis ? "Tilt / Lordosis" : `Rule #${v.rule_id}`}</span>
            <span>Δ ${v.delta !== null ? v.delta + "px" : "N/A"}</span>
          </div>
          <div>${v.desc}</div>
          <div class="violation-detail">${v.message}</div>
          ${isR13Lordosis ? `<div class="violation-detail" style="color:#fbbf24; margin-top:3px;">Passed safer check (C4_PS below C3_PS by +${record.rule13_info.safer_delta}px)</div>` : ""}
        `;
        container.appendChild(card);
      });

      // 2. Landmark Legend Grid
      const legend = document.getElementById("landmark-legend");
      legend.innerHTML = "";
      Object.keys(window.LANDMARK_COLORS).forEach(name => {
        const rawPt = record.landmarks[name];
        const pts = Array.isArray(rawPt) ? rawPt : (rawPt ? [rawPt] : []);
        const color = window.LANDMARK_COLORS[name] || "#ffffff";
        const div = document.createElement("div");
        div.className = "legend-item";
        let coordText = "None";
        if (pts.length === 1) {
          coordText = `${Math.round(pts[0].x)},${Math.round(pts[0].y)}`;
        } else if (pts.length > 1) {
          coordText = pts.map(p => `(${Math.round(p.x)},${Math.round(p.y)})`).join(" ");
        }
        div.innerHTML = `
          <div class="legend-color" style="background:${color};"></div>
          <span class="legend-name">${name} ${pts.length > 1 ? `<span style="color:#ef4444; font-size:0.7rem;">(${pts.length}x DUP)</span>` : ""}</span>
          <span class="legend-coords">${coordText}</span>
        `;
        legend.appendChild(div);
      });

      // 3. Rule Checklist
      const checklist = document.getElementById("rule-checklist");
      checklist.innerHTML = "";
      window.GEOMETRIC_RULES.forEach(r => {
        const v = record.rule_violations.find(viol => viol.rule_id === r.id);
        const li = document.createElement("li");
        
        let statusBadge = `<span class="rule-pass">✓ PASS</span>`;
        if (v) {
          if (r.id === 13 && record.rule13_info && record.rule13_info.is_lordosis_tilt_case) {
            statusBadge = `<span class="rule-warning">⚠️ LORDOSIS</span>`;
          } else {
            statusBadge = `<span class="rule-fail">✗ FAIL (${v.delta}px)</span>`;
          }
        }

        li.innerHTML = `
          <span>#${r.id} ${r.a} → ${r.b}</span>
          ${statusBadge}
        `;
        checklist.appendChild(li);
      });
    }

    function fitImageToScreen() {
      if (!imageLoaded || !currentImage.width) return;
      const w = wrapper.clientWidth;
      const h = wrapper.clientHeight;

      const scaleX = (w * 0.85) / currentImage.width;
      const scaleY = (h * 0.85) / currentImage.height;
      scale = Math.min(scaleX, scaleY, 1.5);

      panX = (w - currentImage.width * scale) / 2;
      panY = (h - currentImage.height * scale) / 2;
      updateZoomLabel();
    }

    function updateZoomLabel() {
      document.getElementById("zoom-level").textContent = `${Math.round(scale * 100)}%`;
    }

    function resizeCanvas() {
      canvas.width = wrapper.clientWidth * window.devicePixelRatio;
      canvas.height = wrapper.clientHeight * window.devicePixelRatio;
      canvas.style.width = `${wrapper.clientWidth}px`;
      canvas.style.height = `${wrapper.clientHeight}px`;
      render();
    }

    /* -------------------------------------------------------------
       Collision-Free Non-Overlapping Label Placement Algorithm
       ------------------------------------------------------------- */
    function computeNonOverlappingBadges(record, landmarks, imgScale, imgPanX, imgPanY) {
      const badges = [];
      const badgeH = 18;

      // Initial preferred placement offset based on anatomical landmark position
      Object.keys(landmarks).forEach(name => {
        const rawPt = landmarks[name];
        if (!rawPt) return;
        const pts = Array.isArray(rawPt) ? rawPt : [rawPt];

        pts.forEach((pt, pIdx) => {
          if (!pt || typeof pt.x !== "number") return;

          const screenX = pt.x * imgScale + imgPanX;
          const screenY = pt.y * imgScale + imgPanY;

          const isDuplicate = pts.length > 1;
          const displayName = isDuplicate ? `${name} [${pIdx + 1}]` : name;
          const badgeW = isDuplicate ? 68 : 56;

          let offsetX = 25;
          let offsetY = 0;

          if (name.includes("_PI") || name.includes("_PS")) {
            offsetX = -badgeW - 20; // Posterior corners placed towards the left
            offsetY = name.includes("_PS") ? -16 : 14;
          } else if (name.includes("_AI") || name.includes("_AS")) {
            offsetX = 22;           // Anterior corners placed towards the right
            offsetY = name.includes("_AS") ? -16 : 14;
          } else if (name.includes("_IC")) {
            offsetX = -badgeW / 2;  // Inferior-center placed directly below
            offsetY = 24;
          }

          if (isDuplicate && pIdx > 0) {
            offsetY += 22 * pIdx;
          }

          const hasRuleViolation = record.rule_violations.some(v => v.landmark_a === name || v.landmark_b === name);
          const isViolating = isDuplicate || hasRuleViolation;

          badges.push({
            name: displayName,
            pointX: screenX,
            pointY: screenY,
            x: screenX + offsetX,
            y: screenY + offsetY,
            w: badgeW,
            h: badgeH,
            color: window.LANDMARK_COLORS[name] || "#38bdf8",
            isViolating: isViolating,
            isDuplicate: isDuplicate
          });
        });
      });

      // Relaxation iterations to guarantee zero overlap between badges and landmarks
      const iterations = 45;
      for (let iter = 0; iter < iterations; iter++) {
        // 1. Repel badges from other badges
        for (let i = 0; i < badges.length; i++) {
          for (let j = i + 1; j < badges.length; j++) {
            const b1 = badges[i];
            const b2 = badges[j];

            const dx = (b1.x + b1.w / 2) - (b2.x + b2.w / 2);
            const dy = (b1.y + b1.h / 2) - (b2.y + b2.h / 2);
            const minDistX = (b1.w + b2.w) / 2 + 6;
            const minDistY = (b1.h + b2.h) / 2 + 4;

            if (Math.abs(dx) < minDistX && Math.abs(dy) < minDistY) {
              const overlapX = minDistX - Math.abs(dx);
              const overlapY = minDistY - Math.abs(dy);

              if (overlapX < overlapY) {
                const shiftX = (overlapX / 2) * (dx > 0 ? 1 : -1);
                b1.x += shiftX;
                b2.x -= shiftX;
              } else {
                const shiftY = (overlapY / 2) * (dy > 0 ? 1 : -1);
                b1.y += shiftY;
                b2.y -= shiftY;
              }
            }
          }
        }

        // 2. Repel badges from landmark keypoints so they never obscure any dot
        for (let i = 0; i < badges.length; i++) {
          const b = badges[i];
          for (let j = 0; j < badges.length; j++) {
            const pt = badges[j];
            const nearX = Math.max(b.x, Math.min(pt.pointX, b.x + b.w));
            const nearY = Math.max(b.y, Math.min(pt.pointY, b.y + b.h));
            const dist = Math.hypot(pt.pointX - nearX, pt.pointY - nearY);
            const avoidRadius = 16;

            if (dist < avoidRadius) {
              const force = (avoidRadius - dist);
              const angle = Math.atan2((b.y + b.h / 2) - pt.pointY, (b.x + b.w / 2) - pt.pointX) || (Math.PI / 4);
              b.x += Math.cos(angle) * force * 0.7;
              b.y += Math.sin(angle) * force * 0.7;
            }
          }
        }
      }

      return badges;
    }

    function render() {
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

      if (imageLoaded && currentImage.complete) {
        // Draw the base X-Ray image
        ctx.drawImage(currentImage, panX, panY, currentImage.width * scale, currentImage.height * scale);

        const record = filteredData[currentIndex];
        if (record && record.landmarks) {
          const showBadges = document.getElementById("toggle-badges").checked;
          const showVertebrae = document.getElementById("toggle-connections").checked;

          // 1. Draw Vertebrae Contours if enabled
          if (showVertebrae) {
            VERTEBRAE_GROUPS.forEach(grp => {
              const pts = grp.keys.map(k => {
                const raw = record.landmarks[k];
                return Array.isArray(raw) ? raw[0] : raw;
              }).filter(p => p && typeof p.x === "number");
              if (pts.length >= 3) {
                ctx.beginPath();
                ctx.moveTo(pts[0].x * scale + panX, pts[0].y * scale + panY);
                for (let i = 1; i < pts.length; i++) {
                  ctx.lineTo(pts[i].x * scale + panX, pts[i].y * scale + panY);
                }
                ctx.closePath();
                ctx.fillStyle = grp.color;
                ctx.fill();
                ctx.strokeStyle = grp.color.replace("0.25", "0.6");
                ctx.lineWidth = 1.5;
                ctx.stroke();
              }
            });
          }

          // 2. Draw Violation Arrow Connectors
          record.rule_violations.forEach(v => {
            const rawA = record.landmarks[v.landmark_a];
            const rawB = record.landmarks[v.landmark_b];
            const ptA = Array.isArray(rawA) ? rawA[0] : rawA;
            const ptB = Array.isArray(rawB) ? rawB[0] : rawB;
            if (ptA && ptB && typeof ptA.x === "number" && typeof ptB.x === "number") {
              const ax = ptA.x * scale + panX;
              const ay = ptA.y * scale + panY;
              const bx = ptB.x * scale + panX;
              const by = ptB.y * scale + panY;

              ctx.beginPath();
              ctx.setLineDash([5, 4]);
              ctx.moveTo(ax, ay);
              ctx.lineTo(bx, by);
              ctx.strokeStyle = "#ef4444";
              ctx.lineWidth = 2.5;
              ctx.stroke();
              ctx.setLineDash([]);
            }
          });

          // 3. Compute Collision-Free Badges
          const badges = computeNonOverlappingBadges(record, record.landmarks, scale, panX, panY);

          // 4. Draw Leader Lines and Badges
          badges.forEach(b => {
            if (showBadges) {
              // Leader line from landmark to badge
              ctx.beginPath();
              ctx.moveTo(b.pointX, b.pointY);
              const targetX = b.x > b.pointX ? b.x : b.x + b.w;
              const targetY = b.y + b.h / 2;
              ctx.lineTo(targetX, targetY);
              ctx.strokeStyle = b.isViolating ? "#ef4444" : "rgba(255, 255, 255, 0.4)";
              ctx.lineWidth = b.isViolating ? 1.8 : 1.0;
              ctx.stroke();

              // Badge Card / Pill
              ctx.fillStyle = b.isViolating ? "#7f1d1d" : "#0f172a";
              ctx.strokeStyle = b.isViolating ? "#ef4444" : b.color;
              ctx.lineWidth = b.isViolating ? 2 : 1.5;

              // Rounded rectangle badge
              drawRoundedRect(ctx, b.x, b.y, b.w, b.h, 4);
              ctx.fill();
              ctx.stroke();

              // Badge Text
              ctx.fillStyle = "#ffffff";
              ctx.font = "bold 9.5px monospace";
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillText(b.name, b.x + b.w / 2, b.y + b.h / 2);
            }

            // Landmark Dot
            ctx.beginPath();
            ctx.arc(b.pointX, b.pointY, b.isViolating ? 5.5 : 4, 0, Math.PI * 2);
            ctx.fillStyle = b.color;
            ctx.fill();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = "#ffffff";
            ctx.stroke();

            // Violating ring
            if (b.isViolating) {
              ctx.beginPath();
              ctx.arc(b.pointX, b.pointY, 8.5, 0, Math.PI * 2);
              ctx.strokeStyle = "#ef4444";
              ctx.lineWidth = 2;
              ctx.stroke();
            }
          });
        }
      }

      ctx.restore();
    }

    function drawRoundedRect(ctx, x, y, width, height, radius) {
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + width - radius, y);
      ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
      ctx.lineTo(x + width, y + height - radius);
      ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
      ctx.lineTo(x + radius, y + height);
      ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
    }

    function setupEventListeners() {
      window.addEventListener("resize", resizeCanvas);

      // Pan interactions
      wrapper.addEventListener("mousedown", (e) => {
        isDragging = true;
        startDragX = e.clientX - panX;
        startDragY = e.clientY - panY;
      });

      window.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        panX = e.clientX - startDragX;
        panY = e.clientY - startDragY;
        render();
      });

      window.addEventListener("mouseup", () => { isDragging = false; });

      // Smooth Zoom on Mouse Wheel
      wrapper.addEventListener("wheel", (e) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
        const newScale = Math.max(0.2, Math.min(8.0, scale * zoomFactor));

        panX = mouseX - (mouseX - panX) * (newScale / scale);
        panY = mouseY - (mouseY - panY) * (newScale / scale);
        scale = newScale;

        updateZoomLabel();
        render();
      }, { passive: false });

      // Buttons
      document.getElementById("btn-zoom-in").onclick = () => {
        scale = Math.min(8.0, scale * 1.25);
        updateZoomLabel();
        render();
      };
      document.getElementById("btn-zoom-out").onclick = () => {
        scale = Math.max(0.2, scale * 0.8);
        updateZoomLabel();
        render();
      };
      document.getElementById("btn-reset").onclick = () => {
        fitImageToScreen();
        render();
      };

      document.getElementById("toggle-badges").onchange = render;
      document.getElementById("toggle-connections").onchange = render;

      // Filtering and Search
      document.getElementById("search-input").addEventListener("input", applyFilters);
      document.getElementById("rule-filter").addEventListener("change", applyFilters);
    }

    function applyFilters() {
      const q = document.getElementById("search-input").value.trim().toLowerCase();
      const ruleChoice = document.getElementById("rule-filter").value;

      filteredData = window.VALIDATION_DATA.filter(item => {
        const matchesQuery = !q || item.filename.toLowerCase().includes(q);
        if (!matchesQuery) return false;

        if (ruleChoice === "ALL") return true;
        if (ruleChoice === "LABEL_ERROR") return item.has_label_error;
        if (ruleChoice === "R13_STRICT") return item.rule_violations.some(v => v.rule_id === 13);
        if (ruleChoice === "R13_LORDOSIS") return item.rule13_info && item.rule13_info.is_lordosis_tilt_case;
        if (ruleChoice.startsWith("RULE_")) {
          const rid = parseInt(ruleChoice.replace("RULE_", ""), 10);
          return item.rule_violations.some(v => v.rule_id === rid);
        }
        return true;
      });

      document.getElementById("summary-count").textContent = `${filteredData.length} matching violations`;
      renderSidebarList();
      if (filteredData.length > 0) {
        selectImage(0);
      } else {
        document.getElementById("current-filename").textContent = "No matching images";
        document.getElementById("violation-cards-container").innerHTML = "<div style='color:#94a3b8;'>Try adjusting your search or filter.</div>";
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    window.onload = init;
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Validation & Geometric Consistency Verification Script."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        default=True,
        help="Skip downloading export and images from Label Studio (default: True if raw export exists).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download from Label Studio even if raw files exist.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip image resizing and label generation, validating existing export and images directly.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=640,
        help="Target square image dimension for preprocessing (default: 640).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian heatmap sigma parameter (default: 3.0).",
    )
    parser.add_argument(
        "--raw-exports-dir",
        type=str,
        default="data/raw/exports",
        help="Directory containing raw Label Studio exports.",
    )
    parser.add_argument(
        "--raw-images-dir",
        type=str,
        default="data/raw/images",
        help="Directory containing raw images.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/images",
        help="Directory of resized images.",
    )
    parser.add_argument(
        "--exports-json",
        type=str,
        default="data/exports/export.json",
        help="Path to preprocessed export JSON.",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/labels",
        help="Path to output NPZ labels directory.",
    )
    parser.add_argument(
        "--output-violations-dir",
        type=str,
        default="data/validation_violations",
        help="Output directory to copy violating images and store index.html visualizer.",
    )
    parser.add_argument(
        "--rule13-mode",
        type=str,
        choices=["strict", "safer", "both"],
        default="both",
        help="Rule 13 handling mode: 'strict' (y_C4_PS > y_C3_PI), 'safer' (y_C4_PS > y_C3_PS), or 'both' (default: both).",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("      CERVICAL VERTEBRAE DATASET VALIDATION & PRE-PROCESSING PIPELINE     ")
    print("=" * 80)

    # 1. Preprocessing Steps (up to generate_labels)
    if not args.skip_preprocessing:
        if args.force_download or (not args.skip_download):
            print("\n[Step 1/3] Downloading raw export and images from Label Studio...")
            download_export_and_images(
                export_dir=args.raw_exports_dir,
                img_dir=args.raw_images_dir,
            )
        else:
            print("\n[Step 1/3] Skipping download (using existing data).")

        print("\n[Step 2/3] Resizing images to target square dimension & updating coordinates...")
        raw_exports_dir = Path(args.raw_exports_dir)
        raw_json_files = sorted(
            raw_exports_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        raw_json_path = str(raw_json_files[0]) if raw_json_files else args.exports_json

        resize_images(
            img_dir=args.raw_images_dir,
            json_path=raw_json_path,
            out_img_dir=args.images_dir,
            out_json_path=args.exports_json,
            target_size=args.target_size,
        )

        print("\n[Step 3/3] Generating Heatmaps & GCN Landmark Labels...")
        generate_labels(
            json_path=args.exports_json,
            images_dir=args.images_dir,
            output_dir=args.labels_dir,
            sigma=args.sigma,
        )
    else:
        print("\n[Preprocessing] Skipping steps 1-3 (--skip-preprocessing flag set).")

    # 2. Validation Execution
    if not os.path.exists(args.exports_json):
        print(f"❌ Error: Export file not found at: {args.exports_json}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[Validation] Loading export records from: {args.exports_json}")
    with open(args.exports_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"[Validation] Evaluating {len(records)} records for label integrity & 18 geometric rules...")
    validation_results = []
    for rec in records:
        res = validate_single_record(rec, rule13_mode=args.rule13_mode)
        validation_results.append(res)

    # 3. Terminal Report
    print_terminal_report(validation_results, rule13_mode=args.rule13_mode)

    # 4. Copy Violating Images and Generate Interactive Web UI
    violating_records = [r for r in validation_results if not r["is_valid"]]
    if violating_records:
        print(f"[Web UI] Generating interactive visualizer for {len(violating_records)} violating images...")
        generate_web_ui(
            violating_records=violating_records,
            all_records=validation_results,
            images_source_dir=args.images_dir,
            output_dir=args.output_violations_dir,
        )
    else:
        print("✅ No violations found. Skipping Web UI generation.")


if __name__ == "__main__":
    main()
