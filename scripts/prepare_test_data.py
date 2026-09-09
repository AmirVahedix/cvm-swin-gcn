#!/usr/bin/env python3
"""
Prepares the exact test dataset split for local evaluation on Mac.
Isolates the exact 147 test IDs from the specified training run,
downloads/processes only those test images and generates ground-truth .npz labels.
Optionally fetches artifacts/best.pth directly from MLflow.

Usage:
    python scripts/prepare_test_data.py
    python scripts/prepare_test_data.py --download-weights --run-id 273e011b9d3642e4b8921895379dc2d6
"""

import os
import sys
import json
import argparse
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS
from src.data.loaders.dataset import generate_gaussian_heatmaps


def get_task_filename(task: dict) -> str:
    raw_path = task.get("file_upload") or task.get("data", {}).get("img") or ""
    parsed_url = urlparse(raw_path)
    query_params = parse_qs(parsed_url.query)
    if "d" in query_params:
        filename = os.path.basename(query_params["d"][0])
    else:
        filename = os.path.basename(parsed_url.path)

    if "-" in filename and len(filename.split("-")[0]) == 8:
        return "-".join(filename.split("-")[1:])
    return filename


def extract_landmarks_from_task(task: dict) -> np.ndarray:
    """
    Extracts normalized [13, 2] coordinates from Label Studio task annotations.
    Missing landmarks are represented as [-1.0, -1.0].
    """
    coords = np.full((NUM_LANDMARKS, 2), -1.0, dtype=np.float32)
    annotations = task.get("annotations", [])
    if not annotations:
        return coords

    for ann in annotations:
        for res in ann.get("result", []):
            if res.get("type") != "keypointlabels":
                continue
            labels = res.get("value", {}).get("keypointlabels", [])
            if not labels:
                continue
            lbl_name = labels[0]
            if lbl_name in LANDMARK_CLASSES:
                idx = LANDMARK_CLASSES.index(lbl_name)
                val = res["value"]
                # Label Studio stores coords as percentages [0, 100]
                norm_x = float(val["x"]) / 100.0
                norm_y = float(val["y"]) / 100.0
                coords[idx] = [norm_x, norm_y]

    return coords


def resize_letterbox(img: np.ndarray, target_size: int = 640):
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    scale_x = float(new_w) / float(w)
    scale_y = float(new_h) / float(h)
    return canvas, scale_x, scale_y, pad_x, pad_y, w, h


def adjust_coords_for_letterbox(
    raw_norm_coords: np.ndarray,
    orig_w: int,
    orig_h: int,
    scale_x: float,
    scale_y: float,
    pad_x: int,
    pad_y: int,
    target_size: int = 640,
) -> np.ndarray:
    adjusted = np.full_like(raw_norm_coords, -1.0)
    for i in range(len(raw_norm_coords)):
        rx, ry = raw_norm_coords[i]
        if rx < 0 or ry < 0:
            continue
        orig_px_x = rx * orig_w
        orig_px_y = ry * orig_h
        canvas_px_x = orig_px_x * scale_x + pad_x
        canvas_px_y = orig_px_y * scale_y + pad_y
        adjusted[i] = [canvas_px_x / float(target_size), canvas_px_y / float(target_size)]
    return adjusted


def get_authenticated_session() -> requests.Session:
    load_dotenv()
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    api_token = os.getenv("LABEL_STUDIO_API_TOKEN")
    if api_token:
        session.headers.update({"Authorization": f"Token {api_token}"})

    ls_url = os.getenv("LABEL_STUDIO_URL")
    user = os.getenv("LABEL_STUDIO_USERNAME")
    pwd = os.getenv("LABEL_STUDIO_PASSWORD")
    if ls_url and user and pwd:
        try:
            login_url = f"{ls_url}/user/login/"
            resp = session.get(login_url, timeout=10)
            csrf_token = session.cookies.get("csrftoken", "")
            session.post(
                login_url,
                data={"email": user, "password": pwd, "csrfmiddlewaretoken": csrf_token},
                headers={"Referer": login_url},
                timeout=10,
            )
        except Exception as auth_err:
            print(f"ℹ️ Session login note: {auth_err}")

    return session


def fetch_latest_export(session: requests.Session = None, save_path: str = None) -> list:
    load_dotenv()
    ls_url = (os.getenv("LABEL_STUDIO_URL") or "").rstrip("/")
    project_id = os.getenv("LABEL_STUDIO_PROJECT_ID", "1")
    if not ls_url:
        raise ValueError("LABEL_STUDIO_URL is not set in environment or .env.")

    if session is None:
        session = get_authenticated_session()

    print(f"--> Fetching latest export directly from Label Studio ({ls_url}, project {project_id})...")
    export_url = f"{ls_url}/api/projects/{project_id}/export?exportType=JSON"
    export_resp = session.get(export_url, timeout=120)

    if export_resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch export from Label Studio. Status: {export_resp.status_code}, Response: {export_resp.text[:300]}"
        )

    tasks = export_resp.json()
    print(f"✅ Downloaded latest export from Label Studio ({len(tasks)} total tasks).")

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        print(f"💾 Updated local export file at: {save_path}")

    return tasks


def prepare_test_dataset(
    test_ids_file: str = "dataset/test_image_ids.json",
    export_json_path: str = "data/exports/export.json",
    output_dir: str = "./dataset/test",
    target_size: int = 640,
    sigma: float = 4.0,
    download_missing: bool = True,
    use_local_export: bool = False,
):
    load_dotenv()
    test_ids_path = Path(test_ids_file)
    if not test_ids_path.exists():
        raise FileNotFoundError(f"Test IDs file not found: {test_ids_file}")

    with open(test_ids_path, "r", encoding="utf-8") as f:
        test_ids_list = json.load(f)
    test_ids_set = {int(x) if str(x).isdigit() else str(x) for x in test_ids_list}
    print(f"--> Loaded {len(test_ids_set)} test task IDs from: {test_ids_file}")

    session = get_authenticated_session() if download_missing or not use_local_export else None

    # Load tasks
    tasks = []
    if not use_local_export:
        try:
            tasks = fetch_latest_export(session=session, save_path=export_json_path)
        except Exception as e:
            print(f"⚠️ Failed to download latest export from Label Studio: {e}")
            if os.path.exists(export_json_path):
                print(f"--> Falling back to local Label Studio export: {export_json_path}")
                with open(export_json_path, "r", encoding="utf-8") as f:
                    tasks = json.load(f)
            else:
                raise
    else:
        if os.path.exists(export_json_path):
            print(f"--> Reading local Label Studio export: {export_json_path}")
            with open(export_json_path, "r", encoding="utf-8") as f:
                tasks = json.load(f)
        else:
            raw_candidates = sorted(Path("data/raw/exports").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if raw_candidates:
                print(f"--> Reading fallback export: {raw_candidates[0]}")
                with open(raw_candidates[0], "r", encoding="utf-8") as f:
                    tasks = json.load(f)

    if not tasks:
        raise RuntimeError("No tasks found in Label Studio export.")

    test_tasks = [t for t in tasks if t.get("id") in test_ids_set]
    print(f"--> Matched {len(test_tasks)}/{len(test_ids_set)} tasks for test split.")

    out_path = Path(output_dir)
    img_out = out_path / "images"
    lbl_out = out_path / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    ls_url = os.getenv("LABEL_STUDIO_URL", "")

    saved_count = 0
    for task in tqdm(test_tasks, desc="Processing Test Samples", unit="img"):
        filename = get_task_filename(task)
        stem = Path(filename).stem

        # Try to find local raw image first
        candidate_paths = [
            Path("data/raw/images") / filename,
            Path("data/images") / filename,
            Path("dataset/images") / filename,
        ]
        raw_img = None
        for p in candidate_paths:
            if p.exists():
                raw_img = cv2.imread(str(p))
                if raw_img is not None:
                    break

        # Download if missing locally
        if raw_img is None and session and ls_url:
            img_url = task.get("data", {}).get("img") or task.get("file_upload")
            if img_url:
                full_url = img_url if img_url.startswith("http") else f"{ls_url}{img_url}"
                try:
                    resp = session.get(full_url, timeout=15)
                    if resp.status_code == 200:
                        img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                        raw_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                except Exception as dl_err:
                    print(f"⚠️ Failed to download {filename}: {dl_err}")

        if raw_img is None:
            print(f"⚠️ Skipping task {task['id']} ({filename}): Image not available.")
            continue

        # Resize & Letterbox
        canvas, sx, sy, px, py, orig_w, orig_h = resize_letterbox(raw_img, target_size=target_size)
        cv2.imwrite(str(img_out / f"{stem}.png"), canvas)

        # Annotations
        raw_norm_coords = extract_landmarks_from_task(task)
        adjusted_coords = adjust_coords_for_letterbox(
            raw_norm_coords, orig_w, orig_h, sx, sy, px, py, target_size=target_size
        )

        # Heatmaps
        adjusted_px = (adjusted_coords * target_size).astype(np.float32)
        heatmaps_tensor = generate_gaussian_heatmaps(adjusted_px, img_size=target_size, sigma=sigma)

        # Save target .npz
        np.savez_compressed(
            str(lbl_out / f"{stem}.npz"),
            coords=adjusted_coords.astype(np.float32),
            heatmaps=heatmaps_tensor.numpy().astype(np.float32),
        )
        saved_count += 1

    print(f"\n🎉 Done! Successfully generated {saved_count} test image & label pairs in '{output_dir}'.")


def download_mlflow_weights(run_id: str, dest_dir: str = "artifacts") -> str:
    import mlflow
    load_dotenv()

    # Pass MLflow tracking credentials
    user = os.getenv("MLFLOW_TRACKING_USERNAME")
    pwd = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if user:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
    if pwd:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd

    uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.amirlogedixyz.ir")
    mlflow.set_tracking_uri(uri)

    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    target_file = dest_path / "best.pth"

    print(f"--> Connecting to MLflow ({uri}) to fetch 'checkpoints/best.pth' from run '{run_id}'...")
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="checkpoints/best.pth",
        dst_path=str(dest_path),
    )
    if Path(local_path) != target_file and os.path.exists(local_path):
        import shutil
        shutil.copy2(local_path, target_file)
    print(f"✅ Successfully downloaded weights to: {target_file}")
    return str(target_file)


def main():
    parser = argparse.ArgumentParser(description="Prepare test dataset split for local Mac evaluation.")
    parser.add_argument("--test-ids-file", type=str, default="dataset/test_image_ids.json")
    parser.add_argument("--export-json", type=str, default="data/exports/export.json")
    parser.add_argument("--output-dir", type=str, default="./dataset/test")
    parser.add_argument("--target-size", type=int, default=640)
    parser.add_argument("--download-weights", action="store_true", help="Download best.pth checkpoint from MLflow.")
    parser.add_argument("--run-id", type=str, default="273e011b9d3642e4b8921895379dc2d6", help="MLflow Run ID.")
    parser.add_argument("--evaluate", action="store_true", help="Automatically run evaluation after preparing data.")
    parser.add_argument("--device", type=str, default="mps", help="Device to evaluate on ('mps', 'cpu', 'cuda').")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable logging local eval results to MLflow.")
    parser.add_argument(
        "--use-local-export",
        action="store_true",
        help="Use existing local export JSON instead of downloading the latest export from Label Studio.",
    )

    args = parser.parse_args()

    weights_path = Path("artifacts/best.pth")
    if args.download_weights or not weights_path.exists():
        download_mlflow_weights(run_id=args.run_id)

    prepare_test_dataset(
        test_ids_file=args.test_ids_file,
        export_json_path=args.export_json,
        output_dir=args.output_dir,
        target_size=args.target_size,
        use_local_export=args.use_local_export,
    )

    if args.evaluate:
        print("\n" + "=" * 80)
        print(f"STARTING CLINICAL EVALUATION ON DEVICE: {args.device.upper()}")
        print("=" * 80)
        from src.eval import run_evaluation
        run_evaluation(
            weights_path=str(weights_path),
            test_img_dir=str(Path(args.output_dir) / "images"),
            test_npz_dir=str(Path(args.output_dir) / "labels"),
            output_dir="evaluation",
            img_size=args.target_size,
            device_str=args.device,
            log_to_mlflow=not args.no_mlflow,
        )


if __name__ == "__main__":
    main()
