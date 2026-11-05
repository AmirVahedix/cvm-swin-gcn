import json
import os
import shutil
import requests
import argparse
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util import Retry
from tqdm import tqdm

load_dotenv()


def extract_filename_from_url(img_url: str) -> str:
    """
    Extracts a clean image filename from an image URL.
    """
    if "?d=" in img_url:
        parsed = urlparse(img_url)
        params = parse_qs(parsed.query)
        if "d" in params and params["d"]:
            return os.path.basename(params["d"][0])

    base = os.path.basename(img_url.split("?")[0])
    if "-" in base:
        parts = base.split("-", 1)
        if len(parts) > 1 and len(parts[0]) >= 8:
            return parts[1]
    return base


def clear_directory(dir_path):
    """
    Deletes all files and subdirectories inside the specified directory
    without deleting the root directory itself.
    """
    if os.path.exists(dir_path):
        print(f"🧹 Clearing existing data in: {dir_path}")
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"⚠️ Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(dir_path, exist_ok=True)


def download_from_cvm_api(
    export_dir="data/raw/exports",
    img_dir="data/raw/images",
    api_url=None,
    api_token=None,
    skip_review_needed=True,
    skip_excluded=True,
    max_workers=10,
):
    """
    Default data loader that fetches export and images from CVM Backend API (/api/training-data or /api/export).
    Automatically skips images flagged as review_needed or excluded to exclude them from the training process.
    All data is retrieved strictly via HTTP requests from the API.
    """
    api_url = (api_url or os.getenv("CVM_API_URL") or "http://localhost:8000").rstrip("/")
    api_token = api_token or os.getenv("CVM_API_TOKEN") or "O0s66Vi.r8e8WBGfY9I3x4neIm"

    clear_directory(img_dir)
    os.makedirs(export_dir, exist_ok=True)

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers,
        max_retries=retries,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    if api_token:
        session.headers.update({"Authorization": f"Bearer {api_token}"})

    tasks = []

    # 1. Fetch dataset strictly via HTTP from CVM Backend API
    print(f"📡 Connecting to CVM Backend API at {api_url}...")
    try:
        health_resp = session.get(f"{api_url}/api/health", timeout=10)
        if health_resp.status_code != 200:
            raise ConnectionError(
                f"CVM API health check failed at {api_url}/api/health (HTTP {health_resp.status_code})."
            )
        print("🔑 Connected to CVM Backend API successfully.")

        # Request training data directly (with exclude_review_needed=True and exclude_excluded=True)
        export_endpoint = f"{api_url}/api/training-data" if skip_review_needed else f"{api_url}/api/export"
        export_resp = session.get(export_endpoint, timeout=60)
        if export_resp.status_code == 200:
            raw_json = export_resp.json()
            if isinstance(raw_json, dict) and "data" in raw_json:
                tasks = raw_json["data"]
            elif isinstance(raw_json, list):
                tasks = raw_json
        else:
            # Fallback to /api/export
            export_resp2 = session.get(
                f"{api_url}/api/export?exclude_review_needed={'true' if skip_review_needed else 'false'}&exclude_excluded={'true' if skip_excluded else 'false'}",
                timeout=60,
            )
            if export_resp2.status_code == 200:
                raw_json = export_resp2.json()
                tasks = raw_json.get("data", raw_json) if isinstance(raw_json, dict) else raw_json
            else:
                raise RuntimeError(
                    f"Failed to fetch data from CVM API (HTTP {export_resp.status_code} on {export_endpoint}, HTTP {export_resp2.status_code} on /api/export)."
                )
    except Exception as conn_err:
        raise ConnectionError(
            f"Could not retrieve dataset from CVM Backend API at {api_url}: {conn_err}. "
            "Please make sure the CVM backend server is running and accessible."
        ) from conn_err

    if not tasks:
        raise ValueError(f"Received empty dataset from CVM API at {api_url}.")

    # 2. Filter out review_needed and excluded images strictly
    filtered_tasks = []
    skipped_review_count = 0
    skipped_excluded_count = 0
    test_cohort_count = 0
    val_cohort_count = 0
    train_cohort_count = 0

    for t in tasks:
        is_review_needed = bool(t.get("reviewNeeded") or t.get("review_needed"))
        is_excluded = bool(t.get("excluded") or t.get("isExcluded") or t.get("is_excluded"))
        if skip_review_needed and is_review_needed:
            skipped_review_count += 1
            continue
        if skip_excluded and is_excluded:
            skipped_excluded_count += 1
            continue

        is_test = bool(
            t.get("isTestCohort")
            or t.get("is_test_cohort")
            or t.get("testCohort")
            or t.get("test_cohort")
            or t.get("cohort") == "test"
        )
        is_val = bool(
            t.get("isValCohort")
            or t.get("is_val_cohort")
            or t.get("valCohort")
            or t.get("val_cohort")
            or t.get("cohort") == "val"
        )

        if is_test:
            test_cohort_count += 1
        elif is_val:
            val_cohort_count += 1
        else:
            train_cohort_count += 1

        filtered_tasks.append(t)

    print(f"\n📊 Total records in CVM dataset: {len(tasks)}")
    if skipped_review_count > 0:
        print(f"🚫 Skipped review_needed images: {skipped_review_count} (excluded from training)")
    if skipped_excluded_count > 0:
        print(f"🚫 Skipped excluded images: {skipped_excluded_count} (excluded from training)")
    print(f"🏋️ Train cohort images: {train_cohort_count} (training cohort)")
    print(f"🔍 Validation cohort images: {val_cohort_count} (fixed validation cohort)")
    print(f"🧪 Test cohort images: {test_cohort_count} (fixed test cohort)")
    print(f"🎯 Total images to load: {len(filtered_tasks)}")

    # 3. Save JSON export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"cvm_backend_export_{timestamp}.json"
    export_filepath = os.path.join(export_dir, export_filename)
    default_export_path = os.path.join(export_dir, "export.json")

    export_payload = {
        "totalExported": len(filtered_tasks),
        "skippedReviewNeeded": skipped_review_count,
        "skippedExcluded": skipped_excluded_count,
        "cohortCounts": {
            "train": train_cohort_count,
            "val": val_cohort_count,
            "test": test_cohort_count,
        },
        "dataSource": "cvm-backend-api",
        "timestamp": timestamp,
        "data": filtered_tasks,
    }
    with open(export_filepath, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=4)
    with open(default_export_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=4)
    print(f"💾 Saved CVM export to: {os.path.abspath(export_filepath)} and {default_export_path}\n")

    # 4. Download images strictly via HTTP API
    success_count = 0

    def download_single_cvm_image(task):
        filename = task.get("filename")
        if not filename:
            img_url = task.get("imageUrl") or task.get("data", {}).get("img")
            if img_url:
                filename = os.path.basename(img_url.split("?")[0])
            else:
                return f"⚠️ Skipping task {task.get('taskId')}: No filename"

        dest_path = os.path.join(img_dir, filename)

        img_url = task.get("imageUrl") or f"{api_url}/images/{filename}"
        try:
            resp = session.get(img_url, stream=True, timeout=20)
            if resp.status_code == 200:
                with open(dest_path, "wb") as f_out:
                    for chunk in resp.iter_content(1024 * 64):
                        f_out.write(chunk)
                return f"✅ Downloaded {filename}"
            else:
                return f"⚠️ Failed downloading {filename}: HTTP {resp.status_code}"
        except Exception as e:
            return f"❌ Network error downloading {filename}: {e}"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(download_single_cvm_image, task): task
            for task in filtered_tasks
        }

        for future in tqdm(
            as_completed(future_to_task),
            total=len(filtered_tasks),
            desc="Loading CVM Images (API)",
            unit="img",
            colour="green",
        ):
            try:
                res = future.result()
                if res and res.startswith("✅"):
                    success_count += 1
                else:
                    tqdm.write(res)
            except Exception as exc:
                tqdm.write(f"❌ Thread exception: {exc}")

    print(f"\n🚀 Done! Loaded {success_count}/{len(filtered_tasks)} images for training pipeline.")
    return success_count


def download_export_and_images(
    export_dir="data/raw/exports",
    img_dir="data/raw/images",
    data_source=None,
    api_url=None,
    api_token=None,
    skip_review_needed=True,
    skip_excluded=True,
    max_workers=10,
):
    """
    Entry point to download data and labels from CVM Backend API.
    """
    return download_from_cvm_api(
        export_dir=export_dir,
        img_dir=img_dir,
        api_url=api_url,
        api_token=api_token,
        skip_review_needed=skip_review_needed,
        skip_excluded=skip_excluded,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download dataset and export JSON for training strictly via CVM Backend API."
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("CVM_API_URL", "http://localhost:8000"),
        help="URL for CVM Backend API (default: http://localhost:8000 or $CVM_API_URL).",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=os.getenv("CVM_API_TOKEN", ""),
        help="Bearer authentication token for CVM Backend API.",
    )
    parser.add_argument(
        "--skip-review-needed",
        action="store_true",
        default=True,
        help="Skip images marked as review_needed (default: True).",
    )
    parser.add_argument(
        "--skip-excluded",
        action="store_true",
        default=True,
        help="Skip images marked as excluded (default: True).",
    )
    parser.add_argument(
        "--max-workers",
        "-j",
        type=int,
        default=int(os.getenv("MAX_WORKERS", "10")),
        help="Number of concurrent worker threads.",
    )
    parser.add_argument(
        "--out_img_dir",
        default="data/raw/images",
        help="Directory to save the downloaded images",
    )
    parser.add_argument(
        "--out_export_dir",
        default="data/raw/exports",
        help="Directory to save the raw JSON export file",
    )

    args = parser.parse_args()

    download_export_and_images(
        export_dir=args.out_export_dir,
        img_dir=args.out_img_dir,
        api_url=args.api_url,
        api_token=args.api_token,
        skip_review_needed=args.skip_review_needed,
        skip_excluded=args.skip_excluded,
        max_workers=args.max_workers,
    )
