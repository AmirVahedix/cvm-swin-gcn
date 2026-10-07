#!/usr/bin/env python3
"""
dataset/test/upload_to_label_studio.py

Standalone script to upload all test images from the local 'images' directory
directly into Label Studio Project ID 4 (or specified project).

Key Features:
- Reads environment configuration (LABEL_STUDIO_URL, LABEL_STUDIO_API_TOKEN, etc.) from .env
- Duplicate Prevention: Queries existing project tasks first and skips images already uploaded
- Deduplication: Provides a --dedup flag to detect and remove duplicate tasks in the project
- Safe Network Handling: Does NOT auto-retry non-idempotent POST requests to avoid duplicates
"""

import os
import sys
import mimetypes
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def find_and_load_env():
    """Locate and load .env from repo root or parent directories."""
    current = Path(__file__).resolve().parent
    for p in [current, current.parent, current.parent.parent, current.parent.parent.parent]:
        env_path = p / ".env"
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path)
            return env_path
    load_dotenv()
    return None


def get_authenticated_session(ls_url: str, api_token: str = None, username: str = None, password: str = None, workers: int = 4):
    """
    Creates a requests.Session with connection pooling.
    Note: POST is intentionally excluded from auto-retries to prevent creating duplicate tasks.
    """
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]  # ONLY idempotent GET requests are retried
    )
    adapter = HTTPAdapter(pool_connections=workers * 2, pool_maxsize=workers * 2, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if api_token:
        session.headers.update({"Authorization": f"Token {api_token}"})
        try:
            resp = session.get(f"{ls_url}/api/current-user/whoami", timeout=10)
            if resp.status_code == 200:
                print("🔑 Authenticated successfully using API Token.")
                return session
        except Exception as e:
            print(f"⚠️ Token auth verification warning: {e}")

    # Fallback to session login
    if username and password:
        login_url = f"{ls_url}/user/login/"
        try:
            resp = session.get(login_url, timeout=10)
            csrf_token = session.cookies.get("csrftoken", "")
            login_data = {
                "email": username,
                "password": password,
                "csrfmiddlewaretoken": csrf_token,
            }
            login_resp = session.post(
                login_url,
                data=login_data,
                headers={"Referer": login_url},
                timeout=10,
            )
            if login_resp.status_code in [200, 302]:
                print("🔑 Authenticated successfully via username/password session.")
                return session
            else:
                print(f"❌ Session login failed with status code {login_resp.status_code}.")
        except Exception as e:
            print(f"❌ Error logging in via session: {e}")

    return session


def get_existing_project_tasks(session: requests.Session, ls_url: str, project_id: int):
    """
    Fetches all tasks currently in the project.
    Returns a dict mapping clean_filename -> list of task_ids.
    """
    tasks_by_filename = defaultdict(list)
    page = 1
    page_size = 250

    while True:
        url = f"{ls_url}/api/tasks?project={project_id}&page={page}&page_size={page_size}"
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"⚠️ Warning: Could not fetch existing tasks for project {project_id} (HTTP {resp.status_code})")
            break

        data = resp.json()
        tasks = data.get("tasks", []) if isinstance(data, dict) else data
        if not tasks:
            break

        for t in tasks:
            tid = t.get("id")
            img_val = t.get("data", {}).get("image") or t.get("data", {}).get("img") or ""
            fn = os.path.basename(img_val)
            clean_fn = fn.split("-", 1)[1] if "-" in fn else fn
            if clean_fn:
                tasks_by_filename[clean_fn].append(tid)

        total_fetched = page * page_size
        total_available = data.get("total", len(tasks)) if isinstance(data, dict) else len(tasks)
        if total_fetched >= total_available or len(tasks) < page_size:
            break
        page += 1

    return tasks_by_filename


def deduplicate_project_tasks(session: requests.Session, ls_url: str, project_id: int, tasks_by_filename: dict):
    """
    Deletes duplicate tasks in the project, keeping only the earliest task ID for each image.
    """
    to_delete = []
    for fn, tids in tasks_by_filename.items():
        if len(tids) > 1:
            sorted_tids = sorted(tids)
            keep_id = sorted_tids[0]
            for extra_id in sorted_tids[1:]:
                to_delete.append((extra_id, fn, keep_id))

    if not to_delete:
        print(f"✨ Project {project_id}: No duplicates found! Every image has exactly one task.")
        return 0

    print(f"\n🔍 Project {project_id}: Found {len(to_delete)} duplicate tasks to remove across {len([fn for fn, tids in tasks_by_filename.items() if len(tids) > 1])} images:")
    for extra_id, fn, keep_id in to_delete:
        print(f"   - Removing Task ID {extra_id} (duplicate of {keep_id} for '{fn}')")

    deleted_count = 0
    for extra_id, fn, keep_id in to_delete:
        try:
            r = session.delete(f"{ls_url}/api/tasks/{extra_id}", timeout=15)
            if r.status_code in [200, 204]:
                deleted_count += 1
            else:
                print(f"⚠️ Failed to delete Task ID {extra_id}: HTTP {r.status_code}")
        except Exception as e:
            print(f"⚠️ Error deleting Task ID {extra_id}: {e}")

    print(f"🧹 Project {project_id}: Successfully deleted {deleted_count}/{len(to_delete)} duplicate tasks.\n")
    return deleted_count


def upload_single_image(session: requests.Session, ls_url: str, project_id: int, image_path: Path):
    """Uploads a single image to Label Studio project import endpoint."""
    filename = image_path.name
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "application/octet-stream"

    import_url = f"{ls_url}/api/projects/{project_id}/import"
    try:
        with open(image_path, "rb") as f:
            files = {"file": (filename, f, mime_type)}
            resp = session.post(import_url, files=files, timeout=60)

        if resp.status_code in [200, 201]:
            return True, filename, None
        else:
            return False, filename, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, filename, str(e)


def main():
    parser = argparse.ArgumentParser(description="Upload test images to Label Studio Project ID 4 (or specified project).")
    parser.add_argument("--project-id", type=int, default=4, help="Label Studio project ID (default: 4)")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "images"),
        help="Path to directory containing images (default: ./images relative to script)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent upload workers (default: 4)")
    parser.add_argument("--dedup", action="store_true", help="Find and delete duplicate tasks in the project")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without uploading or deleting")

    args = parser.parse_args()

    env_path = find_and_load_env()
    if env_path:
        print(f"📄 Loaded environment from: {env_path}")
    else:
        print("⚠️ No .env file found explicitly; relying on existing environment variables.")

    ls_url = (os.getenv("LABEL_STUDIO_URL") or "").rstrip("/")
    api_token = os.getenv("LABEL_STUDIO_API_TOKEN")
    username = os.getenv("LABEL_STUDIO_USERNAME")
    password = os.getenv("LABEL_STUDIO_PASSWORD")

    if not ls_url:
        print("❌ Error: LABEL_STUDIO_URL is not set in environment or .env.")
        sys.exit(1)

    images_dir = Path(args.images_dir).resolve()
    if not images_dir.is_dir():
        print(f"❌ Error: Images directory not found: {images_dir}")
        sys.exit(1)

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
    image_files = sorted(
        [
            f
            for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions and not f.name.startswith(".")
        ],
        key=lambda p: p.name,
    )

    print(f"📁 Found {len(image_files)} images in local directory: {images_dir}")
    print(f"🎯 Target Label Studio Project ID: {args.project_id}")
    print(f"🌐 Label Studio URL: {ls_url}")

    if not image_files:
        print(f"⚠️ No image files found in: {images_dir}")
        sys.exit(0)

    session = get_authenticated_session(
        ls_url=ls_url,
        api_token=api_token,
        username=username,
        password=password,
        workers=args.workers,
    )

    # Verify project exists
    try:
        proj_resp = session.get(f"{ls_url}/api/projects/{args.project_id}", timeout=10)
        if proj_resp.status_code != 200:
            print(f"❌ Failed to access project {args.project_id}. HTTP {proj_resp.status_code}: {proj_resp.text[:200]}")
            sys.exit(1)
        proj_info = proj_resp.json()
        current_task_count = proj_info.get("task_number", 0)
        print(f"📋 Project Title: '{proj_info.get('title')}' (Current total tasks: {current_task_count})")
    except Exception as e:
        print(f"❌ Error checking project {args.project_id}: {e}")
        sys.exit(1)

    # Fetch existing tasks in project
    print("🔎 Checking existing tasks in project...")
    existing_tasks = get_existing_project_tasks(session, ls_url, args.project_id)
    print(f"📊 Project currently has {sum(len(v) for v in existing_tasks.values())} total tasks covering {len(existing_tasks)} unique images.")

    # Deduplication mode
    if args.dedup:
        if args.dry_run:
            print("🔎 Dry run mode: would deduplicate tasks.")
            for fn, tids in existing_tasks.items():
                if len(tids) > 1:
                    print(f"  Would keep Task {sorted(tids)[0]} and delete {sorted(tids)[1:]} for {fn}")
            return
        deduplicate_project_tasks(session, ls_url, args.project_id, existing_tasks)
        existing_tasks = get_existing_project_tasks(session, ls_url, args.project_id)
        print(f"📊 Remaining tasks: {sum(len(v) for v in existing_tasks.values())}")
        return

    # Filter out images that are already uploaded
    files_to_upload = [img for img in image_files if img.name not in existing_tasks]
    already_uploaded = [img for img in image_files if img.name in existing_tasks]

    print(f"ℹ️ Already uploaded: {len(already_uploaded)} images (will be skipped)")
    print(f"📥 Pending upload:    {len(files_to_upload)} images")

    if not files_to_upload:
        print("\n✅ All local images are already present in Label Studio Project! Nothing to upload.")
        dup_count = sum(len(v) - 1 for v in existing_tasks.values() if len(v) > 1)
        if dup_count > 0:
            print(f"\n💡 Notice: There are {dup_count} duplicate tasks in the project.")
            print(f"   You can run: python dataset/test/upload_to_label_studio.py --dedup to remove them.")
        return

    if args.dry_run:
        print("\n🔎 Dry run mode enabled. Files that would be uploaded:")
        for idx, img in enumerate(files_to_upload[:10], start=1):
            print(f"  {idx}. {img.name}")
        if len(files_to_upload) > 10:
            print(f"  ... and {len(files_to_upload) - 10} more.")
        return

    print(f"\n🚀 Starting upload of {len(files_to_upload)} images to Project {args.project_id} with {args.workers} workers...")

    success_count = 0
    failed_uploads = []

    if tqdm:
        pbar = tqdm(total=len(files_to_upload), desc=f"Project {args.project_id}", unit="img")
    else:
        pbar = None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {
            executor.submit(upload_single_image, session, ls_url, args.project_id, img_path): img_path
            for img_path in files_to_upload
        }

        for future in as_completed(future_to_file):
            success, filename, err = future.result()
            if success:
                success_count += 1
            else:
                failed_uploads.append((filename, err))

            if pbar:
                pbar.update(1)
            else:
                completed = success_count + len(failed_uploads)
                if completed % 10 == 0 or completed == len(files_to_upload):
                    print(f"Progress: {completed}/{len(files_to_upload)} uploaded...")

    if pbar:
        pbar.close()

    print("\n" + "=" * 50)
    print(f"✅ Successfully uploaded: {success_count}/{len(files_to_upload)} images")
    if failed_uploads:
        print(f"❌ Failed uploads ({len(failed_uploads)}):")
        for fn, err in failed_uploads:
            print(f"   - {fn}: {err}")
    else:
        print(f"🎉 All {success_count} images uploaded successfully to Project {args.project_id}!")
    print("=" * 50)


if __name__ == "__main__":
    main()
