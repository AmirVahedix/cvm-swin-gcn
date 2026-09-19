#!/usr/bin/env python3
"""
dataset/test/upload_to_label_studio.py

Standalone script to split test images from the local 'images' directory
equally into 3 groups (49 images each) and upload each group to Label Studio
projects (default project IDs: 5, 6, 7).

Key Features:
- Reads environment configuration (LABEL_STUDIO_URL, LABEL_STUDIO_API_TOKEN, etc.) from .env
- Deterministic splitting: Sorts the images and assigns 49 images per project
- Duplicate Prevention: Checks existing project tasks first and skips already-uploaded images
- Safe Retries: Does not auto-retry non-idempotent POST requests to avoid duplicates
- Deduplication: Supports --dedup to clean up any duplicates across the projects
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
    Fetches all tasks currently in the specified project.
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
    Deletes duplicate tasks in a project, keeping only the earliest task ID for each image.
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

    print(f"\n🔍 Project {project_id}: Found {len(to_delete)} duplicate tasks to remove:")
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


def split_list_into_chunks(items: list, num_chunks: int) -> list:
    """Splits a list into num_chunks nearly equal parts."""
    n = len(items)
    base_size = n // num_chunks
    remainder = n % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + base_size + (1 if i < remainder else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Split test images equally and upload to 3 Label Studio projects (5, 6, 7)."
    )
    parser.add_argument(
        "--project-ids",
        type=int,
        nargs="+",
        default=[5, 6, 7],
        help="List of Label Studio project IDs (default: 5 6 7)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "images"),
        help="Path to directory containing images (default: ./images relative to script)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent upload workers (default: 4)")
    parser.add_argument("--dedup", action="store_true", help="Find and delete duplicate tasks in the target projects")
    parser.add_argument("--dry-run", action="store_true", help="Preview split and actions without uploading or deleting")

    args = parser.parse_args()
    project_ids = args.project_ids

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

    total_images = len(image_files)
    print(f"📁 Found {total_images} images in: {images_dir}")
    print(f"🎯 Target Project IDs: {project_ids}")
    print(f"🌐 Label Studio URL: {ls_url}")

    if total_images == 0:
        print(f"⚠️ No image files found to process.")
        sys.exit(0)

    # Split images equally across projects
    image_groups = split_list_into_chunks(image_files, len(project_ids))

    print("\n📦 Image Distribution Plan:")
    for pid, group in zip(project_ids, image_groups):
        print(f"   - Project {pid}: {len(group)} images ({group[0].name} ... {group[-1].name})")

    session = get_authenticated_session(
        ls_url=ls_url,
        api_token=api_token,
        username=username,
        password=password,
        workers=args.workers,
    )

    # Verify each project exists and retrieve project titles
    project_meta = {}
    for pid in project_ids:
        try:
            p_resp = session.get(f"{ls_url}/api/projects/{pid}", timeout=10)
            if p_resp.status_code != 200:
                print(f"❌ Failed to access project {pid}. HTTP {p_resp.status_code}: {p_resp.text[:200]}")
                sys.exit(1)
            meta = p_resp.json()
            project_meta[pid] = meta
            print(f"📋 Project {pid} Verified: '{meta.get('title')}' (Current tasks: {meta.get('task_number', 0)})")
        except Exception as e:
            print(f"❌ Error verifying project {pid}: {e}")
            sys.exit(1)

    # Deduplication mode
    if args.dedup:
        print("\n🧹 Running deduplication check on target projects...")
        for pid in project_ids:
            tasks_map = get_existing_project_tasks(session, ls_url, pid)
            if args.dry_run:
                print(f"🔎 Dry run: checking project {pid} ({len(tasks_map)} unique images)...")
                for fn, tids in tasks_map.items():
                    if len(tids) > 1:
                        print(f"  Project {pid}: Would keep Task {sorted(tids)[0]} and delete {sorted(tids)[1:]} for {fn}")
            else:
                deduplicate_project_tasks(session, ls_url, pid, tasks_map)
        return

    if args.dry_run:
        print("\n🔎 Dry-run enabled. No images will be uploaded.")
        for pid, group in zip(project_ids, image_groups):
            print(f"\n--- Project {pid} ('{project_meta[pid].get('title')}') [{len(group)} images] ---")
            for i, img in enumerate(group[:5], start=1):
                print(f"  {i}. {img.name}")
            if len(group) > 5:
                print(f"  ... and {len(group) - 5} more.")
        return

    # Upload each group to its respective project
    total_uploaded = 0
    total_skipped = 0
    all_failed = []

    for pid, group in zip(project_ids, image_groups):
        title = project_meta[pid].get("title", f"Project {pid}")
        print(f"\n" + "=" * 60)
        print(f"📤 Uploading {len(group)} images to Project {pid} ('{title}')...")
        print("=" * 60)

        # Check existing tasks in this project to prevent duplicates
        existing_tasks = get_existing_project_tasks(session, ls_url, pid)
        pending_upload = [img for img in group if img.name not in existing_tasks]
        already_uploaded = [img for img in group if img.name in existing_tasks]

        if already_uploaded:
            print(f"ℹ️ {len(already_uploaded)} images already exist in Project {pid} (skipping them).")
            total_skipped += len(already_uploaded)

        if not pending_upload:
            print(f"✅ All {len(group)} images are already present in Project {pid}!")
            continue

        print(f"🚀 Uploading {len(pending_upload)} images with {args.workers} concurrent workers...")

        proj_success = 0
        proj_failed = []

        if tqdm:
            pbar = tqdm(total=len(pending_upload), desc=f"Project {pid}", unit="img")
        else:
            pbar = None

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(upload_single_image, session, ls_url, pid, img_path): img_path
                for img_path in pending_upload
            }

            for future in as_completed(future_to_file):
                success, filename, err = future.result()
                if success:
                    proj_success += 1
                else:
                    proj_failed.append((filename, err))

                if pbar:
                    pbar.update(1)
                else:
                    done = proj_success + len(proj_failed)
                    if done % 10 == 0 or done == len(pending_upload):
                        print(f"Progress Project {pid}: {done}/{len(pending_upload)} done...")

        if pbar:
            pbar.close()

        total_uploaded += proj_success
        all_failed.extend([(pid, fn, err) for fn, err in proj_failed])
        print(f"✅ Project {pid} Upload Complete: {proj_success}/{len(pending_upload)} uploaded successfully.")

    # Final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL SUMMARY:")
    print(f"   - Total images in dataset/test/images: {total_images}")
    print(f"   - Successfully uploaded:              {total_uploaded}")
    print(f"   - Already present (skipped):          {total_skipped}")
    if all_failed:
        print(f"   - Failed uploads ({len(all_failed)}):")
        for pid, fn, err in all_failed:
            print(f"     [Project {pid}] {fn}: {err}")
    else:
        print("🎉 All 3 projects (5, 6, 7) have their complete 49-image test sets!")
    print("=" * 60)


if __name__ == "__main__":
    main()
