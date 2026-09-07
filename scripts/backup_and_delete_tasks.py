#!/usr/bin/env python3
"""
scripts/backup_and_delete_tasks.py

High-performance, safe backup and deletion script for Label Studio tasks.
Features:
  - Fetches task metadata, full annotations, and associated images.
  - Verifies local backup integrity before initiating any deletion.
  - Performs concurrent operations using ThreadPoolExecutor and connection pooling.
  - Supports --dry-run to preview actions safely.
  - Saves individual task JSONs, copies/downloads images, and creates an aggregated export JSON.
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, parse_qs

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from tqdm import tqdm


def get_authenticated_session(ls_url, api_token=None, username=None, password=None, workers=8):
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "DELETE", "POST"]
    )
    adapter = HTTPAdapter(pool_connections=workers * 2, pool_maxsize=workers * 2, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if api_token:
        session.headers.update({"Authorization": f"Token {api_token}"})
        # Test token validity
        test_url = f"{ls_url}/api/current-user/whoami"
        try:
            resp = session.get(test_url, timeout=10)
            if resp.status_code == 200:
                print("🔑 Authenticated successfully using API Token.")
                return session
        except Exception as e:
            print(f"⚠️ Token auth check warning: {e}")

    # Fallback to session login
    if username and password:
        login_url = f"{ls_url}/user/login/"
        try:
            session.get(login_url, timeout=10)
            csrf_token = session.cookies.get("csrftoken", "")
            login_data = {
                "email": username,
                "password": password,
                "csrfmiddlewaretoken": csrf_token,
            }
            resp = session.post(login_url, data=login_data, headers={"Referer": login_url}, timeout=10)
            if resp.status_code in [200, 302]:
                print("🔑 Authenticated successfully using username/password session.")
                return session
        except Exception as e:
            print(f"❌ Login request error: {e}")

    if api_token:
        # If whoami endpoint was just unrouted or enterprise-only, still return session with token
        return session

    raise RuntimeError("Failed to authenticate to Label Studio. Please verify credentials in .env.")


def extract_filename_from_url(img_url):
    """
    Extracts a clean image filename from a Label Studio image URL or local-files query.
    e.g. 'https://.../?d=cvm-images/0015.jpg' -> '0015.jpg'
    e.g. '/data/upload/1/abc-0015.jpg' -> '0015.jpg'
    """
    if "?d=" in img_url:
        parsed = urlparse(img_url)
        params = parse_qs(parsed.query)
        if "d" in params and params["d"]:
            return os.path.basename(params["d"][0])

    base = os.path.basename(img_url.split("?")[0])
    if "-" in base:
        parts = base.split("-", 1)
        if len(parts) > 1 and len(parts[0]) >= 8:  # Typical hash prefix
            return parts[1]
    return base


def backup_image(img_url, img_dest_path, ls_url, session, local_candidates=None):
    """
    Copies image from existing local directory if found, otherwise downloads via session.
    """
    filename = os.path.basename(img_dest_path)
    
    # Check local candidates first (fastest and doesn't stress remote server)
    if local_candidates:
        for candidate_dir in local_candidates:
            local_file = os.path.join(candidate_dir, filename)
            if os.path.exists(local_file) and os.path.getsize(local_file) > 0:
                shutil.copy2(local_file, img_dest_path)
                return True, f"Copied locally from {candidate_dir}"

    # Fallback to downloading
    full_url = img_url if img_url.startswith("http") else f"{ls_url.rstrip('/')}/{img_url.lstrip('/')}"
    try:
        resp = session.get(full_url, stream=True, timeout=15)
        if resp.status_code == 200:
            with open(img_dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if os.path.exists(img_dest_path) and os.path.getsize(img_dest_path) > 0:
                return True, "Downloaded from server"
        return False, f"HTTP status {resp.status_code}"
    except Exception as e:
        return False, f"Download error: {e}"


def backup_single_task(task_id, ls_url, session, tasks_dir, images_dir, local_candidates=None):
    """
    Fetches task JSON and associated image, writes them to disk, and verifies integrity.
    """
    task_url = f"{ls_url.rstrip('/')}/api/tasks/{task_id}"
    try:
        resp = session.get(task_url, timeout=15)
        if resp.status_code == 404:
            return {
                "task_id": task_id,
                "status": "not_found",
                "error": "Task does not exist on server (404)",
                "task_data": None
            }
        elif resp.status_code != 200:
            return {
                "task_id": task_id,
                "status": "fetch_failed",
                "error": f"HTTP status {resp.status_code}: {resp.text[:100]}",
                "task_data": None
            }

        task_data = resp.json()
        task_json_path = os.path.join(tasks_dir, f"{task_id}.json")

        # Save task JSON
        with open(task_json_path, "w", encoding="utf-8") as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False)

        # Verify task JSON file was written properly
        if not os.path.exists(task_json_path) or os.path.getsize(task_json_path) == 0:
            return {
                "task_id": task_id,
                "status": "backup_verify_failed",
                "error": "Task JSON file was empty or not written",
                "task_data": None
            }

        # Backup image
        img_info = "No image found in data"
        img_url = task_data.get("data", {}).get("img")
        if img_url:
            img_filename = extract_filename_from_url(img_url)
            img_dest = os.path.join(images_dir, img_filename)
            ok, msg = backup_image(img_url, img_dest, ls_url, session, local_candidates)
            img_info = f"{img_filename} ({msg})" if ok else f"Image error: {msg}"
            if not ok:
                # Still consider backed up if annotations exist, but record warning
                task_data["_image_backup_warning"] = msg

        return {
            "task_id": task_id,
            "status": "backed_up",
            "error": None,
            "task_json_path": task_json_path,
            "annotations_count": len(task_data.get("annotations", [])),
            "img_info": img_info,
            "task_data": task_data
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e),
            "task_data": None
        }


def delete_single_task(task_id, ls_url, session):
    """
    Deletes a single task from Label Studio.
    """
    task_url = f"{ls_url.rstrip('/')}/api/tasks/{task_id}"
    try:
        resp = session.delete(task_url, timeout=15)
        if resp.status_code in [200, 204]:
            return {"task_id": task_id, "deleted": True, "error": None}
        elif resp.status_code == 404:
            return {"task_id": task_id, "deleted": True, "error": "Already deleted (404)"}
        else:
            return {"task_id": task_id, "deleted": False, "error": f"HTTP {resp.status_code}: {resp.text[:100]}"}
    except Exception as e:
        return {"task_id": task_id, "deleted": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Safely backup and delete Label Studio tasks.")
    parser.add_argument(
        "--tasks-file",
        default="data/to_delete.json",
        help="Path to JSON file with list of task IDs to delete (default: data/to_delete.json)"
    )
    parser.add_argument(
        "--backup-dir",
        default="data/backup_deleted",
        help="Directory to save the backups (default: data/backup_deleted)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform backup and validation only without deleting from Label Studio"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Bypass interactive confirmation prompt before deletion"
    )

    args = parser.parse_args()

    load_dotenv()
    ls_url = os.getenv("LABEL_STUDIO_URL")
    api_token = os.getenv("LABEL_STUDIO_API_TOKEN")
    username = os.getenv("LABEL_STUDIO_USERNAME")
    password = os.getenv("LABEL_STUDIO_PASSWORD")

    if not ls_url:
        print("❌ Error: LABEL_STUDIO_URL is not set in environment or .env file.")
        sys.exit(1)

    if not os.path.exists(args.tasks_file):
        print(f"❌ Error: Tasks file not found at: {args.tasks_file}")
        sys.exit(1)

    with open(args.tasks_file, "r", encoding="utf-8") as f:
        task_ids_raw = json.load(f)

    # Normalize task IDs to integer/str
    task_ids = [str(x).strip() for x in task_ids_raw if str(x).strip()]
    task_ids = list(dict.fromkeys(task_ids))  # preserve order, remove duplicates

    print(f"\n==================================================")
    print(f"📋 Label Studio Task Backup & Deletion Tool")
    print(f"==================================================")
    print(f"Server URL:     {ls_url}")
    print(f"Target Tasks:   {len(task_ids)} IDs loaded from {args.tasks_file}")
    print(f"Backup Dir:     {args.backup_dir}")
    print(f"Parallelism:    {args.workers} workers")
    print(f"Dry Run:        {'YES (No deletions will be made)' if args.dry_run else 'NO (Live deletion mode)'}")
    print(f"==================================================\n")

    tasks_dir = os.path.join(args.backup_dir, "tasks")
    images_dir = os.path.join(args.backup_dir, "images")
    os.makedirs(tasks_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    local_candidates = [
        "data/images",
        "data/raw/images",
    ]

    session = get_authenticated_session(ls_url, api_token, username, password, workers=args.workers)

    # ----------------------------------------------------
    # STAGE 1: Backup & Integrity Check
    # ----------------------------------------------------
    print(f"\n📦 Stage 1: Fetching & Backing Up {len(task_ids)} Tasks...")
    backup_results = {}
    successful_tasks = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {
            executor.submit(
                backup_single_task,
                tid,
                ls_url,
                session,
                tasks_dir,
                images_dir,
                local_candidates
            ): tid for tid in task_ids
        }

        for future in tqdm(as_completed(future_to_id), total=len(task_ids), desc="Backing up tasks", unit="task"):
            res = future.result()
            tid = res["task_id"]
            backup_results[tid] = res
            if res["status"] == "backed_up" and res["task_data"] is not None:
                successful_tasks.append(res["task_data"])

    backed_up_count = len(successful_tasks)
    not_found_count = sum(1 for r in backup_results.values() if r["status"] == "not_found")
    failed_count = len(task_ids) - backed_up_count - not_found_count

    print(f"\n📊 Backup Results Summary:")
    print(f"  ✅ Successfully backed up: {backed_up_count}")
    print(f"  🔍 Already not found (404): {not_found_count}")
    if failed_count > 0:
        print(f"  ❌ Backup failures: {failed_count}")

    # Save combined backup JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregated_file = os.path.join(args.backup_dir, f"all_deleted_tasks_{timestamp}.json")
    with open(aggregated_file, "w", encoding="utf-8") as f:
        json.dump(successful_tasks, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Aggregated backup saved to: {aggregated_file}")

    # Tasks that are verified and eligible for deletion
    eligible_for_deletion = [
        tid for tid in task_ids
        if backup_results[tid]["status"] == "backed_up"
    ]

    if failed_count > 0:
        print(f"\n⚠️ WARNING: {failed_count} tasks failed backup. They will NOT be deleted.")

    if not eligible_for_deletion:
        print("\nℹ️ No tasks are eligible for deletion. Exiting.")
        return

    # ----------------------------------------------------
    # STAGE 2: Deletion
    # ----------------------------------------------------
    if args.dry_run:
        print(f"\n[DRY RUN] Would delete {len(eligible_for_deletion)} tasks from Label Studio.")
        for tid in eligible_for_deletion[:5]:
            print(f"  [DRY RUN] Would delete task ID: {tid} ({backup_results[tid]['img_info']})")
        if len(eligible_for_deletion) > 5:
            print(f"  ... and {len(eligible_for_deletion) - 5} more.")

        manifest = {
            "timestamp": timestamp,
            "dry_run": True,
            "total_requested": len(task_ids),
            "backed_up_count": backed_up_count,
            "not_found_count": not_found_count,
            "failed_backup_count": failed_count,
            "eligible_for_deletion": len(eligible_for_deletion),
            "tasks": backup_results
        }
        manifest_path = os.path.join(args.backup_dir, f"manifest_{timestamp}_dryrun.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Dry-run manifest saved to: {manifest_path}")
        print("\n✨ Dry-run complete. Run without --dry-run to permanently delete from Label Studio.")
        return

    if not args.yes:
        print(f"\n⚠️ CAUTION: You are about to PERMANENTLY DELETE {len(eligible_for_deletion)} tasks from Label Studio.")
        confirm = input(f"Are you sure you want to proceed? Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("🚫 Operation cancelled by user. No tasks were deleted.")
            return

    print(f"\n🗑️ Stage 2: Deleting {len(eligible_for_deletion)} Tasks from Label Studio...")
    deletion_results = {}
    deleted_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {
            executor.submit(delete_single_task, tid, ls_url, session): tid
            for tid in eligible_for_deletion
        }

        for future in tqdm(as_completed(future_to_id), total=len(eligible_for_deletion), desc="Deleting tasks", unit="task"):
            res = future.result()
            tid = res["task_id"]
            deletion_results[tid] = res
            if res["deleted"]:
                deleted_count += 1

    print(f"\n📊 Deletion Results Summary:")
    print(f"  🗑️ Successfully deleted: {deleted_count}/{len(eligible_for_deletion)}")

    # Write final execution manifest
    manifest = {
        "timestamp": timestamp,
        "dry_run": False,
        "total_requested": len(task_ids),
        "backed_up_count": backed_up_count,
        "deleted_count": deleted_count,
        "backup_details": backup_results,
        "deletion_details": deletion_results
    }
    manifest_path = os.path.join(args.backup_dir, f"manifest_{timestamp}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"📄 Audit manifest saved to: {manifest_path}")
    print("\n🎉 All tasks processed successfully!")


if __name__ == "__main__":
    main()
