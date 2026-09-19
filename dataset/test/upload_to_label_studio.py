#!/usr/bin/env python3
"""
dataset/test/upload_to_label_studio.py

Standalone script to upload all test images from the local 'images' directory
directly into Label Studio Project ID 2 (or specified project).

Reads environment configuration (LABEL_STUDIO_URL, LABEL_STUDIO_API_TOKEN,
LABEL_STUDIO_USERNAME, LABEL_STUDIO_PASSWORD) from .env.
"""

import os
import sys
import mimetypes
import argparse
from pathlib import Path
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
    Creates a requests.Session with connection pooling and retries.
    Authenticates via API token, falling back to session login if needed.
    """
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
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
            resp = session.post(import_url, files=files, timeout=45)

        if resp.status_code in [200, 201]:
            return True, filename, None
        else:
            return False, filename, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, filename, str(e)


def main():
    parser = argparse.ArgumentParser(description="Upload test images to Label Studio Project 2.")
    parser.add_argument("--project-id", type=int, default=2, help="Label Studio project ID (default: 2)")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "images"),
        help="Path to directory containing images (default: ./images relative to script)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent upload workers (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="List images found without uploading")

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

    if not image_files:
        print(f"⚠️ No image files found in: {images_dir}")
        sys.exit(0)

    print(f"📁 Found {len(image_files)} images in: {images_dir}")
    print(f"🎯 Target Label Studio Project ID: {args.project_id}")
    print(f"🌐 Label Studio URL: {ls_url}")

    if args.dry_run:
        print("🔎 Dry run mode enabled. Files to upload:")
        for idx, img in enumerate(image_files[:10], start=1):
            print(f"  {idx}. {img.name}")
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more.")
        return

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
        print(f"📋 Project Title: {proj_info.get('title')} (Current tasks: {proj_info.get('task_number', 0)})")
    except Exception as e:
        print(f"❌ Error checking project {args.project_id}: {e}")
        sys.exit(1)

    print(f"\n🚀 Starting upload with {args.workers} concurrent workers...")

    success_count = 0
    failed_uploads = []

    if tqdm:
        pbar = tqdm(total=len(image_files), desc="Uploading images", unit="img")
    else:
        pbar = None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {
            executor.submit(upload_single_image, session, ls_url, args.project_id, img_path): img_path
            for img_path in image_files
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
                if completed % 10 == 0 or completed == len(image_files):
                    print(f"Progress: {completed}/{len(image_files)} uploaded...")

    if pbar:
        pbar.close()

    print("\n" + "=" * 50)
    print(f"✅ Successfully uploaded: {success_count}/{len(image_files)} images")
    if failed_uploads:
        print(f"❌ Failed uploads ({len(failed_uploads)}):")
        for fn, err in failed_uploads:
            print(f"   - {fn}: {err}")
    else:
        print("🎉 All images uploaded successfully to Label Studio!")
    print("=" * 50)


if __name__ == "__main__":
    main()
