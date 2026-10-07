import json
import os
import shutil
import requests
import argparse
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

load_dotenv()


def extract_filename_from_url(img_url: str) -> str:
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


def process_single_task(task, session, img_dir, ls_url):
    """
    Worker function to process a single image download.
    """
    img_url = task.get("data", {}).get("img") or task.get("data", {}).get("image") or task.get("file_upload")
    if not img_url:
        return f"⚠️ Skipping task {task.get('id')}: No image URL found"

    clean_filename = extract_filename_from_url(img_url)
    img_path = os.path.join(img_dir, clean_filename)

    # Download image
    full_img_url = img_url if img_url.startswith("http") else f"{ls_url.rstrip('/')}/{img_url.lstrip('/')}"
    try:
        response = session.get(full_img_url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(img_path, "wb") as f_img:
                for chunk in response.iter_content(1024):
                    f_img.write(chunk)
        else:
            return f"⚠️ Skipping {clean_filename}: Image download failed (Status {response.status_code})"
    except Exception as e:
        return f"❌ Network Error on {clean_filename}: {e}"

    return f"✅ Downloaded {clean_filename}"


def download_export_and_images(
    export_dir="data/exports", img_dir="data/images"
):
    ls_url = (os.getenv("LABEL_STUDIO_URL") or "").rstrip("/")
    project_id = os.getenv("LABEL_STUDIO_PROJECT_ID")
    api_token = os.getenv("LABEL_STUDIO_API_TOKEN")
    username = os.getenv("LABEL_STUDIO_USERNAME")
    password = os.getenv("LABEL_STUDIO_PASSWORD")

    if not ls_url:
        raise ValueError("LABEL_STUDIO_URL is not set in environment or .env.")
    if not project_id:
        raise ValueError("LABEL_STUDIO_PROJECT_ID is not set in environment or .env.")

    clear_directory(img_dir)

    # Ensure the export directory exists
    os.makedirs(export_dir, exist_ok=True)

    session = requests.Session()

    max_workers = 10
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers, pool_maxsize=max_workers
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    authenticated = False

    # 1. Primary auth: API Token (recommended, stateless, and reliable)
    if api_token:
        session.headers.update({"Authorization": f"Token {api_token}"})
        try:
            whoami_resp = session.get(f"{ls_url}/api/current-user/whoami", timeout=10)
            if whoami_resp.status_code == 200:
                print("🔑 Authenticated successfully using API Token.")
                authenticated = True
            else:
                print(f"⚠️ API Token check warning (HTTP {whoami_resp.status_code}): {whoami_resp.text[:100]}")
                # Keep token header just in case whoami endpoint is restricted
                authenticated = True
        except Exception as e:
            print(f"⚠️ Token auth verification error: {e}")
            authenticated = True

    # 2. Fallback auth: Username/Password session login
    if not authenticated and username and password:
        login_url = f"{ls_url}/user/login/"
        try:
            session.get(login_url, timeout=10)
            csrf_token = session.cookies.get("csrftoken", "")
            login_data = {
                "email": username,
                "password": password,
                "csrfmiddlewaretoken": csrf_token,
            }
            login_response = session.post(
                login_url, data=login_data, headers={"Referer": login_url}, timeout=10
            )
            if login_response.status_code in [200, 302] and ("sessionid" in session.cookies or login_response.status_code == 302):
                print("🔑 Authenticated successfully via session login.")
                authenticated = True
            else:
                print(f"❌ Session login failed (HTTP {login_response.status_code}).")
        except Exception as e:
            print(f"❌ Session login error: {e}")

    if not authenticated:
        raise RuntimeError("Failed to authenticate to Label Studio. Please verify LABEL_STUDIO_API_TOKEN or username/password in .env.")

    print(f"📥 Fetching latest JSON export for project ID: {project_id}...")
    export_url = f"{ls_url}/api/projects/{project_id}/export?exportType=JSON"
    export_response = session.get(export_url)

    if export_response.status_code != 200:
        print(
            f"❌ Failed to download export. HTTP Status: {export_response.status_code}"
        )
        return

    tasks = export_response.json()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"project_{project_id}_export_{timestamp}.json"

    # Construct the full path for the export file
    export_filepath = os.path.join(export_dir, export_filename)

    try:
        with open(export_filepath, "w") as json_file:
            json.dump(tasks, json_file, indent=4)
        print(f"💾 Saved raw JSON export to: {os.path.abspath(export_filepath)}\n")
    except Exception as e:
        print(f"⚠️ Could not save JSON file to disk. Reason: {e}\n")

    success_count = 0

    # ThreadPoolExecutor with tqdm progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                process_single_task,
                task,
                session,
                img_dir,
                ls_url,
            ): task
            for task in tasks
        }

        # Wrap the as_completed iterator with tqdm
        for future in tqdm(
            as_completed(future_to_task),
            total=len(tasks),
            desc="Downloading Images",
            unit="img",
            colour="green",
        ):
            try:
                result = future.result()
                if result and result.startswith("✅"):
                    success_count += 1
                else:
                    # Use tqdm.write so error prints don't break the progress bar visual
                    tqdm.write(result)
            except Exception as exc:
                tqdm.write(f"❌ Thread generated an exception: {exc}")

    print(f"\n🚀 Done! Downloaded {success_count}/{len(tasks)} images concurrently.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Label Studio export programmatically, save the raw JSON with a timestamp, and download the images."
    )
    parser.add_argument(
        "--out_img_dir",
        default="data/raw/images",
        help="Directory to save the downloaded images",
    )
    parser.add_argument(
        "--out_export_dir",
        default="data/raw/exports",
        help="Directory to save the raw JSON export file (default: data/exports)",
    )

    args = parser.parse_args()

    download_export_and_images(export_dir=args.out_export_dir, img_dir=args.out_img_dir)
