import json
import os
import shutil
import requests
import argparse
import sys
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

load_dotenv()


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
    img_url = task["data"]["img"]
    base_name = os.path.splitext(os.path.basename(img_url))[0]
    clean_name = (
        img_url.split("-", 1)[1] if "-" in base_name else os.path.basename(img_url)
    )
    img_ext = os.path.splitext(clean_name)[1]
    clean_base_name = os.path.splitext(clean_name)[0]

    img_path = os.path.join(img_dir, f"{clean_base_name}{img_ext}")

    # Download image
    full_img_url = img_url if img_url.startswith("http") else f"{ls_url}{img_url}"
    try:
        response = session.get(full_img_url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(img_path, "wb") as f_img:
                for chunk in response.iter_content(1024):
                    f_img.write(chunk)
        else:
            return f"⚠️ Skipping {clean_name}: Image download failed (Status {response.status_code})"
    except Exception as e:
        return f"❌ Network Error on {clean_name}: {e}"

    return f"✅ Downloaded {clean_base_name}{img_ext}"


def download_export_and_images(
    project_id, export_dir, img_dir, ls_url, username, password
):
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

    login_url = f"{ls_url}/user/login/"
    session.get(login_url)
    csrf_token = session.cookies.get("csrftoken", "")

    login_data = {
        "email": username,
        "password": password,
        "csrfmiddlewaretoken": csrf_token,
    }

    login_response = session.post(
        login_url, data=login_data, headers={"Referer": login_url}
    )
    if login_response.status_code not in [200, 302]:
        print("❌ Login failed. Please check your credentials and Label Studio URL.")
        return

    print("🔑 Authenticated successfully.")

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

    ls_url = os.environ.get("LABEL_STUDIO_URL")
    project_id = os.environ.get("LABEL_STUDIO_PROJECT_ID")
    ls_user = os.environ.get("LABEL_STUDIO_USERNAME")
    ls_pass = os.environ.get("LABEL_STUDIO_PASSWORD")

    missing_vars = [
        var
        for var, val in zip(
            [
                "LABEL_STUDIO_URL",
                "LABEL_STUDIO_PROJECT_ID",
                "LABEL_STUDIO_USERNAME",
                "LABEL_STUDIO_PASSWORD",
            ],
            [ls_url, project_id, ls_user, ls_pass],
        )
        if not val
    ]

    if missing_vars:
        print(
            f"❌ Error: Missing required environment variables in .env: {', '.join(missing_vars)}"
        )
        sys.exit(1)

    download_export_and_images(
        project_id, args.out_export_dir, args.out_img_dir, ls_url, ls_user, ls_pass
    )
