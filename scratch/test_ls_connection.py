import os
import sys
import json
from pathlib import Path
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv

def main():
    print("=" * 60)
    print("🔍 LABEL STUDIO CONNECTION & DOWNLOAD DIAGNOSTIC TEST")
    print("=" * 60)

    # Load environment
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"📄 Loaded environment from {env_path.resolve()}")
    else:
        load_dotenv()
        print("⚠️ No .env file in current directory, loaded default environment")

    ls_url = (os.getenv("LABEL_STUDIO_URL") or "").rstrip("/")
    api_token = os.getenv("LABEL_STUDIO_API_TOKEN")
    username = os.getenv("LABEL_STUDIO_USERNAME")
    password = os.getenv("LABEL_STUDIO_PASSWORD")
    project_id = os.getenv("LABEL_STUDIO_PROJECT_ID", "1")

    print(f"Server URL:        {ls_url}")
    print(f"Target Project ID: {project_id}")
    print(f"Username:          {username}")
    print(f"Password set:      {'Yes' if password else 'No'}")
    print(f"API Token set:     {'Yes (' + api_token[:6] + '...' + api_token[-4:] + ')' if api_token else 'No'}")
    print("-" * 60)

    if not ls_url:
        print("❌ ERROR: LABEL_STUDIO_URL is empty!")
        return

    # 1. Basic server reachability
    print("\n[Step 1] Testing basic reachability & health...")
    try:
        r_health = requests.get(f"{ls_url}/health", timeout=10)
        print(f"  --> GET /health returned HTTP {r_health.status_code}")
    except Exception as e:
        print(f"  ⚠️ GET /health failed: {e}")
        try:
            r_root = requests.get(ls_url, timeout=10)
            print(f"  --> GET / returned HTTP {r_root.status_code}")
        except Exception as e2:
            print(f"  ❌ Cannot reach server at {ls_url}: {e2}")
            return

    # 2. Test API Token Authentication
    print("\n[Step 2] Testing Token-based Authentication...")
    token_valid = False
    token_session = requests.Session()
    if api_token:
        token_session.headers.update({"Authorization": f"Token {api_token}"})
        try:
            r_whoami = token_session.get(f"{ls_url}/api/current-user/whoami", timeout=10)
            print(f"  --> GET /api/current-user/whoami: HTTP {r_whoami.status_code}")
            if r_whoami.status_code == 200:
                user_info = r_whoami.json()
                print(f"      ✅ Token is VALID. Logged in as: {user_info.get('email')} (ID: {user_info.get('id')})")
                token_valid = True
            else:
                print(f"      ❌ Token rejected: {r_whoami.text[:200]}")
        except Exception as e:
            print(f"  ❌ Error testing token: {e}")

        # Test project access with token
        if token_valid:
            try:
                r_proj = token_session.get(f"{ls_url}/api/projects/{project_id}", timeout=10)
                print(f"  --> GET /api/projects/{project_id}: HTTP {r_proj.status_code}")
                if r_proj.status_code == 200:
                    proj = r_proj.json()
                    print(f"      ✅ Project accessible! Title: '{proj.get('title')}', Total tasks: {proj.get('task_number')}")
                else:
                    print(f"      ⚠️ Project {project_id} access failed: {r_proj.text[:200]}")
            except Exception as e:
                print(f"  ❌ Error testing project access with token: {e}")
    else:
        print("  ⚠️ No LABEL_STUDIO_API_TOKEN provided.")

    # 3. Test Username & Password (Session Login)
    print("\n[Step 3] Testing Username/Password Session Login...")
    session_valid = False
    cookie_session = requests.Session()
    if username and password:
        login_url = f"{ls_url}/user/login/"
        try:
            r_get = cookie_session.get(login_url, timeout=10)
            csrf_token = cookie_session.cookies.get("csrftoken", "")
            print(f"  --> GET {login_url}: HTTP {r_get.status_code} (CSRF token present: {bool(csrf_token)})")
            
            login_data = {
                "email": username,
                "password": password,
                "csrfmiddlewaretoken": csrf_token,
            }
            r_post = cookie_session.post(
                login_url,
                data=login_data,
                headers={"Referer": login_url},
                timeout=10,
                allow_redirects=False,
            )
            print(f"  --> POST {login_url}: HTTP {r_post.status_code}")
            location = r_post.headers.get("Location")
            print(f"      Redirect Location: {location}")
            print(f"      Cookies received: {dict(cookie_session.cookies)}")

            if r_post.status_code in [200, 302] and "sessionid" in cookie_session.cookies:
                print("      ✅ Session Login SUCCESSFUL! 'sessionid' cookie obtained.")
                session_valid = True
            elif r_post.status_code == 302 and location and "/user/login" not in location:
                print("      ✅ Session Login SUCCESSFUL (redirected to dashboard).")
                session_valid = True
            else:
                print(f"      ❌ Session Login FAILED. Credentials may be invalid or rejected.")
                if r_post.status_code == 200:
                    print("      Note: HTTP 200 without redirect on /user/login/ usually means invalid username or password form errors.")
                    # Let's inspect page for form errors
                    if "form-error" in r_post.text or "error" in r_post.text.lower():
                        print("      (Form error detected in HTML response)")

            if session_valid:
                r_proj2 = cookie_session.get(f"{ls_url}/api/projects/{project_id}", timeout=10)
                print(f"  --> GET /api/projects/{project_id} with session cookie: HTTP {r_proj2.status_code}")
                if r_proj2.status_code == 200:
                    print("      ✅ Project accessible via session cookie.")
                else:
                    print(f"      ⚠️ Failed to access project via session: {r_proj2.text[:200]}")
        except Exception as e:
            print(f"  ❌ Error testing session login: {e}")
    else:
        print("  ⚠️ Username or Password not provided.")

    # 4. Test Image Download
    print("\n[Step 4] Testing Tasks and Image Download...")
    active_session = token_session if token_valid else (cookie_session if session_valid else None)
    if not active_session:
        print("  ❌ Neither Token nor Session authentication succeeded. Cannot proceed to download test.")
        return

    try:
        tasks_url = f"{ls_url}/api/tasks?project={project_id}&page_size=5"
        r_tasks = active_session.get(tasks_url, timeout=15)
        print(f"  --> Fetching sample tasks (GET {tasks_url}): HTTP {r_tasks.status_code}")
        if r_tasks.status_code != 200:
            print(f"      ❌ Failed to fetch tasks: {r_tasks.text[:300]}")
            return

        tasks_data = r_tasks.json()
        tasks = tasks_data.get("tasks", []) if isinstance(tasks_data, dict) else tasks_data
        print(f"      Found {len(tasks)} sample tasks.")

        if not tasks:
            print("      ⚠️ No tasks found in project to test image download.")
            return

        sample_task = tasks[0]
        task_id = sample_task.get("id")
        img_val = sample_task.get("data", {}).get("img") or sample_task.get("data", {}).get("image") or sample_task.get("file_upload")
        print(f"  --> Sample Task ID: {task_id}")
        print(f"      Raw Image URI: {img_val}")

        if not img_val:
            print("      ❌ No image URL found in task data.")
            return

        # Build full image URL
        if img_val.startswith("http"):
            full_img_url = img_val
        else:
            full_img_url = f"{ls_url}/{img_val.lstrip('/')}"
        print(f"      Full Image URL: {full_img_url}")

        # Test download with active session
        r_img = active_session.get(full_img_url, timeout=15, stream=True)
        print(f"      GET Image HTTP Status: {r_img.status_code}")
        print(f"      Content-Type: {r_img.headers.get('Content-Type')}")
        print(f"      Content-Length: {r_img.headers.get('Content-Length')} bytes")

        if r_img.status_code == 200:
            content_sample = r_img.raw.read(1024)
            print(f"      ✅ Image Download SUCCESSFUL! Read {len(content_sample)} bytes sample.")
            # Check if it looks like an image or an HTML error page
            if content_sample.startswith(b"<!DOCTYPE") or content_sample.startswith(b"<html"):
                print("      ⚠️ WARNING: Response body is HTML, not binary image! This usually indicates authentication or redirect failure.")
            else:
                print("      ✅ Binary image header verified.")
        else:
            print(f"      ❌ Image Download FAILED with status {r_img.status_code}.")

        # Also test if image can be downloaded with Token vs with Cookie vs without auth
        print("\n  [Auth Comparison for Image Download]:")
        if token_valid:
            r_t = token_session.get(full_img_url, timeout=10)
            print(f"    - With Authorization: Token -> HTTP {r_t.status_code} ({r_t.headers.get('Content-Type')})")
        if session_valid:
            r_c = cookie_session.get(full_img_url, timeout=10)
            print(f"    - With Session Cookie     -> HTTP {r_c.status_code} ({r_c.headers.get('Content-Type')})")
        r_none = requests.get(full_img_url, timeout=10)
        print(f"    - Without Authentication  -> HTTP {r_none.status_code}")

    except Exception as e:
        print(f"  ❌ Error testing image download: {e}")

    print("\n" + "=" * 60)
    print("📋 SUMMARY & DIAGNOSIS")
    print("=" * 60)
    print(f"1. Token Auth:    {'WORKING ✅' if token_valid else 'FAILED ❌'}")
    print(f"2. Session Login: {'WORKING ✅' if session_valid else 'FAILED ❌'}")
    if token_valid and not session_valid:
        print("💡 CRITICAL INSIGHT: Token authentication is working, but username/password session login is failing.")
        print("   Scripts using only username/password (like `download_data.py`) will fail with 'Login failed'!")
    elif not token_valid and session_valid:
        print("💡 CRITICAL INSIGHT: Session login is working, but API token is invalid/expired.")
    elif token_valid and session_valid:
        print("💡 Both Token and Session login are working on local!")
    else:
        print("💡 Both Token and Session login failed. Check credentials, VPN, or network access.")

if __name__ == "__main__":
    main()
