import os
import sys
import time
from pathlib import Path
from ftplib import FTP, FTP_TLS, error_perm
from typing import List, Union, Optional
from dotenv import load_dotenv

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _ensure_remote_dir(ftp: Union[FTP, FTP_TLS], remote_dir: str):
    """
    Recursively navigates to or creates the specified remote directory path.
    """
    if not remote_dir or remote_dir in ("/", "."):
        return

    # Normalize path separators
    normalized_path = remote_dir.replace("\\", "/").strip("/")
    parts = normalized_path.split("/")

    # If original path started with '/', navigate to root first
    if remote_dir.startswith("/"):
        try:
            ftp.cwd("/")
        except Exception:
            pass

    for part in parts:
        if not part:
            continue
        try:
            ftp.cwd(part)
        except error_perm:
            try:
                ftp.mkd(part)
                ftp.cwd(part)
            except error_perm as e:
                # If creating failed, check if directory now exists or re-raise
                try:
                    ftp.cwd(part)
                except Exception:
                    raise RuntimeError(f"Could not change into or create remote FTP directory '{part}': {e}")


def _get_ftp_connection(
    ftp_host: Optional[str] = None,
    ftp_port: Optional[Union[int, str]] = None,
    ftp_user: Optional[str] = None,
    ftp_password: Optional[str] = None,
    use_tls: Optional[Union[bool, str]] = None,
    timeout: int = 30,
) -> tuple[Union[FTP, FTP_TLS], str]:
    """
    Establishes and returns an authenticated FTP or FTP_TLS connection.
    """
    load_dotenv()

    host = ftp_host or os.getenv("FTP_HOST")
    if not host:
        raise ValueError("FTP_HOST is not configured in .env or passed as argument.")

    port_raw = ftp_port or os.getenv("FTP_PORT", "21")
    try:
        port = int(port_raw)
    except (ValueError, TypeError):
        port = 21

    user = ftp_user or os.getenv("FTP_USER") or os.getenv("FTP_USERNAME") or "anonymous"
    pwd = ftp_password or os.getenv("FTP_PASSWORD") or os.getenv("FTP_PASS") or ""

    tls_env = os.getenv("FTP_TLS", "false").lower() in ("true", "1", "yes") if use_tls is None else bool(use_tls)

    if tls_env:
        print(f"🔒 Connecting via FTPS (TLS) to {host}:{port}...")
        ftp = FTP_TLS(timeout=timeout)
        ftp.connect(host, port)
        ftp.auth()
        ftp.login(user=user, passwd=pwd)
        ftp.prot_p()  # Secure data connection
    else:
        print(f"🌐 Connecting via standard FTP to {host}:{port}...")
        ftp = FTP(timeout=timeout)
        ftp.connect(host, port)
        ftp.login(user=user, passwd=pwd)

    ftp.set_pasv(True)
    return ftp, host


def upload_files_to_ftp(
    files: List[Union[str, Path]],
    ftp_host: Optional[str] = None,
    ftp_port: Optional[Union[int, str]] = None,
    ftp_user: Optional[str] = None,
    ftp_password: Optional[str] = None,
    remote_dir: Optional[str] = None,
    use_tls: Optional[Union[bool, str]] = None,
    timeout: int = 60,
) -> bool:
    """
    Uploads a list of local files to the specified FTP remote directory.

    Returns:
        bool: True if all files uploaded successfully, False otherwise.
    """
    load_dotenv()

    host = ftp_host or os.getenv("FTP_HOST")
    if not host:
        print("ℹ️ FTP_HOST not specified or configured in .env; skipping FTP upload.")
        return False

    valid_files = [Path(f) for f in files if f and Path(f).exists()]
    if not valid_files:
        print("⚠️ No existing local files provided for FTP upload.")
        return False

    target_dir = remote_dir or os.getenv("FTP_REMOTE_DIR", "/")

    print(f"\n📤 Starting FTP upload of {len(valid_files)} file(s) to '{host}' (Remote Dir: '{target_dir}')...")

    try:
        ftp, _ = _get_ftp_connection(
            ftp_host=host,
            ftp_port=ftp_port,
            ftp_user=ftp_user,
            ftp_password=ftp_password,
            use_tls=use_tls,
            timeout=timeout,
        )

        try:
            if target_dir and target_dir not in ("/", "."):
                _ensure_remote_dir(ftp, target_dir)

            for file_path in valid_files:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                file_name = file_path.name
                print(f"--> Uploading '{file_name}' ({file_size_mb:.2f} MB)...", end="", flush=True)

                start_time = time.time()
                with open(file_path, "rb") as f:
                    ftp.storbinary(f"STOR {file_name}", f, blocksize=8192 * 4)

                duration = time.time() - start_time
                transfer_rate = (file_size_mb / duration) if duration > 0 else 0.0
                print(f" ✅ Done in {duration:.2f}s ({transfer_rate:.2f} MB/s)")

            print(f"🎉 Successfully uploaded {len(valid_files)} file(s) to FTP host '{host}'.")
            return True

        finally:
            try:
                ftp.quit()
            except Exception:
                ftp.close()

    except Exception as e:
        print(f"❌ FTP upload failed: {e}", file=sys.stderr)
        return False


def upload_file_to_ftp(
    file_path: Union[str, Path],
    ftp_host: Optional[str] = None,
    ftp_port: Optional[Union[int, str]] = None,
    ftp_user: Optional[str] = None,
    ftp_password: Optional[str] = None,
    remote_dir: Optional[str] = None,
    use_tls: Optional[Union[bool, str]] = None,
    timeout: int = 60,
) -> bool:
    """
    Convenience wrapper to upload a single file to FTP.
    """
    return upload_files_to_ftp(
        files=[file_path],
        ftp_host=ftp_host,
        ftp_port=ftp_port,
        ftp_user=ftp_user,
        ftp_password=ftp_password,
        remote_dir=remote_dir,
        use_tls=use_tls,
        timeout=timeout,
    )


def test_ftp_connection(
    ftp_host: Optional[str] = None,
    ftp_port: Optional[Union[int, str]] = None,
    ftp_user: Optional[str] = None,
    ftp_password: Optional[str] = None,
    remote_dir: Optional[str] = None,
    use_tls: Optional[Union[bool, str]] = None,
) -> bool:
    """
    Tests connectivity and authentication to the FTP host.
    """
    load_dotenv()
    host = ftp_host or os.getenv("FTP_HOST")
    if not host:
        print("❌ Error: FTP_HOST is not configured.")
        return False

    print(f"\n[FTP Pre-Flight] Testing connection to {host}...")
    try:
        ftp, _ = _get_ftp_connection(
            ftp_host=ftp_host,
            ftp_port=ftp_port,
            ftp_user=ftp_user,
            ftp_password=ftp_password,
            use_tls=use_tls,
            timeout=15,
        )
        target_dir = remote_dir or os.getenv("FTP_REMOTE_DIR", "/")
        if target_dir and target_dir not in ("/", "."):
            _ensure_remote_dir(ftp, target_dir)

        welcome_msg = ftp.getwelcome()
        print(f"✅ Connected successfully! Server greeting: {welcome_msg}")
        print(f"📂 Current remote directory: {ftp.pwd()}")
        ftp.quit()
        return True
    except Exception as e:
        print(f"❌ FTP connection test failed: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FTP connectivity or upload files to FTP.")
    parser.add_argument("--test", action="store_true", help="Test FTP connection and authentication")
    parser.add_argument("--files", nargs="+", default=[], help="File paths to upload")
    parser.add_argument("--ftp-host", type=str, default=None, help="FTP host")
    parser.add_argument("--ftp-port", type=str, default=None, help="FTP port (default: 21)")
    parser.add_argument("--ftp-user", type=str, default=None, help="FTP username")
    parser.add_argument("--ftp-password", type=str, default=None, help="FTP password")
    parser.add_argument("--ftp-remote-dir", type=str, default=None, help="Remote directory path on FTP server")
    parser.add_argument("--ftp-tls", action="store_true", help="Use FTPS / TLS")

    args = parser.parse_args()

    if args.test or not args.files:
        success = test_ftp_connection(
            ftp_host=args.ftp_host,
            ftp_port=args.ftp_port,
            ftp_user=args.ftp_user,
            ftp_password=args.ftp_password,
            remote_dir=args.ftp_remote_dir,
            use_tls=args.ftp_tls if args.ftp_tls else None,
        )
        if not success and args.test:
            sys.exit(1)

    if args.files:
        success = upload_files_to_ftp(
            files=args.files,
            ftp_host=args.ftp_host,
            ftp_port=args.ftp_port,
            ftp_user=args.ftp_user,
            ftp_password=args.ftp_password,
            remote_dir=args.ftp_remote_dir,
            use_tls=args.ftp_tls if args.ftp_tls else None,
        )
        if not success:
            sys.exit(1)
