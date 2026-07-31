#!/usr/bin/env python3
"""
Standalone MinIO Connection and File Upload Test Script.

Usage:
    python scripts/test_minio.py
    python scripts/test_minio.py --file my_file.pth --object-name best_latest.pth
"""

import sys
import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root directory to sys.path to allow imports from src
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from src.utils.minio_utils import (
    get_minio_credentials,
    get_minio_client,
    ensure_bucket_exists,
    upload_artifact_to_minio,
)


def run_minio_test(
    file_path: str = None,
    object_name: str = None,
    bucket_name: str = None,
    clean_sample: bool = True,
) -> bool:
    print("=" * 60)
    print("           MinIO Connection & Upload Test")
    print("=" * 60)

    load_dotenv()
    creds = get_minio_credentials()

    print("[1] Environment Configuration:")
    print(f"    - Endpoint:   {creds['endpoint'] or '❌ NOT SET'}")
    if creds['access_key']:
        masked_key = creds['access_key'][:4] + "*" * (len(creds['access_key']) - 4)
        print(f"    - Access Key: {masked_key}")
    else:
        print("    - Access Key: ❌ NOT SET")
    print(f"    - Secret Key: {'Set (Hidden)' if creds['secret_key'] else '❌ NOT SET'}")
    print(f"    - Bucket:     {bucket_name or creds['bucket']}")

    if not creds['endpoint'] or not creds['access_key'] or not creds['secret_key']:
        print("\n❌ Error: Missing MinIO environment variables in .env file.")
        return False

    print("\n[2] Connecting to MinIO server...")
    try:
        s3_client = get_minio_client()
        buckets = s3_client.list_buckets()
        bucket_names = [b["Name"] for b in buckets.get("Buckets", [])]
        print(f"    ✅ Connected successfully! Existing buckets: {bucket_names}")
    except Exception as e:
        print(f"    ❌ Failed to connect to MinIO server: {e}")
        return False

    target_bucket = bucket_name or creds["bucket"]
    print(f"\n[3] Verifying bucket '{target_bucket}'...")
    try:
        ensure_bucket_exists(s3_client, target_bucket)
    except Exception as e:
        print(f"    ❌ Failed to verify/create bucket '{target_bucket}': {e}")
        return False

    temp_created = False
    if file_path is None:
        file_path = "minio_test_sample.txt"
        timestamp = datetime.now().isoformat()
        sample_content = (
            f"MinIO Upload Test File\n"
            f"Generated at: {timestamp}\n"
            f"Status: Testing upload connection before data download or training.\n"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
        temp_created = True
        print(f"\n[4] Created sample file: '{file_path}'")
    else:
        print(f"\n[4] Using target file: '{file_path}'")

    if object_name is None:
        object_name = f"test_uploads/{os.path.basename(file_path)}"

    print(f"\n[5] Uploading '{file_path}' to MinIO as '{object_name}'...")
    success = upload_artifact_to_minio(
        file_path=file_path,
        object_name=object_name,
        bucket_name=target_bucket,
        s3_client=s3_client,
    )

    if not success:
        print("\n❌ MinIO upload test FAILED.")
        if temp_created and clean_sample and os.path.exists(file_path):
            os.remove(file_path)
        return False

    print(f"\n[6] Verifying object '{object_name}' in bucket '{target_bucket}'...")
    try:
        response = s3_client.head_object(Bucket=target_bucket, Key=object_name)
        file_size = response.get("ContentLength", 0)
        print(f"    ✅ Verified! Object size in MinIO: {file_size} bytes.")
    except Exception as e:
        print(f"    ❌ Verification failed for object '{object_name}': {e}")
        if temp_created and clean_sample and os.path.exists(file_path):
            os.remove(file_path)
        return False

    if temp_created and clean_sample and os.path.exists(file_path):
        os.remove(file_path)

    print("\n" + "=" * 60)
    print("✅ MinIO upload test PASSED successfully!")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Standalone MinIO connection and sample upload test script."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to file to upload (default: generates a temporary sample text file).",
    )
    parser.add_argument(
        "--object-name",
        "-o",
        type=str,
        default=None,
        help="Target object key/name in MinIO (default: test_uploads/<filename>).",
    )
    parser.add_argument(
        "--bucket",
        "-b",
        type=str,
        default=None,
        help="Override MinIO bucket name.",
    )
    parser.add_argument(
        "--keep-sample",
        action="store_true",
        help="Do not delete generated sample test file after upload.",
    )

    args = parser.parse_args()

    success = run_minio_test(
        file_path=args.file,
        object_name=args.object_name,
        bucket_name=args.bucket,
        clean_sample=not args.keep_sample,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
