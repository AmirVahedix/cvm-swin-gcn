#!/usr/bin/env python3
"""
Download Checkpoint from MLflow / MinIO Storage.
Streams artifacts directly from the remote MLflow tracking server (backed by MinIO)
with real-time MB-by-MB progress, transfer speed, and ETA.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def download_checkpoint_from_mlflow(
    run_id: str = "2d4d452398b14c688f51ac4431d40698",
    artifact_path: str = "checkpoints/best.pth",
    dest_path: str = "artifacts/stage1_best.pth",
    tracking_uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """Downloads checkpoint artifact from MLflow into local destination file with live progress."""
    load_dotenv()
    import requests
    from tqdm import tqdm
    from mlflow.tracking import MlflowClient

    user = username or os.getenv("MLFLOW_TRACKING_USERNAME")
    pwd = password or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if user:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
    if pwd:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.amirlogedixyz.ir")
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"📥 FETCHING CHECKPOINT FROM MLFLOW / MINIO")
    print(f"   MLflow Tracking URI: {uri}")
    print(f"   Run ID:              {run_id}")
    print(f"   Artifact Target:     {artifact_path}")
    print(f"   Local Destination:   {dest_path}")
    print("=" * 80)

    client = MlflowClient(tracking_uri=uri)

    # 1. Resolve experiment ID and artifact metadata
    try:
        run = client.get_run(run_id)
        exp_id = run.info.experiment_id
    except Exception as e:
        print(f"⚠️ Could not query run metadata: {e}")
        exp_id = "1"

    # Search for actual artifact name and size
    actual_artifact = artifact_path
    file_size = None
    try:
        if "/" in artifact_path:
            parent_dir = artifact_path.split("/")[0]
            artifacts = client.list_artifacts(run_id, path=parent_dir)
        else:
            artifacts = client.list_artifacts(run_id)

        for a in artifacts:
            if a.path == artifact_path or a.path.endswith("best.pth"):
                actual_artifact = a.path
                file_size = a.file_size
                break
    except Exception as e:
        print(f"⚠️ Could not list artifacts ahead of download: {e}")

    # 2. Stream download via direct HTTP endpoint with live byte-level progress bar
    direct_url = f"{uri.rstrip('/')}/api/2.0/mlflow-artifacts/artifacts/{exp_id}/{run_id}/artifacts/{actual_artifact}"
    auth = (user, pwd) if (user and pwd) else None
    temp_file = dest.with_suffix(".tmp")

    print(f"--> Streaming '{actual_artifact}' from {direct_url}...")
    try:
        with requests.get(direct_url, auth=auth, stream=True, timeout=30) as response:
            response.raise_for_status()
            total_bytes = file_size or int(response.headers.get("content-length", 0)) or None

            with open(temp_file, "wb") as f, tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {dest.name}",
                ncols=90,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunk
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        if temp_file.exists():
            shutil.move(str(temp_file), str(dest))
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"✅ Download complete! Successfully saved to: {dest} ({size_mb:.2f} MB)")
            return str(dest)

    except Exception as stream_err:
        print(f"⚠️ Direct HTTP stream failed ({stream_err}), falling back to client.download_artifacts...")
        if temp_file.exists():
            temp_file.unlink()

        downloaded_path = client.download_artifacts(
            run_id=run_id,
            path=actual_artifact,
            dst_path=str(dest.parent),
        )
        if Path(downloaded_path).resolve() != dest.resolve():
            shutil.copy2(downloaded_path, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"✅ Download complete! Saved to: {dest} ({size_mb:.2f} MB)")
        return str(dest)


def main():
    parser = argparse.ArgumentParser(description="Download checkpoint weights from MLflow / MinIO")
    parser.add_argument(
        "--run-id",
        type=str,
        default="2d4d452398b14c688f51ac4431d40698",
        help="MLflow Run ID containing the checkpoint artifact",
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default="checkpoints/best.pth",
        help="Artifact path inside the run (default: checkpoints/best.pth)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="artifacts/stage1_best.pth",
        help="Destination path on local filesystem (default: artifacts/stage1_best.pth)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI",
    )
    args = parser.parse_args()

    try:
        download_checkpoint_from_mlflow(
            run_id=args.run_id,
            artifact_path=args.artifact,
            dest_path=args.dest,
            tracking_uri=args.tracking_uri,
        )
    except Exception as e:
        print(f"❌ Failed to download checkpoint: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
