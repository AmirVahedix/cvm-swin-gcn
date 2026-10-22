#!/usr/bin/env python3
"""
Download Checkpoint from MLflow / MinIO Storage.
Streams artifacts directly from the remote MLflow tracking server (backed by MinIO)
without needing local laptop downloads or manual SCP transfer.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def download_checkpoint_from_mlflow(
    run_id: str,
    artifact_path: str = "checkpoints/best.pth",
    dest_path: str = "artifacts/stage1_best.pth",
    tracking_uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """Downloads checkpoint artifact from MLflow into local destination file."""
    load_dotenv()
    import mlflow
    from mlflow.tracking import MlflowClient

    user = username or os.getenv("MLFLOW_TRACKING_USERNAME")
    pwd = password or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if user:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
    if pwd:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.amirlogedixyz.ir")
    if uri:
        mlflow.set_tracking_uri(uri)

    print("=" * 80)
    print(f"📥 FETCHING CHECKPOINT FROM MLFLOW / MINIO")
    print(f"   MLflow Tracking URI: {uri}")
    print(f"   Run ID:              {run_id}")
    print(f"   Artifact Target:     {artifact_path}")
    print(f"   Local Destination:   {dest_path}")
    print("=" * 80)

    client = MlflowClient(tracking_uri=uri)

    # Validate run exists and check artifact path
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Search for artifact if exact path not found
    actual_artifact = artifact_path
    try:
        root_artifacts = [a.path for a in client.list_artifacts(run_id)]
        if "checkpoints" in root_artifacts:
            ckpt_artifacts = [a.path for a in client.list_artifacts(run_id, path="checkpoints")]
            if artifact_path not in ckpt_artifacts and "checkpoints/best.pth" in ckpt_artifacts:
                actual_artifact = "checkpoints/best.pth"
        elif "best.pth" in root_artifacts:
            actual_artifact = "best.pth"
    except Exception as e:
        print(f"⚠️ Warning: Could not list artifacts ahead of download: {e}")

    print(f"--> Downloading '{actual_artifact}' via MLflow HTTP artifact streaming...")
    downloaded_path = client.download_artifacts(
        run_id=run_id,
        path=actual_artifact,
        dst_path=str(dest.parent),
    )

    if Path(downloaded_path).resolve() != dest.resolve():
        shutil.copy2(downloaded_path, dest)

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"✅ Download complete! File saved to: {dest} ({size_mb:.2f} MB)")
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
