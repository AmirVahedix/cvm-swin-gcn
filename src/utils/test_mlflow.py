import os
import sys
import time
import json
import tempfile
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Try importing MLflow and PIL for image generation test
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    print("❌ Error: 'mlflow' package is not installed. Run 'pip install mlflow' first.", file=sys.stderr)
    sys.exit(1)


def create_dummy_png(filepath: Path):
    """Creates a simple 100x100 PNG test image."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (100, 100), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10, 40), "MLflow Test", fill=(255, 255, 0))
        img.save(filepath, format="PNG")
    except ImportError:
        # Fallback to minimal binary PNG header if PIL is not installed
        with open(filepath, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7T8\x82\x00\x00\x00\x00IEND\xaeB`\x82")


def run_mlflow_test(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    tracking_username: str | None = None,
    tracking_password: str | None = None,
) -> bool:
    """
    Comprehensive verification test for MLflow:
    1. Server connectivity
    2. Authentication & experiment retrieval
    3. Run creation & params/metrics logging
    4. Text & JSON artifact upload
    5. Image/binary artifact upload
    6. Listing artifacts (S3 / MinIO backend test)
    7. Downloading artifacts (verification of round-trip read/write)
    """
    load_dotenv()

    # Configure authentication environment variables if provided
    user = tracking_username or os.getenv("MLFLOW_TRACKING_USERNAME")
    pwd = tracking_password or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if user:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
    if pwd:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.amirlogedixyz.ir")
    mlflow.set_tracking_uri(uri)
    print(f"\n=======================================================", flush=True)
    print(f"🔍 [1/6] Connecting to MLflow Tracking Server...", flush=True)
    print(f"    URI: {uri}", flush=True)
    if user:
        print(f"    User: {user}", flush=True)

    client = MlflowClient()

    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "cvm-swin-gcn")
    try:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            exp_id = client.create_experiment(exp_name)
            print(f"    Created new experiment: '{exp_name}' (ID: {exp_id})", flush=True)
        else:
            exp_id = exp.experiment_id
            print(f"    Found existing experiment: '{exp_name}' (ID: {exp_id})", flush=True)
        mlflow.set_experiment(exp_name)
    except Exception as e:
        print(f"\n❌ [FAILED] Could not connect to MLflow server or manage experiments: {e}", file=sys.stderr)
        return False

    test_run_name = run_name or f"conn-test-{int(time.time())}"
    print(f"\n🚀 [2/6] Starting test run '{test_run_name}'...", flush=True)

    try:
        with mlflow.start_run(run_name=test_run_name) as run:
            run_id = run.info.run_id
            print(f"    ✅ Run created successfully! Run ID: {run_id}", flush=True)
            print(f"    Artifact URI: {run.info.artifact_uri}", flush=True)

            # --- Log Parameters & Metrics ---
            print(f"\n📊 [3/6] Logging parameters and sample metrics...", flush=True)
            mlflow.log_params({
                "test_mode": True,
                "framework": "PyTorch",
                "python_version": sys.version.split()[0],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            for step in range(1, 4):
                mlflow.log_metrics({"test_loss": 1.0 / step, "test_mae": 2.5 / step}, step=step)
            print("    ✅ Parameters and metrics logged successfully.", flush=True)

            # --- Upload Artifacts (Text, JSON, Image) ---
            print(f"\n📁 [4/6] Uploading test artifacts to MLflow / MinIO backend...", flush=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)

                # 1. JSON test artifact
                json_path = tmp_dir_path / "test_report.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({"status": "PASS", "run_id": run_id, "time": time.time()}, f, indent=2)
                
                print("    --> Uploading JSON metadata...", flush=True)
                mlflow.log_artifact(str(json_path), artifact_path="diagnostics")

                # 2. PNG test chart
                img_path = tmp_dir_path / "test_chart.png"
                create_dummy_png(img_path)
                print("    --> Uploading sample image/chart PNG...", flush=True)
                mlflow.log_artifact(str(img_path), artifact_path="training_metric_charts")

            print("    ✅ All test artifacts uploaded to MLflow!", flush=True)

            # --- Test Listing Artifacts from S3 / MinIO ---
            print(f"\n🔍 [5/6] Testing Artifact Listing (Verifying S3/MinIO read access)...", flush=True)
            artifacts = client.list_artifacts(run_id)
            artifact_paths = [a.path for a in artifacts]
            print(f"    Root artifact directories found: {artifact_paths}", flush=True)

            chart_artifacts = client.list_artifacts(run_id, path="training_metric_charts")
            print(f"    Files in 'training_metric_charts': {[a.path for a in chart_artifacts]}", flush=True)

            # --- Test Downloading Artifacts ---
            print(f"\n📥 [6/6] Testing Artifact Download (Round-trip verification)...", flush=True)
            with tempfile.TemporaryDirectory() as dl_dir:
                downloaded_file = client.download_artifacts(run_id, "diagnostics/test_report.json", dst_path=dl_dir)
                if os.path.exists(downloaded_file):
                    with open(downloaded_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    print(f"    ✅ Successfully downloaded artifact and read content: {data}", flush=True)
                else:
                    raise RuntimeError("Downloaded file does not exist locally.")

            print(f"\n=======================================================", flush=True)
            print(f"🎉 ALL MLFLOW CHECKS PASSED SUCCESSFULLY!", flush=True)
            print(f"🌐 Check run in your browser: {uri}/#/experiments/{exp_id}/runs/{run_id}/artifacts", flush=True)
            print(f"=======================================================\n", flush=True)
            return True

    except Exception as e:
        print(f"\n❌ [TEST FAILED] Error during MLflow artifact verification:\n{e}", file=sys.stderr)
        print("\n💡 Troubleshooting Tips:", file=sys.stderr)
        print("1. If you see 'The Access Key Id you provided does not exist in our records':", file=sys.stderr)
        print("   - Check that MINIO_ROOT_PASSWORD and PG_PASSWORD in .env are surrounded by quotes.", file=sys.stderr)
        print("   - Restart the MLflow docker stack: docker compose down && docker compose up -d", file=sys.stderr)
        print("2. If you see 'Connection refused' or SSL error:", file=sys.stderr)
        print("   - Ensure MLflow is running and accessible from this machine.", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Test MLflow server connectivity, metrics logging, and artifact upload.")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI (default: https://mlflow.amirlogedixyz.ir or from .env)")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name (default: cvm-swin-gcn)")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow test run name")
    parser.add_argument("--username", type=str, default=None, help="MLflow basic auth username")
    parser.add_argument("--password", type=str, default=None, help="MLflow basic auth password")

    args = parser.parse_args()

    success = run_mlflow_test(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_username=args.username,
        tracking_password=args.password,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
