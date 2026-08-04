import os
import sys
import time
import json
import tempfile
import argparse
from pathlib import Path
from dotenv import load_dotenv
import mlflow

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_mlflow_test(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> bool:
    """
    Tests connectivity to MLflow by creating an experiment, starting a run,
    logging fake metrics over multiple steps, and uploading a test artifact.

    Returns:
        bool: True if test succeeded, False otherwise.
    """
    load_dotenv()

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        print(f"🔗 Setting MLflow tracking URI: {uri}", flush=True)
    else:
        current_uri = mlflow.get_tracking_uri()
        print(f"ℹ️  No tracking URI provided; using default MLflow URI: {current_uri}", flush=True)

    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "cvm-swin-gcn-test")
    mlflow.set_experiment(exp_name)
    print(f"🧪 MLflow Experiment Set: '{exp_name}'", flush=True)

    test_run_name = run_name or f"test-run-{int(time.time())}"
    print(f"🚀 Starting test run: '{test_run_name}'...", flush=True)

    try:
        with mlflow.start_run(run_name=test_run_name) as run:
            run_id = run.info.run_id
            exp_id = run.info.experiment_id
            print(f"✅ Active MLflow Run created! Run ID: {run_id} | Exp ID: {exp_id}", flush=True)

            # 1. Log parameters
            params = {
                "test_mode": True,
                "framework": "PyTorch",
                "model_architecture": "CephalometricSwinGCN",
                "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "python_version": sys.version.split()[0],
            }
            mlflow.log_params(params)
            print(f"📊 Logged {len(params)} test parameters.", flush=True)

            # 2. Log fake metrics over 5 steps
            print("📈 Logging fake training & validation metrics over 5 steps...", flush=True)
            fake_metrics_history = [
                {"train_loss": 0.850, "val_loss": 0.910, "val_mae": 14.20, "val_sdr_2_5": 58.0},
                {"train_loss": 0.620, "val_loss": 0.680, "val_mae": 10.50, "val_sdr_2_5": 68.5},
                {"train_loss": 0.410, "val_loss": 0.450, "val_mae": 7.10, "val_sdr_2_5": 79.0},
                {"train_loss": 0.260, "val_loss": 0.310, "val_mae": 4.80, "val_sdr_2_5": 87.5},
                {"train_loss": 0.150, "val_loss": 0.190, "val_mae": 2.45, "val_sdr_2_5": 94.2},
            ]

            for step, metrics in enumerate(fake_metrics_history, start=1):
                mlflow.log_metrics(metrics, step=step)
                time.sleep(0.1)

            # 3. Create and upload test artifact
            print("📁 Uploading test artifact to MLflow...", flush=True)
            artifact_info = {
                "status": "SUCCESS",
                "message": "MLflow connection test completed successfully.",
                "test_run_name": test_run_name,
                "run_id": run_id,
                "experiment_id": exp_id,
                "tracking_uri": mlflow.get_tracking_uri(),
                "artifact_uri": run.info.artifact_uri,
                "logged_metrics_count": len(fake_metrics_history),
                "timestamp": time.time(),
            }

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / "mlflow_test_summary.json"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_info, f, indent=4)

                mlflow.log_artifact(str(tmp_path), artifact_path="test_results")

            print("🎉 Artifact successfully uploaded to 'test_results/mlflow_test_summary.json'!", flush=True)
            print(f"✨ MLflow connection test PASSED for run {run_id}.", flush=True)
            return True

    except Exception as e:
        print(f"❌ MLflow connection test FAILED: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Test MLflow server connectivity, metrics logging, and artifact upload.")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow test run name")

    args = parser.parse_args()

    success = run_mlflow_test(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
