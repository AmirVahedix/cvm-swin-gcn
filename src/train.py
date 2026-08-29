import os
import sys
import time
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dotenv import load_dotenv
import mlflow

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import get_dataloaders, NUM_LANDMARKS
from src.models.model import CephalometricSwinGCN
from src.models.losses import AdaptiveWingLoss, WingLoss, AnatomicalGraphLoss
from src.eval import run_evaluation
from src.utils.ftp_utils import upload_files_to_ftp

TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_NPZ_DIR = "dataset/train/labels"

VAL_IMG_DIR = "dataset/val/images"
VAL_NPZ_DIR = "dataset/val/labels"

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
LLRD_DECAY_RATE = 0.8
LAMBDA_HM = 1.0
LAMBDA_CD = 10.0
LAMBDA_GRAPH = 2.0
EARLY_STOPPING_PATIENCE = 30
SAVE_PATH = "./artifacts/best.pth"


def get_llrd_param_groups(
    model: torch.nn.Module,
    base_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    decay_rate: float = 0.8,
) -> list[dict]:
    """
    Constructs parameter groups for Layer-wise Learning Rate Decay (LLRD).
    - Task heads, U-Net decoder, GCN layers, and Soft-Argmax receive base_lr (1.0x).
    - Swin backbone stages receive progressively decayed learning rates (lr * decay_rate^(num_stages - stage_idx)).
    - 1D tensors (biases, norm parameters, log_temperature) have weight_decay=0.0.
    """
    param_groups = []
    num_stages = 4  # Swin backbone has 4 feature stages (0, 1, 2, 3)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Exclude 1D tensors (biases, norm scales/shifts, log_temperature) from weight decay
        curr_weight_decay = 0.0 if (param.ndim <= 1 or "bias" in name or "norm" in name or "log_temperature" in name) else weight_decay

        # Compute layer-wise learning rate decay factor
        if "backbone" in name:
            if "patch_embed" in name or "absolute_pos_embed" in name:
                scale = decay_rate ** (num_stages + 1)
            else:
                scale = decay_rate ** num_stages
                for i in range(num_stages):
                    if f"layers.{i}" in name or f"stages.{i}" in name:
                        scale = decay_rate ** (num_stages - i)
                        break
        else:
            # U-Net decoder, GCN heads, Soft-Argmax temperature -> Base LR (1.0x)
            scale = 1.0

        group_lr = base_lr * scale
        param_groups.append(
            {
                "params": [param],
                "lr": group_lr,
                "weight_decay": curr_weight_decay,
            }
        )

    return param_groups


def get_landmark_weights(device: torch.device) -> torch.Tensor:

    """
    Constructs landmark loss weights prioritizing difficult lower-vertebral posterior landmarks.
    """
    weights = torch.ones(NUM_LANDMARKS, dtype=torch.float32, device=device)
    weights[5] = 2.0   # C3_PI
    weights[10] = 2.5  # C4_PI (worst performing landmark)
    weights[12] = 2.0  # C4_AI
    return weights


def train_epoch(
    model,
    dataloader,
    optimizer,
    awl_loss,
    wing_loss,
    graph_loss,
    landmark_weights,
    lambda_hm,
    lambda_cd,
    lambda_graph,
    device,
    epoch: int = 1,
    epochs: int = 1,
):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch}/{epochs}] Train",
        leave=False,
    )
    for step, batch in enumerate(pbar, 1):
        images = batch["image"].to(device)
        gt_heatmaps = batch["heatmaps"].to(device)
        gt_coords = batch["coords"].to(device)

        optimizer.zero_grad()

        pred_heatmaps, pred_coords = model(images)

        loss_hm = awl_loss(pred_heatmaps, gt_heatmaps)
        loss_cd = wing_loss(pred_coords, gt_coords, landmark_weights=landmark_weights)
        loss_g = graph_loss(pred_coords, gt_coords)

        total_loss = (lambda_hm * loss_hm) + (lambda_cd * loss_cd) + (lambda_graph * loss_g)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += total_loss.item()
        pbar.set_postfix(loss=f"{epoch_loss / step:.4f}")

    return epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def validate_epoch(
    model,
    dataloader,
    awl_loss,
    wing_loss,
    graph_loss,
    landmark_weights,
    lambda_hm,
    lambda_cd,
    lambda_graph,
    device,
    img_size=640,
    epoch: int = 1,
    epochs: int = 1,
):
    model.eval()
    epoch_loss = 0.0
    all_radial_errors = []

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch}/{epochs}] Val",
        leave=False,
    )
    with torch.no_grad():
        for step, batch in enumerate(pbar, 1):
            images = batch["image"].to(device)
            gt_heatmaps = batch["heatmaps"].to(device)
            gt_coords = batch["coords"].to(device)

            pred_heatmaps, pred_coords = model(images)

            loss_hm = awl_loss(pred_heatmaps, gt_heatmaps)
            loss_cd = wing_loss(pred_coords, gt_coords, landmark_weights=landmark_weights)
            loss_g = graph_loss(pred_coords, gt_coords)

            total_loss = (lambda_hm * loss_hm) + (lambda_cd * loss_cd) + (lambda_graph * loss_g)
            epoch_loss += total_loss.item()

            # Radial errors (in pixels) across landmarks
            pred_px = pred_coords * img_size
            gt_px = gt_coords * img_size
            radial_errors = torch.sqrt(torch.sum((pred_px - gt_px) ** 2, dim=-1))  # [B, N]
            all_radial_errors.append(radial_errors.cpu())

            pbar.set_postfix(loss=f"{epoch_loss / step:.4f}")

    val_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    if len(all_radial_errors) > 0:
        all_errors = torch.cat(all_radial_errors, dim=0)  # [Total_Samples, N]
        val_mae = all_errors.mean().item()
        val_rmse = torch.sqrt((all_errors ** 2).mean()).item()
        sdr_2_0 = (all_errors <= 2.0).float().mean().item() * 100.0
        sdr_2_5 = (all_errors <= 2.5).float().mean().item() * 100.0
        sdr_3_0 = (all_errors <= 3.0).float().mean().item() * 100.0
        sdr_4_0 = (all_errors <= 4.0).float().mean().item() * 100.0
    else:
        val_mae, val_rmse, sdr_2_0, sdr_2_5, sdr_3_0, sdr_4_0 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    metrics = {
        "mae": val_mae,
        "rmse": val_rmse,
        "sdr_2_0": sdr_2_0,
        "sdr_2_5": sdr_2_5,
        "sdr_3_0": sdr_3_0,
        "sdr_4_0": sdr_4_0,
    }

    return val_loss, metrics


def generate_and_log_training_charts(history: dict, log_to_mlflow: bool = True):
    """
    Generates PNG diagram charts for all metrics tracked during training history over epochs,
    and logs them as artifacts inside MLflow.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history or "epochs" not in history or len(history["epochs"]) == 0:
        print("⚠️ No training history available to plot charts.")
        return

    epochs = history["epochs"]
    with tempfile.TemporaryDirectory() as temp_dir:
        target_dir = Path(temp_dir)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.family": "sans-serif"})

        # 1. Loss Chart (Train Loss vs Val Loss)
        fig1, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax1.plot(epochs, history.get("train_loss", []), label="Train Loss", color="#e76f51", linewidth=2)
        ax1.plot(epochs, history.get("val_loss", []), label="Val Loss", color="#2a9d8f", linewidth=2, linestyle="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss Over Epochs", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig1.savefig(target_dir / "chart_training_val_loss.png", bbox_inches="tight")
        plt.close(fig1)

        # 2. Validation MAE Chart
        fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax2.plot(epochs, history.get("val_mae", []), label="Val MAE (px)", color="#e63946", linewidth=2, marker="o", markersize=4)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE (Pixels)")
        ax2.set_title("Validation Mean Absolute Error (MAE) Over Epochs", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig2.savefig(target_dir / "chart_val_mae.png", bbox_inches="tight")
        plt.close(fig2)

        # 3. Validation RMSE Chart
        fig3, ax3 = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax3.plot(epochs, history.get("val_rmse", []), label="Val RMSE (px)", color="#457b9d", linewidth=2, marker="s", markersize=4)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("RMSE (Pixels)")
        ax3.set_title("Validation Root Mean Squared Error (RMSE) Over Epochs", fontsize=12, fontweight="bold")
        ax3.legend(loc="upper right")
        ax3.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig3.savefig(target_dir / "chart_val_rmse.png", bbox_inches="tight")
        plt.close(fig3)

        # 4. Validation SDR Chart
        fig4, ax4 = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax4.plot(epochs, history.get("val_sdr_2_0", []), label="SDR @ 2.0px", color="#d62828", linewidth=1.8)
        ax4.plot(epochs, history.get("val_sdr_2_5", []), label="SDR @ 2.5px", color="#f77f00", linewidth=2.0)
        ax4.plot(epochs, history.get("val_sdr_3_0", []), label="SDR @ 3.0px", color="#fcbf49", linewidth=1.8)
        ax4.plot(epochs, history.get("val_sdr_4_0", []), label="SDR @ 4.0px", color="#003049", linewidth=1.8)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Successful Detection Rate (%)")
        ax4.set_ylim(0, 105)
        ax4.set_title("Validation Successful Detection Rates (SDR) Over Epochs", fontsize=12, fontweight="bold")
        ax4.legend(loc="lower right")
        ax4.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig4.savefig(target_dir / "chart_val_sdr.png", bbox_inches="tight")
        plt.close(fig4)

        # 5. Learning Rate Chart
        fig5, ax5 = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax5.plot(epochs, history.get("learning_rate", []), label="Learning Rate", color="#8338ec", linewidth=2)
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Learning Rate")
        ax5.set_yscale("log")
        ax5.set_title("Learning Rate Schedule Over Epochs", fontsize=12, fontweight="bold")
        ax5.legend(loc="upper right")
        ax5.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig5.savefig(target_dir / "chart_learning_rate.png", bbox_inches="tight")
        plt.close(fig5)

        # 6. Training Dashboard Grid (2x3)
        fig6, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
        axes[0, 0].plot(epochs, history.get("train_loss", []), label="Train Loss", color="#e76f51")
        axes[0, 0].plot(epochs, history.get("val_loss", []), label="Val Loss", color="#2a9d8f", linestyle="--")
        axes[0, 0].set_title("Loss Curves", fontweight="bold", fontsize=10)
        axes[0, 0].set_xlabel("Epoch", fontsize=8)
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        axes[0, 1].plot(epochs, history.get("val_mae", []), label="Val MAE", color="#e63946")
        axes[0, 1].set_title("Validation MAE (px)", fontweight="bold", fontsize=10)
        axes[0, 1].set_xlabel("Epoch", fontsize=8)
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        axes[0, 2].plot(epochs, history.get("val_rmse", []), label="Val RMSE", color="#457b9d")
        axes[0, 2].set_title("Validation RMSE (px)", fontweight="bold", fontsize=10)
        axes[0, 2].set_xlabel("Epoch", fontsize=8)
        axes[0, 2].grid(True, linestyle="--", alpha=0.5)

        axes[1, 0].plot(epochs, history.get("val_sdr_2_0", []), label="SDR 2.0px", color="#d62828")
        axes[1, 0].plot(epochs, history.get("val_sdr_2_5", []), label="SDR 2.5px", color="#f77f00")
        axes[1, 0].plot(epochs, history.get("val_sdr_3_0", []), label="SDR 3.0px", color="#fcbf49")
        axes[1, 0].plot(epochs, history.get("val_sdr_4_0", []), label="SDR 4.0px", color="#003049")
        axes[1, 0].set_title("Validation SDR (%)", fontweight="bold", fontsize=10)
        axes[1, 0].set_xlabel("Epoch", fontsize=8)
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        axes[1, 1].plot(epochs, history.get("learning_rate", []), label="Learning Rate", color="#8338ec")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("Learning Rate Schedule", fontweight="bold", fontsize=10)
        axes[1, 1].set_xlabel("Epoch", fontsize=8)
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        axes[1, 2].plot(epochs, history.get("epoch_time_seconds", []), label="Epoch Time (s)", color="#3a86ff")
        axes[1, 2].set_title("Epoch Duration (Seconds)", fontweight="bold", fontsize=10)
        axes[1, 2].set_xlabel("Epoch", fontsize=8)
        axes[1, 2].grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("TRAINING & VALIDATION METRIC CURVES DASHBOARD", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig6.savefig(target_dir / "chart_training_dashboard.png", bbox_inches="tight")
        plt.close(fig6)

        if log_to_mlflow:
            try:
                active_run = mlflow.active_run()
                if active_run is not None:
                    mlflow.log_artifacts(str(target_dir), artifact_path="training_metric_charts")
                    print(f"--> Successfully logged training metric PNG charts to MLflow artifact path 'training_metric_charts'")
            except Exception as ml_err:
                print(f"⚠️ Warning: Could not log training charts to MLflow: {ml_err}")


def main(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOPPING_PATIENCE,
    lr: float = LR,
    llrd_decay_rate: float = LLRD_DECAY_RATE,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    run_name: str | None = None,
    tracking_username: str | None = None,
    tracking_password: str | None = None,
    skip_eval: bool = False,
    ftp_host: str | None = None,
    ftp_port: int | str | None = None,
    ftp_user: str | None = None,
    ftp_password: str | None = None,
    ftp_remote_dir: str | None = None,
    ftp_tls: bool | None = None,
    skip_ftp: bool = False,
):
    load_dotenv()

    # --- MLflow Setup ---
    user = tracking_username or os.getenv("MLFLOW_TRACKING_USERNAME")
    pwd = tracking_password or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if user:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
    if pwd:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd

    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "cvm-swin-gcn")
    mlflow.set_experiment(exp_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        train_img_dir=TRAIN_IMG_DIR,
        train_npz_dir=TRAIN_NPZ_DIR,
        val_img_dir=VAL_IMG_DIR,
        val_npz_dir=VAL_NPZ_DIR,
        batch_size=batch_size,
    )

    model = CephalometricSwinGCN(num_landmarks=NUM_LANDMARKS).to(device)

    # Layer-wise Learning Rate Decay (LLRD) Parameter Grouping
    param_groups = get_llrd_param_groups(
        model,
        base_lr=lr,
        weight_decay=1e-4,
        decay_rate=llrd_decay_rate,
    )
    optimizer = optim.AdamW(param_groups)
    print(f"--> Initialized AdamW optimizer with LLRD (base LR: {lr}, decay rate: {llrd_decay_rate}, {len(param_groups)} param groups)")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # Losses & Landmark Weights
    awl_loss = AdaptiveWingLoss().to(device)
    wing_loss = WingLoss().to(device)
    graph_loss = AnatomicalGraphLoss(model.adj_matrix).to(device)
    landmark_weights = get_landmark_weights(device)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training with MLflow experiment '{exp_name}'...")

    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "llrd_decay_rate": llrd_decay_rate,
                "lambda_heatmap": LAMBDA_HM,
                "lambda_coord": LAMBDA_CD,
                "lambda_graph": LAMBDA_GRAPH,
                "early_stopping_patience": patience,
                "save_path": SAVE_PATH,
                "img_size": 640,
                "num_landmarks": NUM_LANDMARKS,
                "optimizer": "AdamW-LLRD",
                "scheduler": "ReduceLROnPlateau",
                "heatmap_loss": "AdaptiveWingLoss",
                "coord_loss": "WingLoss",
                "graph_loss": "AnatomicalGraphLoss",
                "device": str(device),
            }
        )


        history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_sdr_2_0": [],
            "val_sdr_2_5": [],
            "val_sdr_3_0": [],
            "val_sdr_4_0": [],
            "learning_rate": [],
            "epoch_time_seconds": [],
        }

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                awl_loss,
                wing_loss,
                graph_loss,
                landmark_weights,
                LAMBDA_HM,
                LAMBDA_CD,
                LAMBDA_GRAPH,
                device,
                epoch=epoch + 1,
                epochs=epochs,
            )

            val_loss, metrics = validate_epoch(
                model,
                val_loader,
                awl_loss,
                wing_loss,
                graph_loss,
                landmark_weights,
                LAMBDA_HM,
                LAMBDA_CD,
                LAMBDA_GRAPH,
                device,
                img_size=640,
                epoch=epoch + 1,
                epochs=epochs,
            )

            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]

            scheduler.step(val_loss)

            # Accumulate history metrics
            history["epochs"].append(epoch + 1)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_mae"].append(float(metrics["mae"]))
            history["val_rmse"].append(float(metrics["rmse"]))
            history["val_sdr_2_0"].append(float(metrics["sdr_2_0"]))
            history["val_sdr_2_5"].append(float(metrics["sdr_2_5"]))
            history["val_sdr_3_0"].append(float(metrics["sdr_3_0"]))
            history["val_sdr_4_0"].append(float(metrics["sdr_4_0"]))
            history["learning_rate"].append(float(current_lr))
            history["epoch_time_seconds"].append(float(epoch_time))

            # Log per-epoch metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": metrics["mae"],
                    "val_rmse": metrics["rmse"],
                    "val_sdr_2_0": metrics["sdr_2_0"],
                    "val_sdr_2_5": metrics["sdr_2_5"],
                    "val_sdr_3_0": metrics["sdr_3_0"],
                    "val_sdr_4_0": metrics["sdr_4_0"],
                    "learning_rate": current_lr,
                    "epoch_time_seconds": epoch_time,
                },
                step=epoch + 1,
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"MAE: {metrics['mae']:.2f} px | RMSE: {metrics['rmse']:.2f} px | "
                f"SDR@2.5px: {metrics['sdr_2_5']:.1f}%"
            )

            # Checkpoint Saving & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "val_mae": metrics["mae"],
                        "val_rmse": metrics["rmse"],
                        "val_sdr_2_5": metrics["sdr_2_5"],
                    },
                    SAVE_PATH,
                )
                print(
                    f"--> Saved new best model locally (Val Loss: {best_val_loss:.4f}, "
                    f"MAE: {metrics['mae']:.2f} px, RMSE: {metrics['rmse']:.2f} px, SDR@2.5px: {metrics['sdr_2_5']:.1f}%)"
                )

                # Log best metrics to MLflow
                mlflow.log_metrics(
                    {
                        "best_val_loss": best_val_loss,
                        "best_val_mae": metrics["mae"],
                        "best_val_rmse": metrics["rmse"],
                        "best_val_sdr_2_5": metrics["sdr_2_5"],
                    },
                    step=epoch + 1,
                )
            else:
                patience_counter += 1
                print(f"No improvement in validation loss for {patience_counter} epoch(s).")
                if patience_counter >= patience:
                    print(f"Early stopping triggered: Validation loss did not improve for {patience} consecutive epochs.")
                    break

        # --- GENERATE & LOG TRAINING METRIC PNG CHARTS TO MLFLOW ---
        print("\n--- Generating and Logging Training Metric PNG Charts to MLflow ---")
        generate_and_log_training_charts(history, log_to_mlflow=True)

        # --- LOG BEST MODEL CHECKPOINT TO MLFLOW ---
        if os.path.exists(SAVE_PATH):
            print("\n--- Logging Best Model Checkpoint to MLflow ---")
            try:
                mlflow.log_artifact(SAVE_PATH, artifact_path="checkpoints")
                print(f"--> Successfully logged best checkpoint artifact '{SAVE_PATH}' to MLflow.")
            except Exception as ml_err:
                print(f"⚠️ Warning: Could not log checkpoint artifact to MLflow: {ml_err}")

        # --- POST-TRAINING EVALUATION & MLFLOW LOGGING ---
        eval_metrics_json_path = None
        if not skip_eval and os.path.exists(SAVE_PATH):
            test_img_dir = "dataset/test/images"
            test_npz_dir = "dataset/test/labels"
            if os.path.exists(test_img_dir) and os.path.exists(test_npz_dir):
                print("\n--- Running Post-Training Evaluation with src/eval.py ---")
                try:
                    summary_metrics, eval_run_folder = run_evaluation(
                        weights_path=SAVE_PATH,
                        test_img_dir=test_img_dir,
                        test_npz_dir=test_npz_dir,
                        output_dir="evaluation",
                        img_size=640,
                    )

                    # Log all test summary metrics into MLflow
                    eval_metrics_to_log = {}
                    for k, v in summary_metrics.items():
                        if isinstance(v, (int, float)):
                            clean_k = f"test_{k}".replace("@", "").replace(".", "_").replace(" ", "_")
                            eval_metrics_to_log[clean_k] = float(v)

                    mlflow.log_metrics(eval_metrics_to_log)

                    # Log evaluation artifacts (report, metrics json, visualization PNGs)
                    if os.path.exists(eval_run_folder):
                        mlflow.log_artifacts(eval_run_folder, artifact_path="evaluation")
                        print(f"--> Successfully logged evaluation artifacts from '{eval_run_folder}' to MLflow.")

                        candidate_json = Path(eval_run_folder) / "metrics.json"
                        if candidate_json.exists():
                            eval_metrics_json_path = str(candidate_json)
                except Exception as e:
                    print(f"❌ Warning: Post-training evaluation failed: {e}")
            else:
                print("⚠️ Test dataset directories not found; skipping post-training evaluation.")

        # --- UPLOAD FINAL MODEL AND METRICS JSON TO FTP HOST ---
        if not skip_ftp:
            print("\n--- Uploading Final Model and Metrics JSON to FTP Host ---")
            ftp_files = []
            if os.path.exists(SAVE_PATH):
                ftp_files.append(SAVE_PATH)

            if eval_metrics_json_path and os.path.exists(eval_metrics_json_path):
                ftp_files.append(eval_metrics_json_path)
            else:
                # Fallback: check if any recent metrics.json exists in evaluation/
                eval_base = Path("evaluation")
                if eval_base.exists():
                    found_jsons = sorted(
                        eval_base.glob("**/metrics.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if found_jsons:
                        ftp_files.append(str(found_jsons[0]))

            if ftp_files:
                try:
                    upload_files_to_ftp(
                        files=ftp_files,
                        ftp_host=ftp_host,
                        ftp_port=ftp_port,
                        ftp_user=ftp_user,
                        ftp_password=ftp_password,
                        remote_dir=ftp_remote_dir,
                        use_tls=ftp_tls,
                    )
                except Exception as ftp_err:
                    print(f"⚠️ Warning: FTP upload encountered an error: {ftp_err}")
            else:
                print("ℹ️ No model checkpoint or metrics.json found to upload to FTP.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Cephalometric Swin-GCN model with MLflow tracking")
    parser.add_argument("--epochs", "-e", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--patience", "-p", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience (epochs)")
    parser.add_argument("--lr", "-l", type=float, default=LR, help="Learning rate")
    parser.add_argument("--llrd-decay-rate", type=float, default=LLRD_DECAY_RATE, help="Layer-wise Learning Rate Decay factor (default: 0.8)")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--tracking-username", "--mlflow-username", type=str, default=None, help="MLflow tracking username")
    parser.add_argument("--tracking-password", "--mlflow-password", type=str, default=None, help="MLflow tracking password")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training evaluation step")
    parser.add_argument("--ftp-host", type=str, default=None, help="FTP host (e.g. ftp.example.com or IP)")
    parser.add_argument("--ftp-port", type=str, default=None, help="FTP port (default: 21)")
    parser.add_argument("--ftp-user", "--ftp-username", type=str, default=None, help="FTP username")
    parser.add_argument("--ftp-password", "--ftp-pass", type=str, default=None, help="FTP password")
    parser.add_argument("--ftp-remote-dir", "--ftp-dir", type=str, default=None, help="Remote directory path on FTP server")
    parser.add_argument("--ftp-tls", action="store_true", help="Use FTPS / TLS encryption for FTP upload")
    parser.add_argument("--skip-ftp", action="store_true", help="Skip uploading model and metrics to FTP server")

    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        llrd_decay_rate=args.llrd_decay_rate,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        run_name=args.run_name,
        tracking_username=args.tracking_username,
        tracking_password=args.tracking_password,
        skip_eval=args.skip_eval,
        ftp_host=args.ftp_host,
        ftp_port=args.ftp_port,
        ftp_user=args.ftp_user,
        ftp_password=args.ftp_password,
        ftp_remote_dir=args.ftp_remote_dir,
        ftp_tls=args.ftp_tls if args.ftp_tls else None,
        skip_ftp=args.skip_ftp,
    )

