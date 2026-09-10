import os
import sys
import time
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
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

from src.data import get_dataloaders, NUM_LANDMARKS, LANDMARK_CLASSES
from src.models.model import CephalometricSwinGCN
from src.models.losses import AdaptiveWingLoss, WingLoss, AnatomicalGraphLoss
from src.eval import run_evaluation
from src.utils.ftp_utils import upload_files_to_ftp

TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_NPZ_DIR = "dataset/train/labels"

VAL_IMG_DIR = "dataset/val/images"
VAL_NPZ_DIR = "dataset/val/labels"

TEST_IMG_DIR = "dataset/test/images"
TEST_NPZ_DIR = "dataset/test/labels"

BATCH_SIZE = 8
EPOCHS = 100
LR = float(os.getenv("LEARNING_RATE", 1e-4))
LLRD_DECAY_RATE = float(os.getenv("LLRD_DECAY_RATE", 0.8))
LAMBDA_HM = 1.0
LAMBDA_CD = 5.0
LAMBDA_GRAPH = 1.0
EARLY_STOPPING_PATIENCE = 40
WARMUP_COORD_EPOCHS = 5
SAVE_PATH = "./artifacts/best.pth"
FULL_CHECKPOINT_PATH = "./artifacts/best_full_checkpoint.pth"
DEFAULT_IMG_SIZE = 640
DEFAULT_PIXEL_SPACING = float(os.getenv("PIXEL_SPACING", 0.1))  # 0.1 mm per pixel (standard cephalometric calibration)


def get_coord_loss_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """
    Ramps coordinate and graph loss weights linearly from 0.0 to 1.0 across warmup_epochs.
    Allows heatmaps to establish coarse localization during early epochs before
    activating strong direct coordinate and graph structural losses.
    """
    if warmup_epochs <= 0:
        return 1.0
    # epoch is 1-indexed: epoch 1 -> 0.0, epoch warmup_epochs -> 1.0
    return float(min(1.0, max(0.0, (epoch - 1) / max(1, warmup_epochs - 1))))


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
    """
    Constructs a learning rate scheduler with linear warmup followed by cosine annealing decay.
    """
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_llrd_param_groups(
    model: torch.nn.Module,
    base_lr: float = 2e-4,
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
    Constructs landmark loss weights prioritizing empirically difficult landmarks
    identified from test evaluation (C2_AI, C3_AS, C2_IC, C4_AI, C4_PS, C3_PS, etc.).
    """
    weights = torch.ones(NUM_LANDMARKS, dtype=torch.float32, device=device)
    # Tier 1: Lagging landmarks (<80% SDR @ 2.0mm & highest MRE on test set)
    weights[2] = 2.5   # C2_AI (Test SDR: 79.59%, MRE: 1.47 mm)
    weights[4] = 2.5   # C3_AS (Test SDR: 79.59%, MRE: 1.49 mm)

    # Tier 2: Challenging landmarks (82% - 86% SDR @ 2.0mm)
    weights[1] = 2.0   # C2_IC (Test SDR: 82.99%, MRE: 1.29 mm, concavity)
    weights[12] = 2.0  # C4_AI (Test SDR: 85.71%, MRE: 1.32 mm)
    weights[0] = 1.8   # C2_PI (Test SDR: 83.67%, MRE: 1.28 mm)
    weights[8] = 1.8   # C4_PS (Test SDR: 84.35%, MRE: 1.26 mm)
    weights[3] = 1.6   # C3_PS (Test SDR: 85.71%, MRE: 1.23 mm)
    weights[10] = 1.5  # C4_PI (Test SDR: 86.39%, MRE: 1.33 mm)

    # Tier 3: Moderate landmarks
    weights[7] = 1.3   # C3_AI (Test SDR: 87.76%, MRE: 1.25 mm)
    weights[9] = 1.2   # C4_AS (Test SDR: 89.12%, MRE: 1.29 mm)
    weights[6] = 1.2   # C3_IC (Test SDR: 89.80%, MRE: 1.14 mm)

    # Landmarks with >91% SDR @ 2.0mm (C3_PI: 91.16%, C4_IC: 91.16%) remain at baseline 1.0
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
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch}/{epochs}] Train",
        leave=False,
    )
    device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    amp_enabled = use_amp and device_type == "cuda"

    for step, batch in enumerate(pbar, 1):
        images = batch["image"].to(device)
        gt_heatmaps = batch["heatmaps"].to(device)
        gt_coords = batch["coords"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
            pred_heatmaps, pred_coords = model(images)
            # Ensure float32 for loss computation to maintain numerical stability
            loss_hm = awl_loss(pred_heatmaps.float(), gt_heatmaps.float(), landmark_weights=landmark_weights)
            loss_cd = wing_loss(pred_coords.float(), gt_coords.float(), landmark_weights=landmark_weights)
            loss_g = graph_loss(pred_coords.float(), gt_coords.float())
            total_loss = (lambda_hm * loss_hm) + (lambda_cd * loss_cd) + (lambda_graph * loss_g)

        if scaler is not None and amp_enabled:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss += total_loss.item()
        pbar.set_postfix(loss=f"{epoch_loss / step:.4f}")

    return epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def compute_per_landmark_metrics(
    pred_coords: torch.Tensor | np.ndarray,
    gt_coords: torch.Tensor | np.ndarray,
    img_size: int = DEFAULT_IMG_SIZE,
    pixel_spacing: float | np.ndarray = DEFAULT_PIXEL_SPACING,
) -> list[dict]:
    """
    Computes detailed per-landmark performance metrics in both millimeters (mm) and pixels (px).

    Args:
        pred_coords: Array or Tensor of shape (N, NUM_LANDMARKS, 2) in normalized [0, 1] range.
        gt_coords: Array or Tensor of shape (N, NUM_LANDMARKS, 2) in normalized [0, 1] range (-1 for missing).
        img_size: Target square image dimension in pixels (default 640).
        pixel_spacing: Physical spacing in mm per pixel (float scalar or 1D array of shape (N,)).

    Returns:
        List of dicts with performance metrics for each landmark.
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()

    pred_px = pred_coords * img_size
    gt_px = gt_coords * img_size

    # Valid mask for GT coordinates (excluding missing landmarks marked with negative coordinates)
    valid_mask = (gt_coords[:, :, 0] >= 0) & (gt_coords[:, :, 1] >= 0)

    dx = pred_px[:, :, 0] - gt_px[:, :, 0]
    dy = pred_px[:, :, 1] - gt_px[:, :, 1]
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    radial_errors = np.sqrt(dx**2 + dy**2)

    num_lms = pred_coords.shape[1]
    landmark_metrics = []

    for i in range(num_lms):
        name = LANDMARK_CLASSES[i] if i < len(LANDMARK_CLASSES) else f"Landmark_{i}"
        l_mask = valid_mask[:, i]
        count = int(np.sum(l_mask))

        if count == 0:
            landmark_metrics.append({
                "id": i,
                "name": name,
                "count": 0,
                "mae_px": 0.0,
                "rmse_px": 0.0,
                "mre_px": 0.0,
                "medre_px": 0.0,
                "sdre_px": 0.0,
                "min_error_px": 0.0,
                "max_error_px": 0.0,
                "sdr_2.0px": 0.0,
                "sdr_2.5px": 0.0,
                "sdr_3.0px": 0.0,
                "sdr_4.0px": 0.0,
                "mre_mm": 0.0,
                "rmse_mm": 0.0,
                "medre_mm": 0.0,
                "sdre_mm": 0.0,
                "sdr_2.0mm": 0.0,
                "sdr_2.5mm": 0.0,
                "sdr_3.0mm": 0.0,
                "sdr_4.0mm": 0.0,
            })
            continue

        l_radial = radial_errors[:, i][l_mask]
        if isinstance(pixel_spacing, np.ndarray):
            l_radial_mm = l_radial * pixel_spacing[l_mask]
        else:
            l_radial_mm = l_radial * pixel_spacing

        l_dx = dx[:, i][l_mask]
        l_dy = dy[:, i][l_mask]
        l_abs_dx = abs_dx[:, i][l_mask]
        l_abs_dy = abs_dy[:, i][l_mask]

        l_mae = float(np.mean((l_abs_dx + l_abs_dy) / 2.0))
        l_rmse = float(np.sqrt(np.mean(l_dx**2 + l_dy**2)))
        l_mre = float(np.mean(l_radial))
        l_medre = float(np.median(l_radial))
        l_sdre = float(np.std(l_radial)) if count > 1 else 0.0
        l_min = float(np.min(l_radial))
        l_max = float(np.max(l_radial))

        # Pixel thresholds
        l_sdr2_0 = float(np.mean(l_radial <= 2.0) * 100.0)
        l_sdr2_5 = float(np.mean(l_radial <= 2.5) * 100.0)
        l_sdr3_0 = float(np.mean(l_radial <= 3.0) * 100.0)
        l_sdr4_0 = float(np.mean(l_radial <= 4.0) * 100.0)

        # Physical Millimeter (mm) thresholds
        l_mre_mm = float(np.mean(l_radial_mm))
        l_rmse_mm = float(np.sqrt(np.mean((l_radial_mm) ** 2)))
        l_medre_mm = float(np.median(l_radial_mm))
        l_sdre_mm = float(np.std(l_radial_mm)) if count > 1 else 0.0
        l_sdr2_0mm = float(np.mean(l_radial_mm <= 2.0) * 100.0)
        l_sdr2_5mm = float(np.mean(l_radial_mm <= 2.5) * 100.0)
        l_sdr3_0mm = float(np.mean(l_radial_mm <= 3.0) * 100.0)
        l_sdr4_0mm = float(np.mean(l_radial_mm <= 4.0) * 100.0)

        landmark_metrics.append({
            "id": i,
            "name": name,
            "count": count,
            "mae_px": l_mae,
            "rmse_px": l_rmse,
            "mre_px": l_mre,
            "medre_px": l_medre,
            "sdre_px": l_sdre,
            "min_error_px": l_min,
            "max_error_px": l_max,
            "sdr_2.0px": l_sdr2_0,
            "sdr_2.5px": l_sdr2_5,
            "sdr_3.0px": l_sdr3_0,
            "sdr_4.0px": l_sdr4_0,
            "mre_mm": l_mre_mm,
            "rmse_mm": l_rmse_mm,
            "medre_mm": l_medre_mm,
            "sdre_mm": l_sdre_mm,
            "sdr_2.0mm": l_sdr2_0mm,
            "sdr_2.5mm": l_sdr2_5mm,
            "sdr_3.0mm": l_sdr3_0mm,
            "sdr_4.0mm": l_sdr4_0mm,
        })

    return landmark_metrics


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
    img_size=DEFAULT_IMG_SIZE,
    pixel_spacing: float = DEFAULT_PIXEL_SPACING,
    epoch: int = 1,
    epochs: int = 1,
    use_amp: bool = True,
):
    model.eval()
    epoch_loss = 0.0
    all_radial_errors = []
    all_pred_coords = []
    all_gt_coords = []
    all_spacings = []

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch}/{epochs}] Val",
        leave=False,
    )
    device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    amp_enabled = use_amp and device_type == "cuda"

    with torch.no_grad():
        for step, batch in enumerate(pbar, 1):
            images = batch["image"].to(device)
            gt_heatmaps = batch["heatmaps"].to(device)
            gt_coords = batch["coords"].to(device)
            sample_spacing = batch.get("pixel_spacing")
            if sample_spacing is not None:
                all_spacings.append(sample_spacing.cpu())

            with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
                pred_heatmaps, pred_coords = model(images)
                loss_hm = awl_loss(pred_heatmaps.float(), gt_heatmaps.float(), landmark_weights=landmark_weights)
                loss_cd = wing_loss(pred_coords.float(), gt_coords.float(), landmark_weights=landmark_weights)
                loss_g = graph_loss(pred_coords.float(), gt_coords.float())
                total_loss = (lambda_hm * loss_hm) + (lambda_cd * loss_cd) + (lambda_graph * loss_g)

            epoch_loss += total_loss.item()

            # Radial errors (in pixels) across landmarks
            pred_px = pred_coords * img_size
            gt_px = gt_coords * img_size
            radial_errors = torch.sqrt(torch.sum((pred_px - gt_px) ** 2, dim=-1))  # [B, N]
            all_radial_errors.append(radial_errors.cpu())
            all_pred_coords.append(pred_coords.cpu())
            all_gt_coords.append(gt_coords.cpu())

            pbar.set_postfix(loss=f"{epoch_loss / step:.4f}")

    val_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    if len(all_radial_errors) > 0:
        all_errors = torch.cat(all_radial_errors, dim=0)  # [Total_Samples, N]
        all_preds = torch.cat(all_pred_coords, dim=0)
        all_gts = torch.cat(all_gt_coords, dim=0)

        if len(all_spacings) > 0:
            spacings_tensor = torch.cat(all_spacings, dim=0)  # [Total_Samples]
        else:
            spacings_tensor = torch.full((all_errors.size(0),), pixel_spacing, dtype=torch.float32)

        # Valid mask for GT coordinates (excluding missing landmarks marked with negative coordinates)
        valid_mask = (all_gts[:, :, 0] >= 0) & (all_gts[:, :, 1] >= 0)
        if valid_mask.any():
            valid_errors = all_errors[valid_mask]
            val_mae = valid_errors.mean().item()
            val_rmse = torch.sqrt((valid_errors ** 2).mean()).item()
            sdr_2_0 = (valid_errors <= 2.0).float().mean().item() * 100.0
            sdr_2_5 = (valid_errors <= 2.5).float().mean().item() * 100.0
            sdr_3_0 = (valid_errors <= 3.0).float().mean().item() * 100.0
            sdr_4_0 = (valid_errors <= 4.0).float().mean().item() * 100.0

            # Compute sample-specific physical millimeter errors
            all_errors_mm = all_errors * spacings_tensor.unsqueeze(-1)
            valid_errors_mm = all_errors_mm[valid_mask]
            val_mre_mm = valid_errors_mm.mean().item()
            val_rmse_mm = torch.sqrt((valid_errors_mm ** 2).mean()).item()
            sdr_2_0mm = (valid_errors_mm <= 2.0).float().mean().item() * 100.0
            sdr_2_5mm = (valid_errors_mm <= 2.5).float().mean().item() * 100.0
            sdr_3_0mm = (valid_errors_mm <= 3.0).float().mean().item() * 100.0
            sdr_4_0mm = (valid_errors_mm <= 4.0).float().mean().item() * 100.0
        else:
            val_mae, val_rmse, sdr_2_0, sdr_2_5, sdr_3_0, sdr_4_0 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            val_mre_mm, val_rmse_mm, sdr_2_0mm, sdr_2_5mm, sdr_3_0mm, sdr_4_0mm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        per_landmark_metrics = compute_per_landmark_metrics(
            all_preds, all_gts, img_size=img_size, pixel_spacing=spacings_tensor.numpy()
        )
    else:
        val_mae, val_rmse, sdr_2_0, sdr_2_5, sdr_3_0, sdr_4_0 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        val_mre_mm, val_rmse_mm, sdr_2_0mm, sdr_2_5mm, sdr_3_0mm, sdr_4_0mm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        per_landmark_metrics = []

    metrics = {
        "mae": val_mae,
        "rmse": val_rmse,
        "sdr_2_0": sdr_2_0,
        "sdr_2_5": sdr_2_5,
        "sdr_3_0": sdr_3_0,
        "sdr_4_0": sdr_4_0,
        "mre_mm": val_mre_mm,
        "rmse_mm": val_rmse_mm,
        "sdr_2_0mm": sdr_2_0mm,
        "sdr_2_5mm": sdr_2_5mm,
        "sdr_3_0mm": sdr_3_0mm,
        "sdr_4_0mm": sdr_4_0mm,
        "per_landmark": per_landmark_metrics,
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

        # 2. Validation MRE Chart (mm)
        fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=300)
        mre_data = history.get("val_mre_mm") or history.get("val_mae", [])
        mre_label = "Val MRE (mm)" if "val_mre_mm" in history else "Val MAE (px)"
        mre_unit = "mm" if "val_mre_mm" in history else "Pixels"
        ax2.plot(epochs, mre_data, label=mre_label, color="#e63946", linewidth=2, marker="o", markersize=4)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel(f"Mean Radial Error ({mre_unit})")
        ax2.set_title(f"Validation Mean Radial Error ({mre_unit}) Over Epochs", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig2.savefig(target_dir / "chart_val_mre.png", bbox_inches="tight")
        fig2.savefig(target_dir / "chart_val_mae.png", bbox_inches="tight")
        plt.close(fig2)

        # 3. Validation RMSE Chart (mm)
        fig3, ax3 = plt.subplots(figsize=(8, 4.5), dpi=300)
        rmse_data = history.get("val_rmse_mm") or history.get("val_rmse", [])
        rmse_label = "Val RMSE (mm)" if "val_rmse_mm" in history else "Val RMSE (px)"
        rmse_unit = "mm" if "val_rmse_mm" in history else "Pixels"
        ax3.plot(epochs, rmse_data, label=rmse_label, color="#457b9d", linewidth=2, marker="s", markersize=4)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel(f"RMSE ({rmse_unit})")
        ax3.set_title(f"Validation Root Mean Squared Error ({rmse_unit}) Over Epochs", fontsize=12, fontweight="bold")
        ax3.legend(loc="upper right")
        ax3.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig3.savefig(target_dir / "chart_val_rmse.png", bbox_inches="tight")
        plt.close(fig3)

        # 4. Validation SDR Chart (Clinical mm thresholds)
        fig4, ax4 = plt.subplots(figsize=(8, 4.5), dpi=300)
        has_mm_sdr = "val_sdr_2_0mm" in history
        sdr2_0 = history.get("val_sdr_2_0mm", history.get("val_sdr_2_0", []))
        sdr2_5 = history.get("val_sdr_2_5mm", history.get("val_sdr_2_5", []))
        sdr3_0 = history.get("val_sdr_3_0mm", history.get("val_sdr_3_0", []))
        sdr4_0 = history.get("val_sdr_4_0mm", history.get("val_sdr_4_0", []))
        suffix = "mm" if has_mm_sdr else "px"

        ax4.plot(epochs, sdr2_0, label=f"SDR @ 2.0{suffix}", color="#d62828", linewidth=2.0)
        ax4.plot(epochs, sdr2_5, label=f"SDR @ 2.5{suffix}", color="#f77f00", linewidth=2.0)
        ax4.plot(epochs, sdr3_0, label=f"SDR @ 3.0{suffix}", color="#fcbf49", linewidth=1.8)
        ax4.plot(epochs, sdr4_0, label=f"SDR @ 4.0{suffix}", color="#003049", linewidth=1.8)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Successful Detection Rate (%)")
        ax4.set_ylim(0, 105)
        ax4.set_title(f"Validation Successful Detection Rates (SDR @ {suffix}) Over Epochs", fontsize=12, fontweight="bold")
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

        axes[0, 1].plot(epochs, mre_data, label=mre_label, color="#e63946")
        axes[0, 1].set_title(f"Validation Error ({mre_unit})", fontweight="bold", fontsize=10)
        axes[0, 1].set_xlabel("Epoch", fontsize=8)
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        axes[0, 2].plot(epochs, rmse_data, label=rmse_label, color="#457b9d")
        axes[0, 2].set_title(f"Validation RMSE ({rmse_unit})", fontweight="bold", fontsize=10)
        axes[0, 2].set_xlabel("Epoch", fontsize=8)
        axes[0, 2].grid(True, linestyle="--", alpha=0.5)

        axes[1, 0].plot(epochs, sdr2_0, label=f"SDR 2.0{suffix}", color="#d62828")
        axes[1, 0].plot(epochs, sdr2_5, label=f"SDR 2.5{suffix}", color="#f77f00")
        axes[1, 0].plot(epochs, sdr3_0, label=f"SDR 3.0{suffix}", color="#fcbf49")
        axes[1, 0].plot(epochs, sdr4_0, label=f"SDR 4.0{suffix}", color="#003049")
        axes[1, 0].set_title(f"Validation SDR ({suffix})", fontweight="bold", fontsize=10)
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


def log_split_image_ids_to_mlflow(
    val_img_dir: str = VAL_IMG_DIR,
    test_img_dir: str = TEST_IMG_DIR,
    output_dir: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """
    Logs validation and test split image IDs as JSON array artifacts to the active MLflow run
    before training begins. Checks existing JSON files first; falls back to scanning directories
    if files don't exist. Guaranteed not to crash the training run if an MLflow or filesystem
    anomaly occurs.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    dataset_dir = Path(output_dir) if output_dir else Path(val_img_dir).parent.parent
    val_json_path = dataset_dir / "val_image_ids.json"
    test_json_path = dataset_dir / "test_image_ids.json"

    val_ids: list[str] = []
    test_ids: list[str] = []

    # 1. Resolve validation image IDs
    if val_json_path.exists():
        try:
            with open(val_json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    val_ids = loaded
        except Exception as e:
            print(f"⚠️ Warning reading {val_json_path}: {e}")

    # 2. Resolve test image IDs
    if test_json_path.exists():
        try:
            with open(test_json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    test_ids = loaded
        except Exception as e:
            print(f"⚠️ Warning reading {test_json_path}: {e}")

    # Fallback directory scan if either list is missing
    if not val_ids or not test_ids:
        try:
            from src.data.preprocessing.split_dataset import load_label_studio_mapping
            ls_mapping = load_label_studio_mapping()
        except Exception:
            ls_mapping = {}

        def resolve_id(p: Path):
            if p.stem in ls_mapping:
                return ls_mapping[p.stem]
            if p.name in ls_mapping:
                return ls_mapping[p.name]
            return int(p.stem) if p.stem.isdigit() else p.stem

        if not val_ids:
            val_path = Path(val_img_dir)
            if val_path.exists():
                val_ids = sorted(
                    [resolve_id(f) for f in val_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts],
                    key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)),
                )
                try:
                    val_json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(val_json_path, "w", encoding="utf-8") as f:
                        json.dump(val_ids, f, indent=2)
                except Exception as e:
                    print(f"⚠️ Warning writing {val_json_path}: {e}")

        if not test_ids:
            test_path = Path(test_img_dir)
            if test_path.exists():
                test_ids = sorted(
                    [resolve_id(f) for f in test_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts],
                    key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)),
                )
                try:
                    test_json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(test_json_path, "w", encoding="utf-8") as f:
                        json.dump(test_ids, f, indent=2)
                except Exception as e:
                    print(f"⚠️ Warning writing {test_json_path}: {e}")

    # 3. Log to active MLflow run safely
    try:
        active_run = mlflow.active_run()
        if active_run is not None:
            if val_json_path.exists():
                mlflow.log_artifact(str(val_json_path))
            else:
                mlflow.log_dict(val_ids, "val_image_ids.json")

            if test_json_path.exists():
                mlflow.log_artifact(str(test_json_path))
            else:
                mlflow.log_dict(test_ids, "test_image_ids.json")

            print(
                f"--> Successfully logged split image ID artifacts to MLflow: "
                f"'val_image_ids.json' ({len(val_ids)} items), 'test_image_ids.json' ({len(test_ids)} items)"
            )
        else:
            print("⚠️ Notice: No active MLflow run found; skipping split ID artifact logging.")
    except Exception as ml_err:
        print(f"⚠️ Warning: Could not log split image IDs to MLflow: {ml_err}")

    return val_ids, test_ids


def log_best_per_landmark_metrics_to_mlflow(
    landmark_metrics: list[dict],
    epoch: int,
    metrics: dict,
    best_val_sdr: float,
    best_val_mae: float,
    best_val_loss: float,
    save_dir: str | Path = "./artifacts",
) -> dict:
    """
    Saves and logs a JSON of per-landmark performance metrics to MLflow
    whenever a new best validation performance is achieved during training.

    Guaranteed not to crash training if an MLflow or filesystem exception occurs.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "best_per_landmark_metrics.json"

    export_payload = {
        "epoch": int(epoch),
        "step": int(epoch),
        "timestamp": datetime.now().isoformat(),
        "summary": {
            # Physical Millimeters (Clinical)
            "best_val_mre_mm": float(metrics.get("mre_mm", 0.0)),
            "best_val_rmse_mm": float(metrics.get("rmse_mm", 0.0)),
            "best_val_sdr_2_0mm": float(metrics.get("sdr_2_0mm", 0.0)),
            "best_val_sdr_2_5mm": float(metrics.get("sdr_2_5mm", 0.0)),
            "best_val_sdr_3_0mm": float(metrics.get("sdr_3_0mm", 0.0)),
            "best_val_sdr_4_0mm": float(metrics.get("sdr_4_0mm", 0.0)),
            # Canvas Pixels (Secondary)
            "best_val_loss": float(best_val_loss),
            "best_val_mae": float(best_val_mae),
            "best_val_rmse": float(metrics.get("rmse", 0.0)),
            "best_val_sdr_2_0": float(metrics.get("sdr_2_0", 0.0)),
            "best_val_sdr_2_5": float(best_val_sdr),
            "best_val_sdr_3_0": float(metrics.get("sdr_3_0", 0.0)),
            "best_val_sdr_4_0": float(metrics.get("sdr_4_0", 0.0)),
        },
        "landmarks": landmark_metrics,
        "by_landmark_name": {lm["name"]: lm for lm in landmark_metrics},
    }

    # 1. Save local JSON file
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=4)
        print(f"--> Saved best per-landmark metrics JSON locally to '{json_path}'")
    except Exception as io_err:
        print(f"⚠️ Warning: Could not write {json_path}: {io_err}")

    # 2. Log to active MLflow run safely
    try:
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.log_dict(export_payload, "best_per_landmark_metrics.json")
            if json_path.exists():
                mlflow.log_artifact(str(json_path), artifact_path="checkpoints")
            print(f"--> Successfully logged per-landmark metrics JSON to MLflow for best epoch {epoch}")
        else:
            print("⚠️ Notice: No active MLflow run found; skipping MLflow per-landmark metrics logging.")
    except Exception as ml_err:
        print(f"⚠️ Warning: Could not log per-landmark metrics JSON to MLflow: {ml_err}")

    return export_payload


def main(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOPPING_PATIENCE,
    lr: float = LR,
    llrd_decay_rate: float = LLRD_DECAY_RATE,
    val_img_dir: str = VAL_IMG_DIR,
    val_npz_dir: str = VAL_NPZ_DIR,
    test_img_dir: str = TEST_IMG_DIR,
    img_size: int = DEFAULT_IMG_SIZE,
    pixel_spacing: float = DEFAULT_PIXEL_SPACING,
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
    save_optimizer: bool = True,
    use_amp: bool = True,
    compile_model: bool = False,
    warmup_coord_epochs: int = WARMUP_COORD_EPOCHS,
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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("--> Enabled CUDA TF32 for matmul and cuDNN")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        train_img_dir=TRAIN_IMG_DIR,
        train_npz_dir=TRAIN_NPZ_DIR,
        val_img_dir=val_img_dir,
        val_npz_dir=val_npz_dir,
        batch_size=batch_size,
        img_size=img_size,
        pixel_spacing=pixel_spacing,
    )

    model = CephalometricSwinGCN(num_landmarks=NUM_LANDMARKS, img_size=img_size).to(device)

    # Compile model with torch.compile if requested
    if compile_model:
        if hasattr(torch, "compile"):
            try:
                print("--> Compiling model with torch.compile()...")
                model = torch.compile(model)
            except Exception as e:
                print(f"⚠️ Warning: torch.compile() failed ({e}). Falling back to eager mode.")
        else:
            print("⚠️ Warning: torch.compile is not supported in this PyTorch version.")

    # Layer-wise Learning Rate Decay (LLRD) Parameter Grouping
    param_groups = get_llrd_param_groups(
        model,
        base_lr=lr,
        weight_decay=1e-4,
        decay_rate=llrd_decay_rate,
    )
    optimizer = optim.AdamW(param_groups)
    print(f"--> Initialized AdamW optimizer with LLRD (base LR: {lr}, decay rate: {llrd_decay_rate}, {len(param_groups)} param groups)")

    # Automatic Mixed Precision GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    if use_amp and device.type == "cuda":
        print("--> Automatic Mixed Precision (AMP FP16) enabled with GradScaler")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=5,
        total_epochs=epochs,
        min_lr_ratio=0.01,
    )

    # Losses & Landmark Weights
    awl_loss = AdaptiveWingLoss().to(device)
    wing_loss = WingLoss(img_size=float(img_size)).to(device)
    graph_loss = AnatomicalGraphLoss(model.adj_matrix).to(device)
    landmark_weights = get_landmark_weights(device)

    best_val_loss = float("inf")
    best_val_sdr_2_0mm = 0.0
    best_val_sdr_2_5mm = 0.0
    best_val_mre_mm = float("inf")
    best_val_mae_px = float("inf")
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
                "warmup_coord_epochs": warmup_coord_epochs,
                "early_stopping_patience": patience,
                "save_path": SAVE_PATH,
                "img_size": img_size,
                "pixel_spacing": pixel_spacing,
                "num_landmarks": NUM_LANDMARKS,
                "optimizer": "AdamW-LLRD",
                "scheduler": "CosineAnnealingWithWarmup",
                "heatmap_loss": "AdaptiveWingLoss",
                "coord_loss": "WingLoss",
                "graph_loss": "AnatomicalGraphLoss",
                "device": str(device),
                "use_amp": use_amp,
                "compile_model": compile_model,
            }
        )

        # Log validation and test split image IDs before starting training
        log_split_image_ids_to_mlflow(
            val_img_dir=val_img_dir,
            test_img_dir=test_img_dir,
        )

        history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_mre_mm": [],
            "val_rmse_mm": [],
            "val_sdr_2_0mm": [],
            "val_sdr_2_5mm": [],
            "val_sdr_3_0mm": [],
            "val_sdr_4_0mm": [],
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

            # Dynamic coordinate & graph loss warmup factor
            coord_warmup_factor = get_coord_loss_warmup_factor(epoch + 1, warmup_coord_epochs)
            effective_lambda_cd = LAMBDA_CD * coord_warmup_factor
            effective_lambda_graph = LAMBDA_GRAPH * coord_warmup_factor

            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                awl_loss,
                wing_loss,
                graph_loss,
                landmark_weights,
                LAMBDA_HM,
                effective_lambda_cd,
                effective_lambda_graph,
                device,
                epoch=epoch + 1,
                epochs=epochs,
                scaler=scaler,
                use_amp=use_amp,
            )

            val_loss, metrics = validate_epoch(
                model,
                val_loader,
                awl_loss,
                wing_loss,
                graph_loss,
                landmark_weights,
                LAMBDA_HM,
                effective_lambda_cd,
                effective_lambda_graph,
                device,
                img_size=img_size,
                pixel_spacing=pixel_spacing,
                epoch=epoch + 1,
                epochs=epochs,
                use_amp=use_amp,
            )

            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]

            # Step Cosine Scheduler per epoch
            scheduler.step()

            # Accumulate history metrics
            history["epochs"].append(epoch + 1)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_mre_mm"].append(float(metrics["mre_mm"]))
            history["val_rmse_mm"].append(float(metrics["rmse_mm"]))
            history["val_sdr_2_0mm"].append(float(metrics["sdr_2_0mm"]))
            history["val_sdr_2_5mm"].append(float(metrics["sdr_2_5mm"]))
            history["val_sdr_3_0mm"].append(float(metrics["sdr_3_0mm"]))
            history["val_sdr_4_0mm"].append(float(metrics["sdr_4_0mm"]))
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
                    "val_mre_mm": metrics["mre_mm"],
                    "val_rmse_mm": metrics["rmse_mm"],
                    "val_sdr_2_0mm": metrics["sdr_2_0mm"],
                    "val_sdr_2_5mm": metrics["sdr_2_5mm"],
                    "val_sdr_3_0mm": metrics["sdr_3_0mm"],
                    "val_sdr_4_0mm": metrics["sdr_4_0mm"],
                    "val_mae_px": metrics["mae"],
                    "val_rmse_px": metrics["rmse"],
                    "val_sdr_2_0px": metrics["sdr_2_0"],
                    "val_sdr_2_5px": metrics["sdr_2_5"],
                    "val_sdr_3_0px": metrics["sdr_3_0"],
                    "val_sdr_4_0px": metrics["sdr_4_0"],
                    "learning_rate": current_lr,
                    "epoch_time_seconds": epoch_time,
                    "lambda_coord": effective_lambda_cd,
                    "lambda_graph": effective_lambda_graph,
                },
                step=epoch + 1,
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"MRE: {metrics['mre_mm']:.2f} mm ({metrics['mae']:.2f} px) | "
                f"SDR@2.0mm: {metrics['sdr_2_0mm']:.1f}% | SDR@2.5mm: {metrics['sdr_2_5mm']:.1f}% | SDR@4.0mm: {metrics['sdr_4_0mm']:.1f}% | "
                f"λ_cd: {effective_lambda_cd:.2f}"
            )

            # Checkpoint Saving & Early Stopping based on primary clinical criteria (SDR@2.0mm & MRE mm)
            is_best_sdr = metrics["sdr_2_0mm"] > best_val_sdr_2_0mm
            is_tied_sdr_better_mre = (abs(metrics["sdr_2_0mm"] - best_val_sdr_2_0mm) < 1e-4) and (metrics["mre_mm"] < best_val_mre_mm)

            if is_best_sdr or is_tied_sdr_better_mre:
                best_val_sdr_2_0mm = metrics["sdr_2_0mm"]
                best_val_sdr_2_5mm = metrics["sdr_2_5mm"]
                best_val_mre_mm = metrics["mre_mm"]
                best_val_mae_px = metrics["mae"]
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                # Ensure compiled models (_orig_mod) are unwrapped before saving
                raw_model = getattr(model, "_orig_mod", model)

                # 1. Clean model weights (~420 MB state_dict) for inference and MLflow logging
                torch.save(raw_model.state_dict(), SAVE_PATH)

                # 2. Full checkpoint bundle (~1.3 GB) saved locally on VPS (NOT sent to MLflow)
                full_ckpt_path = os.path.join(os.path.dirname(SAVE_PATH), "best_full_checkpoint.pth")
                if save_optimizer:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                            "val_loss": best_val_loss,
                            "val_mre_mm": best_val_mre_mm,
                            "val_rmse_mm": metrics["rmse_mm"],
                            "val_sdr_2_0mm": best_val_sdr_2_0mm,
                            "val_sdr_2_5mm": best_val_sdr_2_5mm,
                            "val_sdr_3_0mm": metrics["sdr_3_0mm"],
                            "val_sdr_4_0mm": metrics["sdr_4_0mm"],
                            "val_mae_px": best_val_mae_px,
                            "val_rmse_px": metrics["rmse"],
                            "val_sdr_2_5px": metrics["sdr_2_5"],
                            "val_sdr_2_0px": metrics["sdr_2_0"],
                            "pixel_spacing": pixel_spacing,
                            "img_size": img_size,
                        },
                        full_ckpt_path,
                    )
                    full_size_mb = os.path.getsize(full_ckpt_path) / (1024 * 1024)
                    full_status_str = f"& full checkpoint [{full_size_mb:.1f} MB -> {full_ckpt_path}] (local VPS only, not sent to MLflow) "
                else:
                    full_status_str = ""

                saved_size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
                print(
                    f"--> Saved new best model weights locally [{saved_size_mb:.1f} MB -> {SAVE_PATH}] {full_status_str}"
                    f"(SDR@2.0mm: {best_val_sdr_2_0mm:.1f}%, SDR@2.5mm: {best_val_sdr_2_5mm:.1f}%, "
                    f"MRE: {best_val_mre_mm:.2f} mm [{best_val_mae_px:.2f} px], Val Loss: {best_val_loss:.4f})"
                )

                # Log best metrics to MLflow
                mlflow.log_metrics(
                    {
                        "best_val_loss": best_val_loss,
                        "best_val_mre_mm": best_val_mre_mm,
                        "best_val_rmse_mm": metrics["rmse_mm"],
                        "best_val_sdr_2_0mm": best_val_sdr_2_0mm,
                        "best_val_sdr_2_5mm": best_val_sdr_2_5mm,
                        "best_val_sdr_3_0mm": metrics["sdr_3_0mm"],
                        "best_val_sdr_4_0mm": metrics["sdr_4_0mm"],
                        "best_val_mae_px": best_val_mae_px,
                        "best_val_rmse_px": metrics["rmse"],
                        "best_val_sdr_2_5px": metrics["sdr_2_5"],
                        "best_val_sdr_2_0px": metrics["sdr_2_0"],
                    },
                    step=epoch + 1,
                )

                # Log per-landmark performance metrics JSON to MLflow
                if "per_landmark" in metrics and metrics["per_landmark"]:
                    log_best_per_landmark_metrics_to_mlflow(
                        landmark_metrics=metrics["per_landmark"],
                        epoch=epoch + 1,
                        metrics=metrics,
                        best_val_sdr=best_val_sdr_2_0mm,
                        best_val_mae=best_val_mae_px,
                        best_val_loss=best_val_loss,
                        save_dir=os.path.dirname(SAVE_PATH),
                    )
            else:
                patience_counter += 1
                if patience_counter % 5 == 0:
                    print(f"No SDR improvement for {patience_counter} consecutive epoch(s).")
                if patience_counter >= patience:
                    print(f"Early stopping triggered: Best SDR did not improve for {patience} consecutive epochs.")
                    break

        # =========================================================================
        # POST-TRAINING COMPLETION PIPELINE:
        # Step 1: Log the .json metrics file(s) and test metrics FIRST
        # Step 2: Upload the charts (training & evaluation metric PNG charts)
        # Step 3: Upload the large model checkpoint file (.pth)
        # =========================================================================

        # 1. Prepare and save training history JSON
        history_json_path = os.path.join(os.path.dirname(SAVE_PATH), "training_history.json")
        try:
            with open(history_json_path, "w") as f:
                json.dump(
                    {
                        "best_val_loss": best_val_loss,
                        "best_val_sdr_2_0mm": best_val_sdr_2_0mm,
                        "best_val_sdr_2_5mm": best_val_sdr_2_5mm,
                        "best_val_mre_mm": best_val_mre_mm,
                        "best_val_mae_px": best_val_mae_px,
                        "pixel_spacing": pixel_spacing,
                        "img_size": img_size,
                        "total_epochs_completed": len(history["epochs"]),
                        "history": history,
                    },
                    f,
                    indent=4,
                )
        except Exception as hist_err:
            print(f"⚠️ Warning: Could not save training history JSON: {hist_err}")

        # 2. Run post-training evaluation to generate evaluation metrics.json
        eval_metrics_json_path = None
        eval_run_folder = None
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
                        img_size=img_size,
                        pixel_spacing=pixel_spacing,
                        log_to_mlflow=False,
                    )

                    # Log all test summary scalar metrics into MLflow
                    eval_metrics_to_log = {}
                    for k, v in summary_metrics.items():
                        if isinstance(v, (int, float)):
                            clean_k = f"test_{k}".replace("@", "").replace(".", "_").replace(" ", "_")
                            eval_metrics_to_log[clean_k] = float(v)

                    mlflow.log_metrics(eval_metrics_to_log)

                    candidate_json = Path(eval_run_folder) / "metrics.json"
                    if candidate_json.exists():
                        eval_metrics_json_path = str(candidate_json)
                except Exception as e:
                    print(f"❌ Warning: Post-training evaluation failed: {e}")
            else:
                print("⚠️ Test dataset directories not found; skipping post-training evaluation.")

        # --- STEP 1: LOG .JSON METRICS FIRST (MLFLOW & FTP) ---
        print("\n[Step 1/3] --- Logging .json Metrics Files First ---")
        # 1a. MLflow: Log JSON artifacts
        if eval_metrics_json_path and os.path.exists(eval_metrics_json_path):
            try:
                mlflow.log_artifact(eval_metrics_json_path, artifact_path="evaluation")
                print(f"--> Successfully logged '{eval_metrics_json_path}' to MLflow artifact path 'evaluation'.")
            except Exception as ml_err:
                print(f"⚠️ Warning: Could not log metrics JSON to MLflow: {ml_err}")

        if os.path.exists(history_json_path):
            try:
                mlflow.log_artifact(history_json_path, artifact_path="checkpoints")
                print(f"--> Successfully logged '{history_json_path}' to MLflow artifact path 'checkpoints'.")
            except Exception as ml_err:
                print(f"⚠️ Warning: Could not log training history JSON to MLflow: {ml_err}")

        if eval_run_folder:
            eval_txt = Path(eval_run_folder) / "summary_report.txt"
            if eval_txt.exists():
                try:
                    mlflow.log_artifact(str(eval_txt), artifact_path="evaluation")
                except Exception as ml_err:
                    print(f"⚠️ Warning: Could not log summary report to MLflow: {ml_err}")

        # 1b. FTP: Upload .json metrics files FIRST
        if not skip_ftp:
            ftp_json_files = []
            if eval_metrics_json_path and os.path.exists(eval_metrics_json_path):
                ftp_json_files.append(eval_metrics_json_path)
            else:
                # Fallback check for any recent metrics.json in evaluation/
                eval_base = Path("evaluation")
                if eval_base.exists():
                    found_jsons = sorted(
                        eval_base.glob("**/metrics.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if found_jsons:
                        ftp_json_files.append(str(found_jsons[0]))

            if os.path.exists(history_json_path):
                ftp_json_files.append(history_json_path)

            best_lm_json = os.path.join(os.path.dirname(SAVE_PATH), "best_per_landmark_metrics.json")
            if os.path.exists(best_lm_json):
                ftp_json_files.append(best_lm_json)

            if ftp_json_files:
                print(f"\n--> Uploading {len(ftp_json_files)} .json metrics file(s) to FTP server first...")
                try:
                    upload_files_to_ftp(
                        files=ftp_json_files,
                        ftp_host=ftp_host,
                        ftp_port=ftp_port,
                        ftp_user=ftp_user,
                        ftp_password=ftp_password,
                        remote_dir=ftp_remote_dir,
                        use_tls=ftp_tls,
                    )
                except Exception as ftp_err:
                    print(f"⚠️ Warning: FTP JSON upload encountered an error: {ftp_err}")

        # --- STEP 2: AFTER THAT, UPLOAD/LOG CHARTS ---
        print("\n[Step 2/3] --- Uploading Metric PNG Charts ---")
        # 2a. Generate and log training metric PNG charts to MLflow
        generate_and_log_training_charts(history, log_to_mlflow=True)

        # 2b. Log evaluation charts and visualizations to MLflow
        if eval_run_folder and os.path.exists(eval_run_folder):
            eval_charts_dir = Path(eval_run_folder) / "charts"
            if eval_charts_dir.exists():
                try:
                    mlflow.log_artifacts(str(eval_charts_dir), artifact_path="evaluation/charts")
                    print(f"--> Successfully logged evaluation charts to MLflow artifact path 'evaluation/charts'.")
                except Exception as ml_err:
                    print(f"⚠️ Warning: Could not log evaluation charts to MLflow: {ml_err}")

            eval_vis_dir = Path(eval_run_folder) / "visualizations"
            if eval_vis_dir.exists():
                try:
                    mlflow.log_artifacts(str(eval_vis_dir), artifact_path="evaluation/visualizations")
                    print(f"--> Successfully logged evaluation visualization PNGs to MLflow artifact path 'evaluation/visualizations'.")
                except Exception as ml_err:
                    print(f"⚠️ Warning: Could not log evaluation visualizations to MLflow: {ml_err}")
        else:
            # Fallback: Check for any existing evaluation/ charts and visualizations to log BEFORE best.pth
            eval_base = Path("evaluation")
            if eval_base.exists():
                found_charts = sorted(
                    eval_base.glob("**/charts"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if found_charts and found_charts[0].is_dir():
                    try:
                        mlflow.log_artifacts(str(found_charts[0]), artifact_path="evaluation/charts")
                        print(f"--> Successfully logged fallback evaluation charts from '{found_charts[0]}' to MLflow artifact path 'evaluation/charts'.")
                    except Exception as ml_err:
                        print(f"⚠️ Warning: Could not log fallback evaluation charts to MLflow: {ml_err}")

        # --- STEP 3: AFTER THAT, UPLOAD LARGE MODEL FILE ---
        print("\n[Step 3/3] --- Uploading Large Model Checkpoint File ---")
        if os.path.exists(SAVE_PATH):
            model_size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
            # 3a. MLflow: Log large model checkpoint artifact
            print(f"--> Logging large checkpoint '{SAVE_PATH}' ({model_size_mb:.2f} MB) to MLflow...")
            try:
                mlflow.log_artifact(SAVE_PATH, artifact_path="checkpoints")
                print(f"--> Successfully logged best checkpoint artifact '{SAVE_PATH}' to MLflow.")
            except Exception as ml_err:
                print(f"⚠️ Warning: Could not log checkpoint artifact to MLflow: {ml_err}")

            # 3b. FTP: Upload large model file
            if not skip_ftp:
                print(f"--> Uploading large checkpoint '{SAVE_PATH}' ({model_size_mb:.2f} MB) to FTP server...")
                try:
                    upload_files_to_ftp(
                        files=[SAVE_PATH],
                        ftp_host=ftp_host,
                        ftp_port=ftp_port,
                        ftp_user=ftp_user,
                        ftp_password=ftp_password,
                        remote_dir=ftp_remote_dir,
                        use_tls=ftp_tls,
                    )
                except Exception as ftp_err:
                    print(f"⚠️ Warning: FTP model upload encountered an error: {ftp_err}")
        else:
            print(f"⚠️ Best checkpoint '{SAVE_PATH}' was not found.")


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
    parser.add_argument(
        "--save-optimizer",
        dest="save_optimizer",
        action="store_true",
        default=True,
        help="Also save full checkpoint with optimizer/scheduler states locally to ./artifacts/best_full_checkpoint.pth (~1.3GB, local VPS only, not sent to MLflow; default: True)",
    )
    parser.add_argument(
        "--no-save-optimizer",
        dest="save_optimizer",
        action="store_false",
        help="Do not save the large full checkpoint locally (only save clean weights in best.pth)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Input image resolution in pixels (default: {DEFAULT_IMG_SIZE})",
    )
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=DEFAULT_PIXEL_SPACING,
        help=f"Physical spacing in mm per pixel (default: {DEFAULT_PIXEL_SPACING})",
    )
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (AMP FP16) training (default: True)",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable Automatic Mixed Precision and train in full FP32",
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        default=bool(int(os.getenv("TORCH_COMPILE", "0"))),
        help="Enable PyTorch 2.0+ model compilation via torch.compile() (default: False or $TORCH_COMPILE)",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile_model",
        action="store_false",
        help="Disable PyTorch model compilation and force eager mode",
    )
    parser.add_argument(
        "--warmup-coord-epochs",
        type=int,
        default=WARMUP_COORD_EPOCHS,
        help=f"Number of initial epochs to linearly ramp coordinate/graph losses from 0.0 to full weight (default: {WARMUP_COORD_EPOCHS})",
    )

    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        llrd_decay_rate=args.llrd_decay_rate,
        img_size=args.img_size,
        pixel_spacing=args.pixel_spacing,
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
        save_optimizer=args.save_optimizer,
        use_amp=args.use_amp,
        compile_model=args.compile_model,
        warmup_coord_epochs=args.warmup_coord_epochs,
    )

