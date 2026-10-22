#!/usr/bin/env python3
"""
Stage 2 (Fine Vertebrae Model) Training Engine for CVM Landmark Detection.

Trains the high-resolution CephalometricSwinGCN network on strictly square, isotropic,
padded ROI crops of the cervical spine (C2-C4) with empirical bounding-box jitter,
adaptive wing loss, coordinate wing loss, anatomical graph loss, and midline-axis shape loss.
Logs standalone and reprojected normalized metrics to MLflow and local history files.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow

from src.data.constants import (
    LANDMARK_CLASSES,
    NUM_LANDMARKS,
)

C3_PI_IDX = LANDMARK_CLASSES.index("C3_PI")
C3_AI_IDX = LANDMARK_CLASSES.index("C3_AI")
DEFAULT_IMG_SIZE = 640
from src.data.loaders.roi_dataset import get_roi_dataloaders
from src.models.losses import (
    AdaptiveWingLoss,
    AnatomicalGraphLoss,
    MorphometricShapeLoss,
    WingLoss,
)
from src.models.model import CephalometricSwinGCN
from src.inference import resolve_device
from src.train import (
    compute_per_landmark_metrics,
    compute_validation_summary_metrics,
    get_cosine_schedule_with_warmup,
    get_landmark_weights,
    get_llrd_param_groups,
)
from src.utils.metrics import (
    compute_reference_length,
    compute_normalized_errors,
)
from src.utils.roi_utils import (
    ROIBoundingBox,
    inverse_transform_coords,
    validate_stage1_box,
)


def validate_stage2_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    awl_loss: Optional[nn.Module] = None,
    wing_loss: Optional[nn.Module] = None,
    graph_loss: Optional[nn.Module] = None,
    morph_loss: Optional[nn.Module] = None,
    landmark_weights: Optional[torch.Tensor] = None,
    lambda_hm: float = 50.0,
    lambda_cd: float = 2.0,
    lambda_graph: float = 0.0,
    lambda_morph: float = 0.0,
    device: Optional[torch.device] = None,
    target_size: int = 448,
    canvas_size: int = 640,
    use_amp: bool = True,
    epoch: int = 1,
    epochs: int = 60,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates Stage 2 model on the validation split:
    - Computes ROI-space losses.
    - Reprojects predicted coordinates back to full canvas space using exact half-pixel alignment.
    - Evaluates standardized normalized landmark error (% of GT C3 reference length).
    """
    model.eval()
    epoch_loss = 0.0
    epoch_loss_hm = 0.0
    epoch_loss_cd = 0.0
    epoch_loss_g = 0.0
    epoch_loss_morph = 0.0

    all_canvas_preds = []
    all_canvas_gts = []

    device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    amp_enabled = use_amp and device_type == "cuda"

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}] Val (Stage 2 ROI)", leave=False)
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            gt_coords_roi = batch["coords"].to(device)
            gt_heatmaps = batch["heatmaps"].to(device)

            u1_list = batch["bbox_u1"].cpu().numpy()
            v1_list = batch["bbox_v1"].cpu().numpy()
            D_list = batch["bbox_D"].cpu().numpy()

            with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
                pred_heatmaps, pred_coords_roi = model(images)

                loss_cd = wing_loss(pred_coords_roi.float(), gt_coords_roi.float(), landmark_weights=landmark_weights)
                total_loss = lambda_cd * loss_cd

                loss_g = None
                if graph_loss is not None and lambda_graph > 0.0:
                    loss_g = graph_loss(pred_coords_roi.float(), gt_coords_roi.float())
                    total_loss = total_loss + (lambda_graph * loss_g)

                loss_morph = None
                if morph_loss is not None and lambda_morph > 0.0:
                    loss_morph = morph_loss(pred_coords_roi.float(), gt_coords_roi.float())
                    total_loss = total_loss + (lambda_morph * loss_morph)

                loss_hm = None
                if awl_loss is not None and lambda_hm > 0.0:
                    loss_hm = awl_loss(pred_heatmaps.float(), gt_heatmaps.float(), landmark_weights=landmark_weights)
                    total_loss = total_loss + (lambda_hm * loss_hm)

            epoch_loss += total_loss.item()
            epoch_loss_cd += (lambda_cd * loss_cd).item()
            if loss_g is not None:
                epoch_loss_g += (lambda_graph * loss_g).item()
            if loss_morph is not None:
                epoch_loss_morph += (lambda_morph * loss_morph).item()
            if loss_hm is not None:
                epoch_loss_hm += (lambda_hm * loss_hm).item()

            # Reproject coordinates back to canvas space [0, 1] for standardized evaluation
            pred_coords_roi_np = pred_coords_roi.detach().cpu().numpy()
            gt_coords_roi_np = gt_coords_roi.detach().cpu().numpy()
            B = images.shape[0]

            for b in range(B):
                bbox_sample = ROIBoundingBox(
                    u1=int(round(u1_list[b])),
                    v1=int(round(v1_list[b])),
                    u2=int(round(u1_list[b] + D_list[b])),
                    v2=int(round(v1_list[b] + D_list[b])),
                    D=int(round(D_list[b])),
                    center_u=u1_list[b] + D_list[b] / 2.0,
                    center_v=v1_list[b] + D_list[b] / 2.0,
                    span=D_list[b] / 1.36,
                    img_w=canvas_size,
                    img_h=canvas_size,
                )

                reprojected_pred = inverse_transform_coords(
                    roi_coords=pred_coords_roi_np[b],
                    bbox=bbox_sample,
                    target_size=target_size,
                    output_normalized=True,
                )
                reprojected_gt = inverse_transform_coords(
                    roi_coords=gt_coords_roi_np[b],
                    bbox=bbox_sample,
                    target_size=target_size,
                    output_normalized=True,
                )

                all_canvas_preds.append(reprojected_pred)
                all_canvas_gts.append(reprojected_gt)

    n_batches = max(1, len(dataloader))
    val_loss = epoch_loss / n_batches

    all_canvas_preds_t = torch.tensor(np.array(all_canvas_preds), dtype=torch.float32)
    all_canvas_gts_t = torch.tensor(np.array(all_canvas_gts), dtype=torch.float32)

    norm_summary = compute_validation_summary_metrics(
        all_canvas_preds_t, all_canvas_gts_t, img_size=canvas_size
    )

    per_lm_metrics = compute_per_landmark_metrics(
        all_canvas_preds_t, all_canvas_gts_t, img_size=canvas_size
    )

    # Pixel errors on canvas
    pred_canvas_px = all_canvas_preds_t * float(canvas_size)
    gt_canvas_px = all_canvas_gts_t * float(canvas_size)
    diff = torch.sqrt(torch.sum((pred_canvas_px - gt_canvas_px) ** 2, dim=-1))
    valid_mask = (all_canvas_gts_t[:, :, 0] >= 0) & (all_canvas_gts_t[:, :, 1] >= 0)

    if valid_mask.any():
        val_errors = diff[valid_mask]
        val_mae = val_errors.mean().item()
        val_rmse = torch.sqrt((val_errors ** 2).mean()).item()
        sdr_2_0 = (val_errors <= 2.0).float().mean().item() * 100.0
        sdr_2_5 = (val_errors <= 2.5).float().mean().item() * 100.0
        sdr_3_0 = (val_errors <= 3.0).float().mean().item() * 100.0
        sdr_4_0 = (val_errors <= 4.0).float().mean().item() * 100.0
    else:
        val_mae, val_rmse, sdr_2_0, sdr_2_5, sdr_3_0, sdr_4_0 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    metrics = {
        "val_loss": val_loss,
        "loss_hm": epoch_loss_hm / n_batches,
        "loss_cd": epoch_loss_cd / n_batches,
        "loss_graph": epoch_loss_g / n_batches,
        "loss_morph": epoch_loss_morph / n_batches,
        "mean_norm_err": norm_summary["mean_norm_err"],
        "median_norm_err": norm_summary["median_norm_err"],
        "share_within_2_5pct": norm_summary["share_within_2_5pct"],
        "share_within_5pct": norm_summary["share_within_5pct"],
        "share_within_7_5pct": norm_summary["share_within_7_5pct"],
        "share_within_10pct": norm_summary["share_within_10pct"],
        "rmse_norm_err": norm_summary["rmse_norm_err"],
        "mae": val_mae,
        "rmse": val_rmse,
        "sdr_2_0": sdr_2_0,
        "sdr_2_5": sdr_2_5,
        "sdr_3_0": sdr_3_0,
        "sdr_4_0": sdr_4_0,
        "per_landmark_metrics": per_lm_metrics,
    }

    return val_loss, metrics


def train_stage2_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    awl_loss: Optional[nn.Module] = None,
    wing_loss: Optional[nn.Module] = None,
    graph_loss: Optional[nn.Module] = None,
    morph_loss: Optional[nn.Module] = None,
    landmark_weights: Optional[torch.Tensor] = None,
    lambda_hm: float = 50.0,
    lambda_cd: float = 2.0,
    lambda_graph: float = 1.0,
    lambda_morph: float = 0.0,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = True,
    epoch: int = 1,
    epochs: int = 50,
) -> Tuple[float, Dict[str, float]]:
    """Executes one training epoch for Stage 2."""
    model.train()
    epoch_loss = 0.0
    epoch_loss_hm = 0.0
    epoch_loss_cd = 0.0
    epoch_loss_g = 0.0
    epoch_loss_morph = 0.0

    device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    amp_enabled = use_amp and device_type == "cuda"

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}] Train (Stage 2 ROI)", leave=False)

    for step, batch in enumerate(pbar, 1):
        images = batch["image"].to(device)
        gt_coords = batch["coords"].to(device)
        gt_heatmaps = batch["heatmaps"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
            pred_heatmaps, pred_coords = model(images)

            loss_cd = wing_loss(pred_coords.float(), gt_coords.float(), landmark_weights=landmark_weights)
            total_loss = lambda_cd * loss_cd

            loss_g = None
            if graph_loss is not None and lambda_graph > 0.0:
                loss_g = graph_loss(pred_coords.float(), gt_coords.float())
                total_loss = total_loss + (lambda_graph * loss_g)

            loss_morph = None
            if morph_loss is not None and lambda_morph > 0.0:
                loss_morph = morph_loss(pred_coords.float(), gt_coords.float())
                total_loss = total_loss + (lambda_morph * loss_morph)

            loss_hm = None
            if awl_loss is not None and lambda_hm > 0.0:
                loss_hm = awl_loss(pred_heatmaps.float(), gt_heatmaps.float(), landmark_weights=landmark_weights)
                total_loss = total_loss + (lambda_hm * loss_hm)

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
        epoch_loss_cd += (lambda_cd * loss_cd).item()
        if loss_g is not None:
            epoch_loss_g += (lambda_graph * loss_g).item()
        if loss_morph is not None:
            epoch_loss_morph += (lambda_morph * loss_morph).item()
        if loss_hm is not None:
            epoch_loss_hm += (lambda_hm * loss_hm).item()

        pbar.set_postfix(loss=f"{epoch_loss / step:.4f}")

    n_batches = max(1, len(dataloader))
    loss_dict = {
        "heatmap": epoch_loss_hm / n_batches,
        "coord": epoch_loss_cd / n_batches,
        "graph": epoch_loss_g / n_batches,
        "morph": epoch_loss_morph / n_batches,
    }

    return epoch_loss / n_batches, loss_dict


def train_stage2_main(
    epochs: int = 60,
    batch_size: int = 8,
    lr: float = 1e-4,
    llrd_decay_rate: float = 0.8,
    target_size: int = 448,
    sigma: float = 4.0,
    margin_ratio: float = 0.18,
    jitter_trans: float = 0.06,
    jitter_scale: float = 0.08,
    patience: int = 15,
    lambda_hm: float = 50.0,
    lambda_cd: float = 2.0,
    lambda_graph: float = 1.0,
    lambda_morph: float = 0.0,
    stage1_weights: str = "artifacts/stage1_best.pth",
    warmstart_from_stage1: bool = True,
    save_path: str = "artifacts/stage2_best.pth",
    train_img_dir: str = "dataset/train/images",
    train_npz_dir: str = "dataset/train/labels",
    val_img_dir: str = "dataset/val/images",
    val_npz_dir: str = "dataset/val/labels",
    native_img_dir: Optional[str] = "data/raw/images",
    export_json_path: Optional[str] = "data/exports/export.json",
    experiment_name: Optional[str] = "cvm-stage2-fine-roi",
    tracking_uri: Optional[str] = None,
    tracking_username: Optional[str] = None,
    tracking_password: Optional[str] = None,
    run_name: Optional[str] = None,
    use_amp: bool = True,
    use_balanced_sampling: bool = True,
    balanced_sampling_power: float = 0.5,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Main training routine for Stage 2 (Fine Model)."""
    load_dotenv()
    device = resolve_device("auto")
    print("=" * 80)
    print(f"STARTING STAGE 2 (FINE MODEL) TRAINING | Device: {device} | Resolution: {target_size}x{target_size}")
    print("=" * 80)

    # 1. Initialize Model
    model = CephalometricSwinGCN(
        num_landmarks=NUM_LANDMARKS,
        pretrained=not warmstart_from_stage1,
        img_size=target_size,
    )

    # Warm-start from Stage 1 if requested and weights exist
    if warmstart_from_stage1 and os.path.exists(stage1_weights):
        print(f"--> Warm-starting Stage 2 from Stage 1 checkpoint: '{stage1_weights}'")
        checkpoint = torch.load(stage1_weights, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("stage1."):
                k = k[len("stage1."):]
            cleaned[k] = v
        # Load with strict=False to accommodate resolution difference in heads
        incompatible = model.load_state_dict(cleaned, strict=False)
        print(f"    Loaded weights (missing: {len(incompatible.missing_keys)}, unexpected: {len(incompatible.unexpected_keys)})")
    else:
        print("--> Initialized Stage 2 with ImageNet pretrained Swin-Base backbone.")

    model = model.to(device)

    # 2. Build Dataloaders
    jitter_config = {"trans": jitter_trans, "scale": jitter_scale}
    train_loader, val_loader = get_roi_dataloaders(
        train_img_dir=train_img_dir,
        train_npz_dir=train_npz_dir,
        val_img_dir=val_img_dir,
        val_npz_dir=val_npz_dir,
        batch_size=batch_size,
        target_size=target_size,
        sigma=sigma,
        margin_ratio=margin_ratio,
        jitter=jitter_config,
        num_workers=num_workers,
        use_balanced_sampling=use_balanced_sampling,
        sampling_power=balanced_sampling_power,
        export_json_path=export_json_path,
        native_img_dir=native_img_dir if (native_img_dir and os.path.exists(native_img_dir)) else None,
    )

    # 3. Setup Optimizer, Scheduler, Losses
    param_groups = get_llrd_param_groups(model, base_lr=lr, weight_decay=1e-4, decay_rate=llrd_decay_rate)
    optimizer = optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=5, total_epochs=epochs, min_lr_ratio=0.01)

    awl_loss = AdaptiveWingLoss().to(device) if lambda_hm > 0.0 else None
    wing_loss = WingLoss(img_size=float(target_size)).to(device)
    graph_loss = AnatomicalGraphLoss(model.adj_matrix).to(device) if lambda_graph > 0.0 else None
    morph_loss = MorphometricShapeLoss().to(device) if lambda_morph > 0.0 else None
    landmark_weights = get_landmark_weights(device)  # Uniform 1.0 weights

    # 4. MLflow Tracking Setup
    user = tracking_username or os.getenv("MLFLOW_TRACKING_USERNAME")
    pwd = tracking_password or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if user:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
    if pwd:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or "cvm-stage2-fine-roi"
    mlflow.set_experiment(exp_name)

    actual_run_name = run_name or f"stage2_fine_{target_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_share_5pct = 0.0
    best_val_loss = float("inf")
    best_val_mean_norm = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_share_within_5pct": [],
        "val_share_within_2_5pct": [],
        "val_mean_norm_err": [],
        "val_median_norm_err": [],
        "val_mae_px": [],
    }

    with mlflow.start_run(run_name=actual_run_name) as run:
        mlflow.log_params({
            "stage": 2,
            "target_size": target_size,
            "sigma": sigma,
            "margin_ratio": margin_ratio,
            "jitter_trans": jitter_trans,
            "jitter_scale": jitter_scale,
            "warmstart_from_stage1": warmstart_from_stage1,
            "stage1_weights": stage1_weights,
            "epochs": epochs,
            "batch_size": batch_size,
            "base_lr": lr,
            "llrd_decay_rate": llrd_decay_rate,
            "lambda_hm": lambda_hm,
            "lambda_cd": lambda_cd,
            "lambda_graph": lambda_graph,
            "lambda_morph": lambda_morph,
            "patience": patience,
        })

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_loss_dict = train_stage2_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                awl_loss=awl_loss,
                wing_loss=wing_loss,
                graph_loss=graph_loss,
                morph_loss=morph_loss,
                landmark_weights=landmark_weights,
                lambda_hm=lambda_hm,
                lambda_cd=lambda_cd,
                lambda_graph=lambda_graph,
                lambda_morph=lambda_morph,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                epoch=epoch,
                epochs=epochs,
            )

            val_loss, val_metrics = validate_stage2_epoch(
                model=model,
                dataloader=val_loader,
                awl_loss=awl_loss,
                wing_loss=wing_loss,
                graph_loss=graph_loss,
                morph_loss=morph_loss,
                landmark_weights=landmark_weights,
                lambda_hm=lambda_hm,
                lambda_cd=lambda_cd,
                lambda_graph=lambda_graph,
                lambda_morph=lambda_morph,
                device=device,
                target_size=target_size,
                use_amp=use_amp,
                epoch=epoch,
                epochs=epochs,
            )

            scheduler.step()
            epoch_time = time.time() - t0

            share_5 = val_metrics["share_within_5pct"]
            mean_norm = val_metrics["mean_norm_err"]
            med_norm = val_metrics["median_norm_err"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_share_within_5pct"].append(share_5)
            history["val_share_within_2_5pct"].append(val_metrics["share_within_2_5pct"])
            history["val_mean_norm_err"].append(mean_norm)
            history["val_median_norm_err"].append(med_norm)
            history["val_mae_px"].append(val_metrics["mae"])

            # Checkpoint criteria: Primary is share <= 5%, secondary is mean normalized error
            is_best = (share_5 > best_val_share_5pct) or (
                math.isclose(share_5, best_val_share_5pct, abs_tol=0.2) and mean_norm < best_val_mean_norm
            )

            marker = " "
            if is_best:
                best_val_share_5pct = share_5
                best_val_mean_norm = mean_norm
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                marker = "⭐ [BEST]"

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": val_metrics,
                        "target_size": target_size,
                        "sigma": sigma,
                        "margin_ratio": margin_ratio,
                    },
                    save_path,
                )
            else:
                patience_counter += 1

            print(
                f"Epoch [{epoch:02d}/{epochs}] ({epoch_time:.1f}s) | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Norm Mean: {mean_norm:5.2f}% | Med: {med_norm:5.2f}% | "
                f"≤5%: {share_5:5.1f}% | ≤2.5%: {val_metrics['share_within_2_5pct']:5.1f}% | "
                f"MAE: {val_metrics['mae']:.2f} px {marker}"
            )

            mlflow.log_metrics({
                "stage2_train_loss": train_loss,
                "stage2_val_loss": val_loss,
                "stage2_val_mean_norm_err": mean_norm,
                "stage2_val_median_norm_err": med_norm,
                "stage2_val_share_5pct": share_5,
                "stage2_val_share_2_5pct": val_metrics["share_within_2_5pct"],
                "stage2_val_mae_px": val_metrics["mae"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            if patience_counter >= patience:
                print(f"Early stopping triggered: No validation improvement for {patience} consecutive epochs.")
                break

        print(f"\n✅ Stage 2 Training Finished! Best checkpoint saved to '{save_path}' at epoch {best_epoch}")
        print(f"   Best Val Share ≤5%: {best_val_share_5pct:.2f}% | Best Mean Norm Err: {best_val_mean_norm:.2f}%")

        # Save training history JSON
        history_path = Path(save_path).parent / "stage2_training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact(str(history_path))
        if os.path.exists(save_path):
            mlflow.log_artifact(save_path)

    return {
        "best_epoch": best_epoch,
        "best_val_share_5pct": best_val_share_5pct,
        "best_val_mean_norm": best_val_mean_norm,
        "save_path": save_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2 (Fine Vertebrae Model) Training CLI")
    parser.add_argument("--epochs", "-e", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--target-size", type=int, default=448, choices=[448, 672], help="ROI crop resolution (448 or 672)")
    parser.add_argument("--sigma", type=float, default=4.0, help="Heatmap Gaussian sigma")
    parser.add_argument("--margin-ratio", type=float, default=0.18, help="ROI margin ratio")
    parser.add_argument("--stage1-weights", type=str, default="artifacts/stage1_best.pth", help="Stage 1 weights for warm-start")
    parser.add_argument("--no-warmstart", action="store_true", help="Do not warmstart from Stage 1; use ImageNet pretrained")
    parser.add_argument("--save-path", type=str, default="artifacts/stage2_best.pth", help="Checkpoint save destination")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--mlflow-username", type=str, default=None, help="MLflow username")
    parser.add_argument("--mlflow-password", type=str, default=None, help="MLflow password")

    args = parser.parse_args()

    train_stage2_main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        target_size=args.target_size,
        sigma=args.sigma,
        margin_ratio=args.margin_ratio,
        stage1_weights=args.stage1_weights,
        warmstart_from_stage1=not args.no_warmstart,
        save_path=args.save_path,
        patience=args.patience,
        num_workers=args.num_workers,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_username=args.mlflow_username,
        tracking_password=args.mlflow_password,
    )


if __name__ == "__main__":
    main()
