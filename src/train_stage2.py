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
    extract_subpixel_peaks,
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
            # Use continuous subpixel peak extraction to eliminate SoftArgmax expectation drift
            pred_coords_roi_peaks = extract_subpixel_peaks(pred_heatmaps)
            pred_coords_roi_np = pred_coords_roi_peaks.detach().cpu().numpy()
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
    sigma: float = 2.5,
    margin_ratio: float = 0.18,
    bottom_margin_ratio: Optional[float] = 0.28,
    jitter_trans: float = 0.06,
    jitter_scale: float = 0.08,
    patience: int = 15,
    lambda_hm: float = 50.0,
    lambda_cd: float = 2.0,
    lambda_graph: float = 1.0,
    lambda_morph: float = 0.0,
    stage1_weights: str = "artifacts/stage1_best.pth",
    warmstart_from_stage1: bool = False,
    save_path: str = "artifacts/stage2_best.pth",
    save_optimizer: bool = True,
    train_img_dir: str = "dataset/train/images",
    train_npz_dir: str = "dataset/train/labels",
    val_img_dir: str = "dataset/val/images",
    val_npz_dir: str = "dataset/val/labels",
    test_img_dir: str = "dataset/test/images",
    test_npz_dir: str = "dataset/test/labels",
    skip_eval: bool = False,
    eval_visualizations: int = 10,
    native_img_dir: Optional[str] = "data/raw/images",
    light_predictions_dir: str = "evaluation/predictions_light",
    export_json_path: Optional[str] = "data/exports/export.json",
    experiment_name: Optional[str] = None,
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
    will_warmstart = warmstart_from_stage1 and os.path.exists(stage1_weights)
    if warmstart_from_stage1 and not will_warmstart:
        print(f"⚠️  WARNING: Warmstart requested from '{stage1_weights}', but file does not exist!")
        print("--> Falling back to ImageNet pretrained Swin-Base backbone.")

    model = CephalometricSwinGCN(
        num_landmarks=NUM_LANDMARKS,
        pretrained=not will_warmstart,
        img_size=target_size,
    )

    # Warm-start from Stage 1 if requested and weights exist
    if will_warmstart:
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
        bottom_margin_ratio=bottom_margin_ratio,
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
    landmark_weights = get_landmark_weights(device)

    # Log Balanced Sampling & Landmark Weights Configuration
    has_weighted_sampler = isinstance(getattr(train_loader, "sampler", None), torch.utils.data.WeightedRandomSampler)
    print("\n" + "=" * 80)
    if has_weighted_sampler:
        print("🎯 STAGE 2 CLASS-WEIGHTED SAMPLER: [ENABLED & ACTIVE ✅]")
        print(f"   • Sampler: torch.utils.data.WeightedRandomSampler (replacement=True)")
        print(f"   • Balancing Power: {balanced_sampling_power} (smoothed inverse frequency)")
        print(f"   • Samples drawn per epoch: {len(train_loader.sampler)} from {len(train_loader.dataset)} unique images")
    else:
        print("⚠️ STAGE 2 CLASS-WEIGHTED SAMPLER: [DISABLED / UNIFORM SHUFFLE ⚠️]")
    print("-" * 80)
    print("⚖️  WEIGHTED LANDMARK LOSS IS ACTIVE [ENABLED & APPLIED ✅]")
    print("   • Applied to: WingLoss (Coordinates) & AdaptiveWingLoss (Heatmaps)")
    print("   • Multiplier Values per Landmark:")
    for idx, (name, w) in enumerate(zip(LANDMARK_CLASSES, landmark_weights)):
        val = w.item()
        tier_tag = f"🔥 BOOSTED ({val:.1f}x multiplier)" if val > 1.0 else f"   baseline ({val:.1f}x multiplier)"
        print(f"     [{idx:2d}] {name:8s} -> {val:4.1f}x | {tier_tag}")
    print("=" * 80 + "\n")

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
    full_ckpt_path = os.path.join(
        os.path.dirname(save_path),
        f"{Path(save_path).stem}_full_checkpoint.pth",
    )

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
            "device": str(device),
            "use_amp": use_amp,
            "save_optimizer": save_optimizer,
            "optimizer": "AdamW-LLRD",
            "scheduler": "CosineAnnealingWithWarmup",
            "num_landmarks": NUM_LANDMARKS,
            "train_samples": len(train_loader.dataset) if hasattr(train_loader, "dataset") else len(train_loader),
            "val_samples": len(val_loader.dataset) if hasattr(val_loader, "dataset") else len(val_loader),
            "use_balanced_sampling": use_balanced_sampling,
            "balanced_sampling_power": balanced_sampling_power,
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

                raw_model = getattr(model, "_orig_mod", model)
                raw_model = getattr(raw_model, "module", raw_model)

                # 1. Clean model weights (~420 MB state_dict) for inference and MLflow logging
                torch.save(raw_model.state_dict(), save_path)

                # 2. Full checkpoint bundle (~1.3 GB) saved locally on instance (NOT sent to MLflow)
                if save_optimizer:
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                            "metrics": val_metrics,
                            "target_size": target_size,
                            "sigma": sigma,
                            "margin_ratio": margin_ratio,
                        },
                        full_ckpt_path,
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

            epoch_metrics_to_log = {
                "stage2_train_loss": train_loss,
                "stage2_train_loss_hm": train_loss_dict.get("heatmap", 0.0),
                "stage2_train_loss_cd": train_loss_dict.get("coord", 0.0),
                "stage2_train_loss_graph": train_loss_dict.get("graph", 0.0),
                "stage2_train_loss_morph": train_loss_dict.get("morph", 0.0),
                "stage2_val_loss": val_loss,
                "stage2_val_loss_hm": val_metrics.get("loss_hm", 0.0),
                "stage2_val_loss_cd": val_metrics.get("loss_cd", 0.0),
                "stage2_val_loss_graph": val_metrics.get("loss_graph", 0.0),
                "stage2_val_loss_morph": val_metrics.get("loss_morph", 0.0),
                "stage2_val_mean_norm_err": mean_norm,
                "stage2_val_median_norm_err": med_norm,
                "stage2_val_rmse_norm_err": val_metrics.get("rmse_norm_err", 0.0),
                "stage2_val_share_2_5pct": val_metrics.get("share_within_2_5pct", 0.0),
                "stage2_val_share_5pct": share_5,
                "stage2_val_share_7_5pct": val_metrics.get("share_within_7_5pct", 0.0),
                "stage2_val_share_10pct": val_metrics.get("share_within_10pct", 0.0),
                "stage2_val_mae_px": val_metrics.get("mae", 0.0),
                "stage2_val_rmse_px": val_metrics.get("rmse", 0.0),
                "stage2_val_sdr_2_0px": val_metrics.get("sdr_2_0", 0.0),
                "stage2_val_sdr_2_5px": val_metrics.get("sdr_2_5", 0.0),
                "stage2_val_sdr_3_0px": val_metrics.get("sdr_3_0", 0.0),
                "stage2_val_sdr_4_0px": val_metrics.get("sdr_4_0", 0.0),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time_seconds": epoch_time,
                "best_val_share_5pct": best_val_share_5pct,
                "best_val_mean_norm": best_val_mean_norm,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
            }
            # Log per-landmark validation normalized errors
            per_lm = val_metrics.get("per_landmark_metrics") or []
            if isinstance(per_lm, list):
                for item in per_lm:
                    if isinstance(item, dict):
                        lm_name = item.get("name")
                        if lm_name:
                            err_val = item.get("mean_norm_err", item.get("norm_err_mean"))
                            if err_val is not None:
                                epoch_metrics_to_log[f"val_err_{lm_name}"] = float(err_val)
                            if "share_within_5pct" in item:
                                epoch_metrics_to_log[f"val_share_5pct_{lm_name}"] = float(item["share_within_5pct"])
            elif isinstance(per_lm, dict):
                for lm_name, lm_data in per_lm.items():
                    if isinstance(lm_data, dict):
                        err_val = lm_data.get("mean_norm_err", lm_data.get("norm_err_mean"))
                        if err_val is not None:
                            epoch_metrics_to_log[f"val_err_{lm_name}"] = float(err_val)
                        if "share_within_5pct" in lm_data:
                            epoch_metrics_to_log[f"val_share_5pct_{lm_name}"] = float(lm_data["share_within_5pct"])

            mlflow.log_metrics(epoch_metrics_to_log, step=epoch)

            if patience_counter >= patience:
                print(f"Early stopping triggered: No validation improvement for {patience} consecutive epochs.")
                break

        saved_size_mb = os.path.getsize(save_path) / (1024 * 1024) if os.path.exists(save_path) else 0.0
        print(f"\n✅ Stage 2 Training Finished! Clean weights saved to '{save_path}' [{saved_size_mb:.1f} MB] at epoch {best_epoch}")
        if save_optimizer and os.path.exists(full_ckpt_path):
            full_size_mb = os.path.getsize(full_ckpt_path) / (1024 * 1024)
            print(f"   Full checkpoint [{full_size_mb:.1f} MB] saved locally to '{full_ckpt_path}' (local instance only, not sent to MLflow)")
        print(f"   Best Val Share ≤5%: {best_val_share_5pct:.2f}% | Best Mean Norm Err: {best_val_mean_norm:.2f}%")

        # Save training history JSON
        history_path = Path(save_path).parent / "stage2_training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact(str(history_path))
        if os.path.exists(save_path):
            mlflow.log_artifact(save_path)

        # --- Comprehensive Post-Training Evaluation Across Test & Validation Cohorts (GT-ROI Oracle) ---
        if not skip_eval and os.path.exists(save_path):
            print("\n" + "=" * 80)
            print("🚀 RUNNING POST-TRAINING GT-ROI EVALUATION (TEST & VAL COHORTS)")
            print("=" * 80)

            from src.eval_stage2_roi import run_stage2_roi_evaluation

            # 1. TEST COHORT EVALUATION (Stage 2 GT-ROI)
            if os.path.exists(test_img_dir) and os.path.exists(test_npz_dir):
                print(f"\n--> [1/2] Running Stage 2 GT-ROI Evaluation on TEST cohort ({test_img_dir})...")
                try:
                    test_payload, test_dir = run_stage2_roi_evaluation(
                        stage2_weights=save_path,
                        img_dir=test_img_dir,
                        npz_dir=test_npz_dir,
                        output_dir="evaluation/stage2_roi",
                        split_name="test",
                        stage2_img_size=target_size,
                        margin_ratio=margin_ratio,
                        bottom_margin_ratio=bottom_margin_ratio,
                        native_img_dir=native_img_dir,
                        num_visualizations=eval_visualizations,
                        device_str=str(device),
                        log_to_mlflow=False,
                    )
                    test_p = Path(test_dir)
                    for rep_file in ["metrics.json", "summary_report.txt", "predictions_landmarks.csv"]:
                        if (test_p / rep_file).exists():
                            mlflow.log_artifact(str(test_p / rep_file), artifact_path="evaluation/test")
                    if (test_p / "visualizations").exists():
                        mlflow.log_artifacts(str(test_p / "visualizations"), artifact_path="evaluation/test/visualizations")

                    gen_c = test_payload.get("general", {})
                    for k, v in gen_c.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"test_{k}", float(v))
                    print(f"✅ Test cohort evaluated & logged to MLflow (Mean Norm: {gen_c.get('mean', 0.0):.2f}%, ≤5%: {gen_c.get('share_within_5pct', 0.0):.2f}%)")
                except Exception as t_err:
                    print(f"⚠️ Warning: Test cohort evaluation failed: {t_err}")

            # 2. VALIDATION COHORT EVALUATION (Stage 2 GT-ROI)
            if os.path.exists(val_img_dir) and os.path.exists(val_npz_dir):
                print(f"\n--> [2/2] Running Stage 2 GT-ROI Evaluation on VAL cohort ({val_img_dir})...")
                try:
                    val_payload, val_dir = run_stage2_roi_evaluation(
                        stage2_weights=save_path,
                        img_dir=val_img_dir,
                        npz_dir=val_npz_dir,
                        output_dir="evaluation/stage2_roi",
                        split_name="val",
                        stage2_img_size=target_size,
                        margin_ratio=margin_ratio,
                        bottom_margin_ratio=bottom_margin_ratio,
                        native_img_dir=native_img_dir,
                        num_visualizations=5,
                        device_str=str(device),
                        log_to_mlflow=False,
                    )
                    val_p = Path(val_dir)
                    for rep_file in ["metrics_val.json", "summary_report_val.txt", "predictions_landmarks_val.csv"]:
                        if (val_p / rep_file).exists():
                            mlflow.log_artifact(str(val_p / rep_file), artifact_path="evaluation/val")
                    if (val_p / "visualizations").exists():
                        mlflow.log_artifacts(str(val_p / "visualizations"), artifact_path="evaluation/val/visualizations")

                    gen_v = val_payload.get("general", {})
                    for k, v in gen_v.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"val_{k}", float(v))
                    print(f"✅ Validation cohort evaluated & logged to MLflow (Mean Norm: {gen_v.get('mean', 0.0):.2f}%, ≤5%: {gen_v.get('share_within_5pct', 0.0):.2f}%)")
                except Exception as v_err:
                    print(f"⚠️ Warning: Validation cohort evaluation failed: {v_err}")

    return {
        "best_epoch": best_epoch,
        "best_val_share_5pct": best_val_share_5pct,
        "best_val_mean_norm": best_val_mean_norm,
        "save_path": save_path,
        "full_checkpoint_path": full_ckpt_path if save_optimizer else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2 (Fine Vertebrae Model) Training CLI")
    parser.add_argument("--epochs", "-e", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--target-size", type=int, default=448, choices=[448, 672], help="ROI crop resolution (448 or 672)")
    parser.add_argument("--sigma", type=float, default=2.5, help="Heatmap Gaussian sigma (default: 2.5 for sub-millimeter precision)")
    parser.add_argument("--margin-ratio", type=float, default=0.18, help="ROI margin ratio")
    parser.add_argument("--bottom-margin-ratio", type=float, default=0.28, help="Bottom margin ratio for C4 vertebra (default: 0.28)")
    parser.add_argument("--stage1-weights", type=str, default="artifacts/stage1_best.pth", help="Stage 1 weights for warm-start")
    parser.add_argument("--warmstart", dest="warmstart", action="store_true", help="Warmstart Stage 2 from Stage 1 weights")
    parser.add_argument("--no-warmstart", dest="warmstart", action="store_false", help="Train Stage 2 directly from ImageNet pretrained backbone")
    parser.set_defaults(warmstart=False)
    parser.add_argument("--save-path", type=str, default="artifacts/stage2_best.pth", help="Checkpoint save destination")
    parser.add_argument(
        "--save-optimizer",
        dest="save_optimizer",
        action="store_true",
        default=True,
        help="Also save full checkpoint with optimizer/scheduler states locally to {save_path_stem}_full_checkpoint.pth (~1.3GB, local instance only, not sent to MLflow; default: True)",
    )
    parser.add_argument(
        "--no-save-optimizer",
        dest="save_optimizer",
        action="store_false",
        help="Do not save the large full checkpoint locally (only save clean weights in save_path)",
    )
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--test-img-dir", type=str, default="dataset/test/images", help="Test cohort images dir")
    parser.add_argument("--test-npz-dir", type=str, default="dataset/test/labels", help="Test cohort labels dir")
    parser.add_argument("--native-img-dir", type=str, default="data/raw/images", help="Native resolution raw images dir")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training evaluation across cohorts")
    parser.add_argument("--eval-visualizations", type=int, default=10, help="Number of test visualization panels to save")
    parser.add_argument("--light-predictions-dir", type=str, default="evaluation/predictions_light", help="Output directory for predictions_full")
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
        bottom_margin_ratio=args.bottom_margin_ratio,
        stage1_weights=args.stage1_weights,
        warmstart_from_stage1=args.warmstart,
        save_path=args.save_path,
        save_optimizer=args.save_optimizer,
        patience=args.patience,
        num_workers=args.num_workers,
        test_img_dir=args.test_img_dir,
        test_npz_dir=args.test_npz_dir,
        native_img_dir=args.native_img_dir,
        skip_eval=args.skip_eval,
        eval_visualizations=args.eval_visualizations,
        light_predictions_dir=args.light_predictions_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_username=args.mlflow_username,
        tracking_password=args.mlflow_password,
    )


if __name__ == "__main__":
    main()
