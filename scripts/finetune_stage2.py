#!/usr/bin/env python3
"""
Targeted Fine-Tuning Script for Stage 2 (Fine Vertebrae Model).

Fixes the central concavity bias by training with:
1. Asymmetric Endplate Concavity Loss (penalizing tangential ratio along the inferior endplate)
2. Decoupled / zeroed symmetrical graph loss (lambda_graph = 0.0)
3. Sharpened Gaussian heatmap targets (sigma = 2.0)
4. Warm-start directly from a downloaded or local Stage 2 weights file
5. Automated evaluation and MLflow logging of new checkpoints
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.constants import LANDMARK_CLASSES, NUM_LANDMARKS
from src.data.loaders.roi_dataset import get_roi_dataloaders
from src.models.losses import (
    AdaptiveWingLoss,
    WingLoss,
    AsymmetricEndplateConcavityLoss,
)
from src.models.model import CephalometricSwinGCN
from src.train_stage2 import (
    get_landmark_weights,
    get_llrd_param_groups,
    get_cosine_schedule_with_warmup,
    validate_stage2_epoch,
)
from src.train import resolve_device
from scripts.download_checkpoint import download_checkpoint_from_mlflow


def finetune_stage2(
    stage1_weights: str,
    stage2_weights: str,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 3e-5,
    sigma: float = 2.0,
    target_size: int = 448,
    margin_ratio: float = 0.22,
    bottom_margin_ratio: float = 0.28,
    lambda_hm: float = 50.0,
    lambda_cd: float = 2.0,
    lambda_endplate: float = 5.0,
    save_path: str = "artifacts/stage2_finetuned_best.pth",
    train_img_dir: str = "dataset/train/images",
    train_npz_dir: str = "dataset/train/labels",
    val_img_dir: str = "dataset/val/images",
    val_npz_dir: str = "dataset/val/labels",
    test_img_dir: str = "dataset/test/images",
    test_npz_dir: str = "dataset/test/labels",
    native_img_dir: str = "data/raw/images",
    experiment_name: str = "cvm-stage2-finetune-concavity",
    run_name: Optional[str] = None,
    stage2_run_id: Optional[str] = None,
    stage2_artifact: str = "stage2_best.pth",
    stage1_run_id: Optional[str] = None,
    stage1_artifact: str = "weights/best.pt",
    skip_eval: bool = False,
):
    load_dotenv()
    device = resolve_device("auto")
    print("=" * 85)
    print(f"🎯 STARTING TARGETED STAGE 2 FINE-TUNING FOR ASYMMETRIC CONCAVITIES")
    print(f"   Device:               {device}")
    print(f"   Resolution:           {target_size}x{target_size} (Sigma: {sigma})")
    print(f"   Epochs:               {epochs} (Base LR: {lr})")
    print(f"   Asymmetric Loss wt:   {lambda_endplate} | Graph Loss: 0.0 (Decoupled)")
    print("=" * 85)

    # 1. Download Stage 1 & Stage 2 weights if run IDs provided
    if stage2_run_id and (not os.path.exists(stage2_weights) or os.path.getsize(stage2_weights) == 0):
        print(f"\n--> Downloading Stage 2 initial weights from MLflow run '{stage2_run_id}'...")
        download_checkpoint_from_mlflow(
            run_id=stage2_run_id,
            artifact_path=stage2_artifact,
            dest_path=stage2_weights,
        )

    if stage1_run_id and (not os.path.exists(stage1_weights) or os.path.getsize(stage1_weights) == 0):
        print(f"\n--> Downloading Stage 1 detector weights from MLflow run '{stage1_run_id}'...")
        download_checkpoint_from_mlflow(
            run_id=stage1_run_id,
            artifact_path=stage1_artifact,
            dest_path=stage1_weights,
        )

    # 2. Build Model & Load Weights
    print(f"\n--> Initializing CephalometricSwinGCN and loading '{stage2_weights}'...")
    model = CephalometricSwinGCN(
        num_landmarks=NUM_LANDMARKS,
        pretrained=False,
        img_size=target_size,
    )

    if not os.path.exists(stage2_weights):
        raise FileNotFoundError(f"Cannot find Stage 2 weights at '{stage2_weights}'!")

    checkpoint = torch.load(stage2_weights, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("stage2."):
            k = k[len("stage2."):]
        cleaned[k] = v

    incompatible = model.load_state_dict(cleaned, strict=False)
    print(f"    Loaded weights (missing: {len(incompatible.missing_keys)}, unexpected: {len(incompatible.unexpected_keys)})")
    model = model.to(device)

    # 3. Dataloaders with sharpened sigma
    print(f"\n--> Building ROI Dataloaders (sigma={sigma})...")
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
        jitter_trans=0.04,
        jitter_scale=0.05,
        num_workers=4,
        native_img_dir=native_img_dir if (native_img_dir and os.path.exists(native_img_dir)) else None,
    )

    # 4. Losses & Optimizer
    param_groups = get_llrd_param_groups(model, base_lr=lr, weight_decay=1e-4, decay_rate=0.85)
    optimizer = optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=2, total_epochs=epochs, min_lr_ratio=0.05)

    awl_loss = AdaptiveWingLoss().to(device)
    wing_loss = WingLoss(img_size=float(target_size)).to(device)
    endplate_loss = AsymmetricEndplateConcavityLoss(weight_tangential=2.5, weight_depth=1.0).to(device)
    landmark_weights = get_landmark_weights(device)

    # 5. MLflow Init
    import mlflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.amirlogedixyz.ir")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    actual_run_name = run_name or f"stage2_finetune_concavity_{int(time.time())}"

    best_val_loss = float("inf")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=actual_run_name) as run:
        mlflow.log_params({
            "stage2_base_weights": stage2_weights,
            "stage2_run_id": stage2_run_id,
            "epochs": epochs,
            "lr": lr,
            "sigma": sigma,
            "lambda_hm": lambda_hm,
            "lambda_cd": lambda_cd,
            "lambda_graph": 0.0,
            "lambda_endplate": lambda_endplate,
            "target_size": target_size,
        })

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_loss_hm = 0.0
            epoch_loss_cd = 0.0
            epoch_loss_endplate = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}] Finetune Stage 2", leave=False)
            for batch in pbar:
                images = batch["image"].to(device)
                gt_coords = batch["coords"].to(device)
                gt_heatmaps = batch["heatmaps"].to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred_heatmaps, pred_coords = model(images)
                    l_hm = awl_loss(pred_heatmaps, gt_heatmaps, landmark_weights)
                    l_cd = wing_loss(pred_coords, gt_coords, landmark_weights)
                    l_endplate = endplate_loss(pred_coords, gt_coords)

                    total_loss = (lambda_hm * l_hm) + (lambda_cd * l_cd) + (lambda_endplate * l_endplate)

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += total_loss.item()
                epoch_loss_hm += (lambda_hm * l_hm).item()
                epoch_loss_cd += (lambda_cd * l_cd).item()
                epoch_loss_endplate += (lambda_endplate * l_endplate).item()

            scheduler.step()
            num_batches = len(train_loader)
            train_loss = epoch_loss / num_batches

            # Validation
            val_loss, val_metrics = validate_stage2_epoch(
                model=model,
                dataloader=val_loader,
                awl_loss=awl_loss,
                wing_loss=wing_loss,
                graph_loss=None,
                landmark_weights=landmark_weights,
                lambda_hm=lambda_hm,
                lambda_cd=lambda_cd,
                lambda_graph=0.0,
                target_size=target_size,
                device=device,
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_loss_hm": epoch_loss_hm / num_batches,
                "train_loss_cd": epoch_loss_cd / num_batches,
                "train_loss_endplate": epoch_loss_endplate / num_batches,
                "val_loss": val_loss,
                "val_mean_error_norm": val_metrics.get("mean_error_norm", 0.0),
                "val_sdr_2_0": val_metrics.get("sdr_2_0", 0.0),
                "val_sdr_3_0": val_metrics.get("sdr_3_0", 0.0),
            }, step=epoch)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} (Endplate: {(epoch_loss_endplate / num_batches):.4f}) | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val SDR @ 2%: {val_metrics.get('sdr_2_0', 0.0):.2f}%"
            )

            # Checkpoint best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                }, save_path)
                print(f"  --> Saved new best checkpoint to '{save_path}'")

        # 6. Log final weights to MLflow
        print(f"\n--> Logging final best weights '{save_path}' to MLflow...")
        mlflow.log_artifact(save_path, artifact_path="checkpoints")

        # 7. Automated Cascaded Evaluation
        if not skip_eval:
            print("\n--> Running Cascaded Two-Stage Evaluation with fine-tuned Stage 2...")
            from src.eval_two_stage import run_two_stage_evaluation
            eval_payload, eval_dir = run_two_stage_evaluation(
                stage1_weights=stage1_weights,
                stage2_weights=save_path,
                img_dir=test_img_dir,
                npz_dir=test_npz_dir,
                val_img_dir=val_img_dir,
                val_npz_dir=val_npz_dir,
                output_dir="evaluation/two_stage_finetuned",
                run_name=f"{actual_run_name}_eval",
                stage1_img_size=640,
                stage2_img_size=target_size,
                margin_ratio=margin_ratio,
                bottom_margin_ratio=bottom_margin_ratio,
                use_peaks=True,
                num_visualizations=15,
                device_str=str(device),
                log_to_mlflow=True,
                run_id=run.info.run_id,
            )
            print(f"✅ Evaluation complete. Artifacts saved to '{eval_dir}' and logged to MLflow.")


def main():
    parser = argparse.ArgumentParser(description="Targeted Fine-Tuning CLI for Stage 2")
    parser.add_argument("--stage1-weights", type=str, default="artifacts/stage1_best.pt", help="Stage 1 YOLO weights")
    parser.add_argument("--stage2-weights", type=str, default="artifacts/stage2_best.pth", help="Stage 2 Swin weights")
    parser.add_argument("--stage1-run-id", type=str, default="9ce919cc321d423fac3aa0929b14872f")
    parser.add_argument("--stage1-artifact", type=str, default="weights/best.pt")
    parser.add_argument("--stage2-run-id", type=str, default="7ea3e77e3bae4c9392ddf11aee4a9f6a")
    parser.add_argument("--stage2-artifact", type=str, default="stage2_best.pth")
    parser.add_argument("--epochs", type=int, default=20, help="Fine-tuning epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Fine-tuning learning rate (default: 3e-5)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Target Gaussian sigma (default: 2.0)")
    parser.add_argument("--lambda-endplate", type=float, default=5.0, help="Weight for asymmetric endplate loss")
    parser.add_argument("--save-path", type=str, default="artifacts/stage2_best.pth")
    parser.add_argument("--skip-eval", action="store_true")

    args = parser.parse_args()

    finetune_stage2(
        stage1_weights=args.stage1_weights,
        stage2_weights=args.stage2_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sigma=args.sigma,
        lambda_endplate=args.lambda_endplate,
        save_path=args.save_path,
        stage1_run_id=args.stage1_run_id,
        stage1_artifact=args.stage1_artifact,
        stage2_run_id=args.stage2_run_id,
        stage2_artifact=args.stage2_artifact,
        skip_eval=args.skip_eval,
    )


if __name__ == "__main__":
    main()
