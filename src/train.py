import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from src.data import get_dataloaders
from src.models.model import CephalometricSwinGCN
import boto3
from dotenv import load_dotenv
import os
from src.data import NUM_LANDMARKS

TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_NPZ_DIR = "dataset/train/labels"

VAL_IMG_DIR = "dataset/val/images"
VAL_NPZ_DIR = "dataset/val/labels"

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
LAMBDA_HM = 1.0
LAMBDA_CD = 10.0
EARLY_STOPPING_PATIENCE = 10
SAVE_PATH = "./artifacts/best.pth"

load_dotenv()


s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_USERNAME") or os.getenv("MINIO_USER") or os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_PASSWORD") or os.getenv("MINIO_SECRET_KEY"),
)

BUCKET_NAME = os.getenv("MINIO_BUCKET", "cvm-artifacts")


def upload_artifact_to_minio(file_path, object_name=None):
    """Uploads a file to MinIO."""
    if object_name is None:
        object_name = os.path.basename(file_path)

    try:
        s3_client.upload_file(file_path, BUCKET_NAME, object_name)
        print(f"✅ Successfully uploaded {object_name} to MinIO.")
    except Exception as e:
        print(f"❌ Failed to upload {object_name} to MinIO: {e}")


def train_epoch(
    model,
    dataloader,
    optimizer,
    mse_loss,
    l1_loss,
    lambda_hm,
    lambda_cd,
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

        loss_heatmap = mse_loss(pred_heatmaps, gt_heatmaps)
        loss_coord = l1_loss(pred_coords, gt_coords)

        total_loss = (lambda_hm * loss_heatmap) + (lambda_cd * loss_coord)

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        pbar.set_postfix(loss=f"{epoch_loss / step:.4f}")

    return epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def validate_epoch(
    model,
    dataloader,
    mse_loss,
    l1_loss,
    lambda_hm,
    lambda_cd,
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

            loss_heatmap = mse_loss(pred_heatmaps, gt_heatmaps)
            loss_coord = l1_loss(pred_coords, gt_coords)

            total_loss = (lambda_hm * loss_heatmap) + (lambda_cd * loss_coord)
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


def main(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, patience: int = EARLY_STOPPING_PATIENCE):
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

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # --- Training Loop ---
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            mse_loss,
            l1_loss,
            LAMBDA_HM,
            LAMBDA_CD,
            device,
            epoch=epoch + 1,
            epochs=epochs,
        )

        val_loss, metrics = validate_epoch(
            model,
            val_loader,
            mse_loss,
            l1_loss,
            LAMBDA_HM,
            LAMBDA_CD,
            device,
            img_size=640,
            epoch=epoch + 1,
            epochs=epochs,
        )

        epoch_time = time.time() - start_time

        scheduler.step(val_loss)

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

            # --- MINIO UPLOAD TRIGGER ---
            upload_artifact_to_minio(SAVE_PATH, "best_latest.pth")
            # upload_artifact_to_minio(SAVE_PATH, f"best_epoch_{epoch + 1}.pth")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(f"Early stopping triggered: Validation loss did not improve for {patience} consecutive epochs.")
                break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Cephalometric Swin-GCN model")
    parser.add_argument("--epochs", "-e", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--patience", "-p", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience (epochs)")
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)



