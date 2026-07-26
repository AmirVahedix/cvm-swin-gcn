import torch
import torch.nn as nn
import torch.optim as optim
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

BATCH_SIZE = 8
EPOCHS = 1
LR = 1e-4
LAMBDA_HM = 1.0
LAMBDA_CD = 10.0
SAVE_PATH = "./artifacts/best.pth"

load_dotenv()


s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
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
):
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
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

    return epoch_loss / len(dataloader)


def validate_epoch(model, dataloader, mse_loss, l1_loss, lambda_hm, lambda_cd, device):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            gt_heatmaps = batch["heatmaps"].to(device)
            gt_coords = batch["coords"].to(device)

            pred_heatmaps, pred_coords = model(images)

            loss_heatmap = mse_loss(pred_heatmaps, gt_heatmaps)
            loss_coord = l1_loss(pred_coords, gt_coords)

            total_loss = (lambda_hm * loss_heatmap) + (lambda_cd * loss_coord)
            epoch_loss += total_loss.item()

    return epoch_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        train_img_dir=TRAIN_IMG_DIR,
        train_npz_dir=TRAIN_NPZ_DIR,
        val_img_dir=VAL_IMG_DIR,
        val_npz_dir=VAL_NPZ_DIR,
    )

    model = CephalometricSwinGCN(num_landmarks=NUM_LANDMARKS).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,  # type: ignore
    )

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # --- Training Loop ---
    best_val_loss = float("inf")

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            mse_loss,
            l1_loss,
            LAMBDA_HM,
            LAMBDA_CD,
            device,
        )

        val_loss = validate_epoch(
            model, val_loader, mse_loss, l1_loss, LAMBDA_HM, LAMBDA_CD, device
        )

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Checkpoint Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                SAVE_PATH,
            )
            print(f"--> Saved new best model locally (Val Loss: {best_val_loss:.4f})")

            # --- MINIO UPLOAD TRIGGER ---
            upload_artifact_to_minio(SAVE_PATH, "best_latest.pth")
            # upload_artifact_to_minio(SAVE_PATH, f"best_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    main()
