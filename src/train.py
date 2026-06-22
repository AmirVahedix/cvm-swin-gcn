import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataloader import get_dataloaders
from src.models.model import CephalometricSwinGCN

TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_NPZ_DIR = "dataset/train/labels"

VAL_IMG_DIR = "dataset/val/images"
VAL_NPZ_DIR = "dataset/val/labels"

BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
LAMBDA_HM = 1.0
LAMBDA_CD = 10.0
SAVE_PATH = "best_swin_gcn_model.pth"


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

    model = CephalometricSwinGCN(num_landmarks=13).to(device)

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
            print(f"--> Saved new best model (Val Loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    main()
