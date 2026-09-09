import os
from .dataset import CVMDataset
from .transforms import get_transforms
from torch.utils.data import DataLoader


TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_NPZ_DIR = "dataset/train/labels"

VAL_IMG_DIR = "dataset/val/images"
VAL_NPZ_DIR = "dataset/val/labels"

TEST_IMG_DIR = "dataset/test/images"
TEST_NPZ_DIR = "dataset/test/labels"


def get_dataloaders(
    train_img_dir,
    train_npz_dir,
    val_img_dir,
    val_npz_dir,
    batch_size=8,
    img_size=1024,
    num_workers=4,
):
    train_transform, val_transform = get_transforms(img_size=img_size)

    train_files = [
        f for f in os.listdir(train_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    val_files = [
        f for f in os.listdir(val_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    train_dataset = CVMDataset(
        train_img_dir,
        train_npz_dir,
        train_files,
        transform=train_transform,
        img_size=img_size,
    )
    val_dataset = CVMDataset(
        val_img_dir, val_npz_dir, val_files, transform=val_transform, img_size=img_size
    )

    pin_memory = torch.cuda.is_available() or (num_workers > 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader


def get_test_dataloader(
    test_img_dir=TEST_IMG_DIR,
    test_npz_dir=TEST_NPZ_DIR,
    batch_size=8,
    img_size=1024,
    num_workers=4,
):
    """
    Factory function specifically for the test dataset.
    """
    _, test_transform = get_transforms(img_size=img_size)

    test_files = [
        f for f in os.listdir(test_img_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    test_dataset = CVMDataset(
        image_dir=test_img_dir,
        npz_dir=test_npz_dir,
        image_filenames=test_files,
        transform=test_transform,
        img_size=img_size,
    )

    pin_memory = torch.cuda.is_available() or (num_workers > 0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return test_loader


if __name__ == "__main__":

    def get_image_files(directory):
        return [
            f for f in os.listdir(directory) if f.endswith((".png", ".jpg", ".jpeg"))
        ]

    train_files = get_image_files(TRAIN_IMG_DIR)
    val_files = get_image_files(VAL_IMG_DIR)
    test_files = get_image_files(TEST_IMG_DIR)

    print(
        f"Loaded from disk - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}"
    )

    train_loader, val_loader = get_dataloaders(
        train_img_dir=TRAIN_IMG_DIR,
        train_npz_dir=TRAIN_NPZ_DIR,
        val_img_dir=VAL_IMG_DIR,
        val_npz_dir=VAL_NPZ_DIR,
    )
    test_loader = get_test_dataloader()

    print("\nVerifying Train Loader...")
    for batch in train_loader:
        print("Batch verification successful:")
        print(f" -> Images batch tensor shape:    {batch['image'].shape}")
        print(f" -> Heatmaps batch tensor shape:  {batch['heatmaps'].shape}")
        print(f" -> Coordinates batch tensor shape: {batch['coords'].shape}")
        break
