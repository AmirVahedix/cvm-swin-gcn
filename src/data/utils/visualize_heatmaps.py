import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2


def find_corresponding_image(base_name, images_dir):
    """
    Attempts to find the corresponding image file regardless of extension.
    """
    # Common image extensions to check
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for ext in extensions:
        img_path = os.path.join(images_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def visualize_random_samples(npz_dir, images_dir, num_samples=4):
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))

    if len(npz_files) < num_samples:
        print(f"Not enough .npz files. Found {len(npz_files)}, need {num_samples}.")
        num_samples = len(npz_files)
        if num_samples == 0:
            return

    # Pick random NPZ files
    selected_npz = random.sample(npz_files, num_samples)

    # Setup matplotlib grid (2x2 for 4 samples)
    rows = int(np.ceil(num_samples / 2))
    cols = 2 if num_samples > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx, npz_path in enumerate(selected_npz):
        base_name = os.path.splitext(os.path.basename(npz_path))[0]

        # 1. Find and load the original image
        img_path = find_corresponding_image(base_name, images_dir)
        if not img_path:
            axes[idx].set_title(f"Image not found for\n{base_name}")
            axes[idx].axis("off")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert for matplotlib

        # 2. Load the heatmaps
        with np.load(npz_path) as data:
            heatmaps = data["heatmaps"]  # Shape: (13, H, W)

        # 3. Create a composite heatmap
        # Taking the max across axis 0 flattens the 13 channels into 1 map
        # showing all keypoints simultaneously.
        composite_heatmap = np.max(heatmaps, axis=0)

        # Mask out values close to zero so they become transparent
        heatmap_masked = np.ma.masked_where(composite_heatmap < 0.1, composite_heatmap)

        # 4. Plot original image
        axes[idx].imshow(img)

        # 5. Overlay the heatmap
        # alpha controls transparency, cmap 'jet' gives the classic blue-to-red look
        axes[idx].imshow(heatmap_masked, cmap="jet", alpha=0.6)

        axes[idx].set_title(f"Overlay: {base_name}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Update these paths to match your directories
    NPZ_DIRECTORY = "data/heatmaps"
    IMAGES_DIRECTORY = "data/images"

    visualize_random_samples(NPZ_DIRECTORY, IMAGES_DIRECTORY, num_samples=2)
