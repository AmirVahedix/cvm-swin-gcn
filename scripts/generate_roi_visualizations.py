#!/usr/bin/env python3
"""
Generate publication-quality side-by-side ROI visualizations for cephalometric cervical vertebrae landmarks.
Left panel: Ground Truth
Right panel: Model Prediction
Zoomed strictly to the cervical vertebrae region (C2, C3, C4) with anatomical contours and landmark indices.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import get_test_dataloader
from src.eval import (
    load_model,
    predict_batch,
    create_side_by_side_roi_visualization,
)


def generate_roi_visualizations(
    weights_path: str = "artifacts/best.pth",
    test_img_dir: str = "dataset/test/images",
    test_npz_dir: str = "dataset/test/labels",
    output_dir: str = "evaluation/visualizations",
    num_samples: int = 8,
    img_size: int = 640,
    panel_size: int = 560,
    device_str: str | None = None,
    use_tta: bool = False,
) -> list[str]:
    """
    Renders side-by-side Ground Truth vs Model Prediction ROI visualizations
    and saves them to output_dir.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"--> Loading model weights from: {weights_path} onto {device}")
    model = load_model(weights_path, device=device, img_size=img_size)

    print(f"--> Loading test dataset from: {test_img_dir}")
    test_loader = get_test_dataloader(
        test_img_dir=test_img_dir,
        test_npz_dir=test_npz_dir,
        batch_size=4,
        img_size=img_size,
        num_workers=0,
    )

    saved_paths = []
    saved_count = 0

    with torch.no_grad():
        for batch in test_loader:
            if num_samples > 0 and saved_count >= num_samples:
                break

            images = batch["image"].to(device)
            gt_coords = batch["coords"].cpu().numpy()
            filenames = batch["filename"]
            spacings = (
                batch["pixel_spacing"].cpu().numpy()
                if "pixel_spacing" in batch
                else np.full(len(filenames), 0.1)
            )

            _, pred_coords = predict_batch(
                model=model,
                images=images,
                use_tta=use_tta,
            )
            pred_coords = pred_coords.cpu().numpy()

            for i in range(len(filenames)):
                if num_samples > 0 and saved_count >= num_samples:
                    break

                fname = filenames[i]
                orig_img_path = Path(test_img_dir) / fname
                sample_spacing = float(spacings[i]) if hasattr(spacings, "__getitem__") else 0.1

                if orig_img_path.exists():
                    raw_bgr = cv2.imread(str(orig_img_path))
                    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                else:
                    tensor_np = images[i].cpu().numpy().transpose(1, 2, 0)
                    raw_rgb = (np.clip(tensor_np, 0, 1) * 255).astype(np.uint8)

                vis_rgb = create_side_by_side_roi_visualization(
                    image_rgb=raw_rgb,
                    gt_coords=gt_coords[i],
                    pred_coords=pred_coords[i],
                    sample_name=fname,
                    pixel_spacing=sample_spacing,
                    panel_size=panel_size,
                )

                out_path = out_dir / f"vis_{saved_count + 1:02d}_{Path(fname).stem}.png"
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), vis_bgr)
                saved_paths.append(str(out_path))
                saved_count += 1
                print(f"  [{saved_count}/{num_samples}] Saved: {out_path.name}")

    print(f"\n✅ Successfully generated {len(saved_paths)} side-by-side ROI visualizations in '{out_dir}'")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side ROI comparison PNGs (Ground Truth vs Prediction) for CVM."
    )
    parser.add_argument("--weights", type=str, default="artifacts/best.pth", help="Model checkpoint path.")
    parser.add_argument("--test-img-dir", type=str, default="dataset/test/images", help="Test images directory.")
    parser.add_argument("--test-npz-dir", type=str, default="dataset/test/labels", help="Test labels directory.")
    parser.add_argument("--output-dir", type=str, default="evaluation/visualizations", help="Output directory for PNGs.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to visualize (-1 for all).")
    parser.add_argument("--img-size", type=int, default=640, help="Input model resolution.")
    parser.add_argument("--panel-size", type=int, default=560, help="Resolution per panel in pixels.")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on.")
    parser.add_argument("--use-tta", action="store_true", help="Enable Test-Time Augmentation.")

    args = parser.parse_args()

    generate_roi_visualizations(
        weights_path=args.weights,
        test_img_dir=args.test_img_dir,
        test_npz_dir=args.test_npz_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        img_size=args.img_size,
        panel_size=args.panel_size,
        device_str=args.device,
        use_tta=args.use_tta,
    )


if __name__ == "__main__":
    main()
