import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.constants import NUM_LANDMARKS
from src.models.model import CephalometricSwinGCN
from src.models.yolo_stage1 import CervicalYOLODetector
from src.utils.roi_utils import (
    ROIBoundingBox,
    crop_roi_with_padding,
    inverse_transform_coords,
    transform_coords_to_roi,
)


class TwoStageCephalometricSwinGCN(nn.Module):
    """
    Two-Stage Cascade Landmark Detection System for Cervical Vertebrae:
    - Stage 1 (YOLO ROI Detector): Fast, robust object detector that outputs isotropic square ROI
      covering cervical vertebrae C2, C3, and C4.
    - Stage 2 (Fine Swin-GCN Refiner): High-resolution model operating directly on 448x448 (or 672x672)
      cervical vertebrae crops.
    - Cascaded Inference: Runs YOLO -> Crop & Pad ROI -> Stage 2 Swin-GCN -> Exact Inverse Re-projection.
    """
    def __init__(
        self,
        num_landmarks: int = NUM_LANDMARKS,
        stage1_weights: Optional[str] = "artifacts/yolo_stage1.pt",
        stage2_img_size: int = 448,
        margin_ratio: float = 0.22,
        min_size: int = 180,
        stage2_backbone: str = "swin_base_patch4_window7_224",
        pretrained_stage2: bool = True,
        init_temperature: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.stage2_img_size = stage2_img_size
        self.margin_ratio = margin_ratio
        self.min_size = min_size

        # Stage 1: YOLO Object Detector for Cervical Spine ROI
        self.stage1_detector = CervicalYOLODetector(
            weights=stage1_weights,
            device=device,
        )

        # Stage 2: Fine vertebrae ROI refiner (448x448 or 672x672)
        self.stage2 = CephalometricSwinGCN(
            num_landmarks=num_landmarks,
            pretrained=pretrained_stage2,
            img_size=stage2_img_size,
            init_temperature=init_temperature,
        )

    @property
    def stage1(self):
        """Property returning the stage1 detector for API compatibility."""
        return self.stage1_detector

    def freeze_stage1(self) -> None:
        """No-op kept for backward compatibility."""
        self.stage1_detector.eval()

    def unfreeze_stage1(self) -> None:
        """No-op kept for backward compatibility."""
        pass

    def load_stage2_weights(
        self,
        weights_path: str,
        device: torch.device,
        strict: bool = False,
    ) -> None:
        """Loads weights into Stage 2 model."""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Stage 2 weights not found at: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=device)
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

        self.stage2.load_state_dict(cleaned, strict=strict)

    def extract_rois_from_yolo(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[ROIBoundingBox]]:
        """
        Runs YOLO detector on a batch of full images and extracts padded isotropic square ROIs.

        Args:
            images: [B, 3, H, W] tensor in range [0, 1].

        Returns:
            roi_batch: [B, 3, stage2_img_size, stage2_img_size] tensor.
            bboxes: List of B ROIBoundingBox objects.
        """
        B, C, H, W = images.shape
        device = images.device

        bboxes = self.stage1_detector.predict_batch_rois(images, img_w=W, img_h=H)

        crops = []
        for b in range(B):
            img_np = (images[b].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            crop_np, _ = crop_roi_with_padding(
                image=img_np,
                bbox=bboxes[b],
                target_size=self.stage2_img_size,
            )
            crop_t = torch.from_numpy(crop_np).permute(2, 0, 1).float() / 255.0
            crops.append(crop_t)

        roi_batch = torch.stack(crops, dim=0).to(device)
        return roi_batch, bboxes

    def forward_stage2_fine(
        self,
        roi_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs forward pass through Stage 2 fine model on ROI crops."""
        return self.stage2(roi_images)

    def forward(
        self,
        images: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Full Cascaded Two-Stage Forward Pass:
        1. Stage 1 YOLO Object Detection -> ROI BBoxes
        2. Isotropic Padded ROI Cropping
        3. Stage 2 Fine Swin-GCN Landmark Regression
        4. Invertible Coordinate Re-projection to Full Image Canvas Space [0, 1]

        Args:
            images: [B, 3, H, W] full cephalogram tensor.
            return_intermediates: If True, returns dict with intermediate tensors and bboxes.

        Returns:
            final_coords: [B, 13, 2] in normalized [0, 1] canvas space, OR dictionary.
        """
        B, C, H, W = images.shape
        device = images.device

        # 1 & 2. YOLO ROI Detection and cropping
        roi_batch, bboxes = self.extract_rois_from_yolo(images)

        # 3. Stage 2 Fine forward pass
        stage2_hms, stage2_coords_roi = self.stage2(roi_batch)

        # 4. Inverse coordinate re-projection to canvas space [0, 1]
        reprojected_coords = torch.zeros_like(stage2_coords_roi)
        fallback_count = 0

        for b in range(B):
            bbox = bboxes[b]
            if bbox.is_fallback:
                fallback_count += 1
            reprojected_coords[b] = inverse_transform_coords(
                roi_coords=stage2_coords_roi[b],
                bbox=bbox,
                target_size=self.stage2_img_size,
                output_normalized=True,
            )

        if return_intermediates:
            return {
                "stage2_heatmaps": stage2_hms,
                "stage2_coords_roi": stage2_coords_roi,
                "final_coords": reprojected_coords,
                "bboxes": bboxes,
                "roi_batch": roi_batch,
                "fallback_count": fallback_count,
            }

        return reprojected_coords
