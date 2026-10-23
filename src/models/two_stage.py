import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.constants import NUM_LANDMARKS
from src.models.model import CephalometricSwinGCN
from src.utils.roi_utils import (
    ROIBoundingBox,
    compute_roi_bbox,
    crop_roi_with_padding,
    inverse_transform_coords,
    transform_coords_to_roi,
    validate_stage1_box,
)


class TwoStageCephalometricSwinGCN(nn.Module):
    """
    Two-Stage Coarse-to-Fine Landmark Detection System for Cervical Vertebrae:
    - Stage 1 (Coarse Model): Operates on 640x640 full cephalograms to localize cervical landmarks globally.
    - Stage 2 (Fine Model): Operates on 448x448 (or 672x672) strictly square, isotropic cervical spine ROIs.
    - Cascaded Inference: Runs Stage 1 -> Shared BBox Extraction -> Crop & Pad -> Stage 2 -> Exact Inverse Coordinate Re-projection.
    """
    def __init__(
        self,
        num_landmarks: int = NUM_LANDMARKS,
        stage1_img_size: int = 640,
        stage2_img_size: int = 448,
        margin_ratio: float = 0.18,
        min_size: int = 180,
        stage1_backbone: str = "swin_base_patch4_window7_224",
        stage2_backbone: str = "swin_base_patch4_window7_224",
        pretrained_stage1: bool = True,
        pretrained_stage2: bool = True,
        init_temperature: float = 0.1,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.stage1_img_size = stage1_img_size
        self.stage2_img_size = stage2_img_size
        self.margin_ratio = margin_ratio
        self.min_size = min_size

        # Stage 1: Coarse global landmark detector (640x640)
        self.stage1 = CephalometricSwinGCN(
            num_landmarks=num_landmarks,
            pretrained=pretrained_stage1,
            img_size=stage1_img_size,
            init_temperature=init_temperature,
        )

        # Stage 2: Fine vertebrae ROI refiner (448x448 or 672x672)
        self.stage2 = CephalometricSwinGCN(
            num_landmarks=num_landmarks,
            pretrained=pretrained_stage2,
            img_size=stage2_img_size,
            init_temperature=init_temperature,
        )

    def freeze_stage1(self) -> None:
        """Freezes Stage 1 parameters and sets to eval mode."""
        self.stage1.eval()
        for p in self.stage1.parameters():
            p.requires_grad = False

    def unfreeze_stage1(self) -> None:
        """Unfreezes Stage 1 parameters."""
        for p in self.stage1.parameters():
            p.requires_grad = True

    def load_stage1_weights(
        self,
        weights_path: str,
        device: torch.device,
        strict: bool = False,
    ) -> None:
        """Loads weights into Stage 1 model."""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Stage 1 weights not found at: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=device)
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

        self.stage1.load_state_dict(cleaned, strict=strict)
        self.freeze_stage1()
        print(f"--> Successfully loaded and frozen Stage 1 weights from '{weights_path}'")

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
        print(f"--> Successfully loaded Stage 2 weights from '{weights_path}'")

    def forward_coarse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs Stage 1 on full images."""
        return self.stage1(x)

    def forward_fine(self, roi_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs Stage 2 on cropped ROI images."""
        return self.stage2(roi_images)

    def extract_rois_from_coarse_coords(
        self,
        images: torch.Tensor,
        coarse_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[ROIBoundingBox]]:
        """
        Crops strictly square, isotropic ROIs from batch of canvas images using coarse coordinates.

        Args:
            images: [B, C, H, W] normalized canvas images on device.
            coarse_coords: [B, 13, 2] in normalized [0, 1] canvas space.

        Returns:
            roi_batch: [B, C, target_size, target_size] tensor on same device.
            bboxes: List of ROIBoundingBox objects.
        """
        B, C, H, W = images.shape
        device = images.device
        dtype = images.dtype

        # Move images and coords to numpy for padding-aware crop
        images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()  # [B, H, W, C]
        coords_np = coarse_coords.detach().cpu().numpy()  # [B, 13, 2]

        cropped_list = []
        bboxes = []

        for b in range(B):
            bbox = compute_roi_bbox(
                coords=coords_np[b],
                img_w=W,
                img_h=H,
                margin_ratio=self.margin_ratio,
                min_size=self.min_size,
            )
            bboxes.append(bbox)

            cropped_patch, _ = crop_roi_with_padding(
                image=images_np[b],
                bbox=bbox,
                target_size=self.stage2_img_size,
                interpolation=cv2.INTER_LINEAR,
            )
            cropped_list.append(cropped_patch)

        # Re-stack into [B, C, target_size, target_size] tensor
        stacked_np = np.stack(cropped_list, axis=0).transpose(0, 3, 1, 2)  # [B, C, K, K]
        roi_batch = torch.from_numpy(stacked_np).to(device=device, dtype=dtype)

        return roi_batch, bboxes

    def forward(
        self,
        images: torch.Tensor,
        refine_2pass: bool = False,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Full Cascaded Two-Stage Forward Pass:
        1. Stage 1 Coarse Forward
        2. Bounding Box Extraction & Padded ROI Cropping
        3. Stage 2 Fine Forward
        4. Invertible Coordinate Re-projection to Canvas Space
        5. (Optional) 2-Pass Refinement: Re-centers crop on Pass 1 output and predicts Pass 2.

        Args:
            images: [B, 3, 640, 640] full cephalogram tensor.
            refine_2pass: If True, executes a second pass using Pass 1's refined center.
            return_intermediates: If True, returns full dict with intermediate tensors and bboxes.

        Returns:
            final_coords: [B, 13, 2] in normalized [0, 1] canvas space, OR dictionary if return_intermediates=True.
        """
        B, C, H, W = images.shape
        device = images.device

        # Pass 1: Stage 1 Coarse forward pass
        with torch.no_grad():
            stage1_hms, stage1_coords = self.stage1(images)

        # Extract Stage 2 ROIs based on Stage 1 predictions
        roi_batch_1, bboxes_1 = self.extract_rois_from_coarse_coords(images, stage1_coords)

        # Stage 2 Fine forward pass
        stage2_hms_1, stage2_coords_roi_1 = self.stage2(roi_batch_1)

        # Inverse coordinate re-projection to canvas space [0, 1]
        reprojected_coords_1 = torch.zeros_like(stage2_coords_roi_1)
        fallback_count = 0

        for b in range(B):
            bbox = bboxes_1[b]
            if bbox.is_fallback:
                # Sanity fallback: use Stage 1 coordinates directly
                reprojected_coords_1[b] = stage1_coords[b]
                fallback_count += 1
            else:
                reprojected_coords_1[b] = inverse_transform_coords(
                    roi_coords=stage2_coords_roi_1[b],
                    bbox=bbox,
                    target_size=self.stage2_img_size,
                    output_normalized=True,
                )

        final_coords = reprojected_coords_1

        # Optional Pass 2: Iterative re-centering refinement
        if refine_2pass:
            roi_batch_2, bboxes_2 = self.extract_rois_from_coarse_coords(images, final_coords)
            stage2_hms_2, stage2_coords_roi_2 = self.stage2(roi_batch_2)
            reprojected_coords_2 = torch.zeros_like(stage2_coords_roi_2)

            for b in range(B):
                bbox = bboxes_2[b]
                if bbox.is_fallback:
                    reprojected_coords_2[b] = final_coords[b]
                else:
                    reprojected_coords_2[b] = inverse_transform_coords(
                        roi_coords=stage2_coords_roi_2[b],
                        bbox=bbox,
                        target_size=self.stage2_img_size,
                        output_normalized=True,
                    )
            final_coords = reprojected_coords_2

        # Safeguard against extreme localized crop drift (> 50px from coarse Stage 1)
        drift = torch.norm(final_coords - stage1_coords, dim=-1) * float(self.stage1_img_size)
        final_coords = torch.where(drift.unsqueeze(-1) > 50.0, stage1_coords, final_coords)

        if return_intermediates:
            return {
                "stage1_heatmaps": stage1_hms,
                "stage1_coords": stage1_coords,
                "stage2_heatmaps": stage2_hms_1,
                "stage2_coords_roi": stage2_coords_roi_1,
                "final_coords": final_coords,
                "bboxes": bboxes_1,
                "roi_batch": roi_batch_1,
                "fallback_count": fallback_count,
            }

        return final_coords
