import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from src.utils.roi_utils import ROIBoundingBox


class CervicalYOLODetector:
    """
    Stage 1 YOLO Object Detection Model for Cervical Vertebrae ROI localization.
    Detects the isotropic square ROI covering cervical vertebrae C2, C3, and C4.
    Outputs ROIBoundingBox instances compatible with Stage 2 fine Swin refiner.
    """
    def __init__(
        self,
        weights: Optional[str] = None,
        conf_threshold: float = 0.25,
        device: Optional[Union[str, torch.device]] = None,
        default_model: str = "yolo26s.pt",
    ):
        self.conf_threshold = conf_threshold
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = str(device)

        model_path = weights if (weights and os.path.exists(weights)) else default_model
        self.model = YOLO(model_path)
        self.weights_path = weights

    def eval(self):
        """No-op for API compatibility with PyTorch modules."""
        return self

    def parameters(self):
        """Yields underlying model parameters if available."""
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Module):
            return self.model.model.parameters()
        return iter([])

    def predict_roi(
        self,
        image: Union[np.ndarray, str, Path],
        target_size: int = 640,
        margin_ratio: float = 0.05,
    ) -> ROIBoundingBox:
        """
        Runs YOLO detection on an image and returns an isotropic square ROIBoundingBox.

        Args:
            image: Image as numpy array (H, W, C) BGR/RGB, or path string.
            target_size: Working input size for inference.
            margin_ratio: Additional safety margin around YOLO bounding box.

        Returns:
            ROIBoundingBox covering the detected vertebrae.
        """
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Could not load image from: {image}")
            img_h, img_w = img_bgr.shape[:2]
            source = img_bgr
        else:
            img_h, img_w = image.shape[:2]
            source = image

        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            device=self.device,
            imgsz=target_size,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            # Fallback to standard anatomical anchor when no detection is found
            c_u = img_w * 0.42
            c_v = img_h * 0.52
            span = 170.0
            D = int(round(max(span * 1.5, 180.0)))
            r = D / 2.0
            return ROIBoundingBox(
                u1=int(round(c_u - r)),
                v1=int(round(c_v - r)),
                u2=int(round(c_u - r)) + D,
                v2=int(round(c_v - r)) + D,
                D=D,
                center_u=c_u,
                center_v=c_v,
                span=span,
                img_w=img_w,
                img_h=img_h,
                is_fallback=True,
                fallback_reason="YOLO detected zero bounding boxes",
            )

        # Select highest-confidence detection of cervical spine (class 0)
        best_box = None
        best_conf = -1.0
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id == 0 and conf > best_conf:
                best_conf = conf
                best_box = box

        if best_box is None:
            best_box = boxes[0]

        # Extract [x1, y1, x2, y2] in pixel coordinates
        xyxy = best_box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

        # Convert to isotropic square box with margin
        box_w = x2 - x1
        box_h = y2 - y1
        span = max(box_w, box_h)
        c_u = (x1 + x2) / 2.0
        c_v = (y1 + y2) / 2.0

        D = int(round(max(span * (1.0 + 2.0 * margin_ratio), 180.0)))
        r = D / 2.0
        u1 = int(round(c_u - r))
        v1 = int(round(c_v - r))

        return ROIBoundingBox(
            u1=u1,
            v1=v1,
            u2=u1 + D,
            v2=v1 + D,
            D=D,
            center_u=c_u,
            center_v=c_v,
            span=span,
            img_w=img_w,
            img_h=img_h,
            is_fallback=False,
        )

    def predict_batch_rois(
        self,
        images: Union[List[np.ndarray], torch.Tensor],
        img_w: int = 640,
        img_h: int = 640,
    ) -> List[ROIBoundingBox]:
        """
        Runs batch prediction returning a list of ROIBoundingBox objects.
        """
        if isinstance(images, torch.Tensor):
            # Tensor [B, 3, H, W] in [0, 1] normalized
            imgs_np = []
            for b in range(images.shape[0]):
                img_t = images[b].detach().cpu().permute(1, 2, 0).numpy()
                img_u8 = (np.clip(img_t, 0.0, 1.0) * 255.0).astype(np.uint8)
                imgs_np.append(img_u8)
        else:
            imgs_np = images

        boxes_out = []
        for img in imgs_np:
            boxes_out.append(self.predict_roi(img))
        return boxes_out
