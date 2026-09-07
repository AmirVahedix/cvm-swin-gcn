import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss (AWL) for heatmap landmark regression.
    Ref: Wang et al., "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression", ICCV 2019.
    
    Dynamically adapts gradient magnitude for small localization errors near Gaussian heatmap peaks 
    while smoothly penalizing background zero regions.
    """
    def __init__(self, omega: float = 14.0, theta: float = 0.5, epsilon: float = 1.0, alpha: float = 2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmaps of shape [B, N, H, W]
            target: Ground truth heatmaps of shape [B, N, H, W]
        """
        delta = (target - pred).abs().clamp(min=1e-8)

        # Non-linear AWL calculation
        A = self.omega * (1.0 / (1.0 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1.0) * (1.0 / self.epsilon)
        C = (self.theta * A) - (self.omega * torch.log(1.0 + torch.pow(self.theta / self.epsilon, self.alpha - target)))

        mask_small = delta < self.theta
        mask_large = ~mask_small

        loss = torch.zeros_like(delta)
        loss[mask_small] = self.omega * torch.log(1.0 + torch.pow(delta[mask_small] / self.epsilon, self.alpha - target[mask_small]))
        loss[mask_large] = (A[mask_large] * delta[mask_large]) - C[mask_large]

        return loss.mean()


class WingLoss(nn.Module):
    """
    Wing Loss for robust direct coordinate regression.
    Ref: Feng et al., "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks", CVPR 2018.
    
    When inputs are normalized in [0, 1], img_size (default 640.0) automatically converts 
    pixel thresholds (omega=10.0 px, epsilon=2.0 px) into normalized scale [0, 1].
    """
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0, img_size: float = 640.0):
        super().__init__()
        # Calibrate pixel thresholds into normalized coordinate space [0, 1]
        self.omega = omega / img_size
        self.epsilon = epsilon / img_size
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, landmark_weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted coordinates of shape [B, N, 2] in normalized scale [0, 1]
            target: Ground truth coordinates of shape [B, N, 2] in normalized scale [0, 1]
            landmark_weights: Optional per-landmark weights of shape [N] or [1, N]
        """
        delta = (pred - target).abs().clamp(min=1e-8)
        mask_small = delta < self.omega
        mask_large = ~mask_small
        
        loss = torch.zeros_like(delta)
        loss[mask_small] = self.omega * torch.log(1.0 + delta[mask_small] / self.epsilon)
        loss[mask_large] = delta[mask_large] - self.C.to(delta.device)

        # Average across (x, y) dimensions per landmark -> shape [B, N]
        landmark_loss = loss.mean(dim=-1)

        # Apply valid ground-truth mask (excluding missing landmarks marked as -1.0)
        valid_mask = (target[:, :, 0] >= 0) & (target[:, :, 1] >= 0)
        landmark_loss = landmark_loss * valid_mask.float()

        if landmark_weights is not None:
            landmark_loss = landmark_loss * landmark_weights.unsqueeze(0).to(delta.device)

        valid_count = valid_mask.float().sum().clamp(min=1.0)
        return landmark_loss.sum() / valid_count


class AnatomicalGraphLoss(nn.Module):
    """
    Anatomical Edge Length Consistency Loss.
    Penalizes deviations in pairwise Euclidean distance between connected GCN landmark nodes
    relative to ground-truth pairwise distances.
    """
    def __init__(self, adj_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer("adj_matrix", adj_matrix)

    def forward(self, pred_coords: torch.Tensor, gt_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_coords: Shape [B, N, 2] in normalized scale [0, 1]
            gt_coords: Shape [B, N, 2] in normalized scale [0, 1]
        """
        device = pred_coords.device
        adj = self.adj_matrix.to(device)

        # Get connected node pairs (upper triangular of non-zero adjacency)
        connected_pairs = torch.triu(adj > 0, diagonal=1).nonzero(as_tuple=False)  # Shape: [E, 2]

        if len(connected_pairs) == 0:
            return torch.tensor(0.0, device=device)

        i_idx = connected_pairs[:, 0]
        j_idx = connected_pairs[:, 1]

        # Compute pairwise vectors with stable sqrt to prevent NaN gradients: [B, E]
        pred_diff_sq = torch.sum((pred_coords[:, i_idx, :] - pred_coords[:, j_idx, :]) ** 2, dim=-1)
        gt_diff_sq = torch.sum((gt_coords[:, i_idx, :] - gt_coords[:, j_idx, :]) ** 2, dim=-1)

        pred_dist = torch.sqrt(pred_diff_sq + 1e-8)
        gt_dist = torch.sqrt(gt_diff_sq + 1e-8)

        # Valid mask for both nodes in pair
        valid_i = (gt_coords[:, i_idx, 0] >= 0) & (gt_coords[:, i_idx, 1] >= 0)
        valid_j = (gt_coords[:, j_idx, 0] >= 0) & (gt_coords[:, j_idx, 1] >= 0)
        valid_pairs = (valid_i & valid_j).float()

        edge_error = (pred_dist - gt_dist).abs() * valid_pairs
        valid_count = valid_pairs.sum().clamp(min=1.0)

        return edge_error.sum() / valid_count
