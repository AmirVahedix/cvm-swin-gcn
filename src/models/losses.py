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

    def forward(self, pred: torch.Tensor, target: torch.Tensor, landmark_weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmaps of shape [B, N, H, W]
            target: Ground truth heatmaps of shape [B, N, H, W]
            landmark_weights: Optional per-landmark weights of shape [N] or [1, N]
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

        # Average across spatial dimensions [H, W] -> shape [B, N]
        per_landmark_loss = loss.mean(dim=(-2, -1))

        if landmark_weights is not None:
            per_landmark_loss = per_landmark_loss * landmark_weights.unsqueeze(0).to(delta.device)

        return per_landmark_loss.mean()


class WingLoss(nn.Module):
    """
    Wing Loss for robust direct coordinate regression.
    Ref: Feng et al., "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks", CVPR 2018.
    
    When inputs are normalized in [0, 1], img_size (default 1024.0) automatically converts 
    pixel thresholds (omega=10.0 px, epsilon=2.0 px) into normalized scale [0, 1].
    """
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0, img_size: float = 1024.0):
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
    Supports relative deformation scaling (|pred - gt| / (gt + eps)) to prevent the loss
    from vanishing into insignificance (~0.004) compared to coordinate loss.
    """
    def __init__(self, adj_matrix: torch.Tensor, relative: bool = True):
        super().__init__()
        self.register_buffer("adj_matrix", adj_matrix)
        self.relative = relative

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

        if self.relative:
            # Scale-invariant relative edge length distortion (|pred - gt| / (gt + eps))
            # Ensures meaningful gradient contribution on the order of 0.03 - 0.05
            edge_error = ((pred_dist - gt_dist).abs() / (gt_dist + 1e-4)) * valid_pairs
        else:
            edge_error = (pred_dist - gt_dist).abs() * valid_pairs

        valid_count = valid_pairs.sum().clamp(min=1.0)

        return edge_error.sum() / valid_count


class MorphometricShapeLoss(nn.Module):
    """
    Morphometric Shape Loss for Vertebrae C3 and C4.
    Penalizes deviations in Shape Index (SI) and Taper Ratio (TR) using the exact
    anatomical midline axis formulation from cvm-backend/app/static/calculator.js (classifyC3C4Shape).

    Mathematical Formulation (Anatomical Midline Axis Mode):
      - Inferior Midline: m_inf = (pi + ai) / 2
      - Superior Midline: m_sup = (ps + as) / 2
      - Longitudinal Axis Vector: v = m_sup - m_inf, h_mid = ||v||
      - Unit Vector Longitudinal: u = (ux, uy) = v / h_mid
      - Transverse Axis Unit Vector (pointing anteriorly): w = (-uy, ux) (flipped if w . (ai - pi) < 0)
      - Projected Widths along transverse axis:
          wi = |(ai - pi) . w|
          ws = |(as - ps) . w|
          mean_width = (wi + ws) / 2
      - Projected Heights along longitudinal axis:
          hp = |(ps - pi) . u|
          ha = |(as - ai) . u|
          mean_height = (hp + ha) / 2
      - Shape Index: SI = mean_height / mean_width
      - Taper Ratio: TR = ha / hp
    """
    def __init__(
        self,
        weight_si: float = 1.0,
        weight_taper: float = 1.0,
        beta: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight_si = weight_si
        self.weight_taper = weight_taper
        self.beta = beta
        self.eps = eps

    @staticmethod
    def compute_vertebra_metrics(
        ps: torch.Tensor,
        as_pt: torch.Tensor,
        pi: torch.Tensor,
        ai: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Shape Index (SI) and Taper Ratio (TR) for a batch of vertebrae,
        strictly matching cvm-backend/app/static/calculator.js.

        Args:
            ps: Posterior-Superior coordinates [B, 2]
            as_pt: Anterior-Superior coordinates [B, 2]
            pi: Posterior-Inferior coordinates [B, 2]
            ai: Anterior-Inferior coordinates [B, 2]
            eps: Epsilon for division stability

        Returns:
            si: Shape Index [B, 1] (mean_height / mean_width)
            tr: Taper Ratio [B, 1] (ha / hp)
        """
        # Midline points
        m_inf = (pi + ai) / 2.0  # [B, 2]
        m_sup = (ps + as_pt) / 2.0  # [B, 2]

        # Midline longitudinal axis vector pointing superiorly
        v = m_sup - m_inf  # [B, 2]
        h_mid = torch.sqrt(torch.sum(v ** 2, dim=-1, keepdim=True) + 1e-8)  # [B, 1]

        # Unit vector along longitudinal axis
        u = v / h_mid  # [B, 2]
        ux = u[:, 0:1]  # [B, 1]
        uy = u[:, 1:2]  # [B, 1]

        # Orthogonal transverse axis unit vector (pointing anteriorly)
        wx = -uy  # [B, 1]
        wy = ux  # [B, 1]
        base_dx = ai[:, 0:1] - pi[:, 0:1]  # [B, 1]
        base_dy = ai[:, 1:2] - pi[:, 1:2]  # [B, 1]

        dot_proj = wx * base_dx + wy * base_dy
        sign = torch.where(dot_proj < 0, -1.0, 1.0)
        wx = wx * sign
        wy = wy * sign

        # Projected widths along transverse axis
        wi = torch.abs((ai[:, 0:1] - pi[:, 0:1]) * wx + (ai[:, 1:2] - pi[:, 1:2]) * wy)
        ws = torch.abs((as_pt[:, 0:1] - ps[:, 0:1]) * wx + (as_pt[:, 1:2] - ps[:, 1:2]) * wy)
        mean_width = (wi + ws) / 2.0

        # Projected heights along longitudinal axis
        hp = torch.abs((ps[:, 0:1] - pi[:, 0:1]) * ux + (ps[:, 1:2] - pi[:, 1:2]) * uy)
        ha = torch.abs((as_pt[:, 0:1] - ai[:, 0:1]) * ux + (as_pt[:, 1:2] - ai[:, 1:2]) * uy)
        mean_height = (hp + ha) / 2.0

        si = mean_height / (mean_width + eps)
        tr = ha / (hp + eps)

        return si, tr

    def forward(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_coords: Shape [B, N, 2] in normalized scale [0, 1]
            gt_coords: Shape [B, N, 2] in normalized scale [0, 1]

        Returns:
            Scalar Smooth L1 morphometric shape loss.
        """
        device = pred_coords.device
        total_loss = torch.tensor(0.0, device=device)
        total_valid = 0.0

        # Vertebra landmark quadruplets (ps, as, pi, ai)
        # C3: ps=3, as=4, pi=5, ai=7
        # C4: ps=8, as=9, pi=10, ai=12
        vertebrae_indices = [(3, 4, 5, 7), (8, 9, 10, 12)]

        for ps_idx, as_idx, pi_idx, ai_idx in vertebrae_indices:
            # Check validity of ground-truth corners across batch
            valid = (
                (gt_coords[:, ps_idx, 0] >= 0) & (gt_coords[:, ps_idx, 1] >= 0) &
                (gt_coords[:, as_idx, 0] >= 0) & (gt_coords[:, as_idx, 1] >= 0) &
                (gt_coords[:, pi_idx, 0] >= 0) & (gt_coords[:, pi_idx, 1] >= 0) &
                (gt_coords[:, ai_idx, 0] >= 0) & (gt_coords[:, ai_idx, 1] >= 0)
            )  # [B]

            if not valid.any():
                continue

            # Compute predicted SI and TR (differentiable)
            pred_si, pred_tr = self.compute_vertebra_metrics(
                pred_coords[:, ps_idx, :],
                pred_coords[:, as_idx, :],
                pred_coords[:, pi_idx, :],
                pred_coords[:, ai_idx, :],
                eps=self.eps,
            )

            # Compute ground truth SI and TR
            with torch.no_grad():
                gt_si, gt_tr = self.compute_vertebra_metrics(
                    gt_coords[:, ps_idx, :],
                    gt_coords[:, as_idx, :],
                    gt_coords[:, pi_idx, :],
                    gt_coords[:, ai_idx, :],
                    eps=self.eps,
                )

            # Smooth L1 loss per sample
            loss_si = F.smooth_l1_loss(pred_si, gt_si, beta=self.beta, reduction="none").squeeze(-1)
            loss_tr = F.smooth_l1_loss(pred_tr, gt_tr, beta=self.beta, reduction="none").squeeze(-1)

            vert_loss = (self.weight_si * loss_si + self.weight_taper * loss_tr) * valid.float()
            total_loss = total_loss + vert_loss.sum()
            total_valid += valid.float().sum()

        if total_valid > 0.0:
            return total_loss / total_valid
        return torch.tensor(0.0, device=device)

