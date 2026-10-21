import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Network (GAT) layer.
    Computes dynamic feature-dependent pairwise attention coefficients alpha_ij
    between connected anatomical landmarks.
    """
    def __init__(self, in_features: int, out_features: int, heads: int = 4, concat: bool = True, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat

        if concat:
            assert out_features % heads == 0, "out_features must be divisible by heads when concat=True"
            self.head_dim = out_features // heads
        else:
            self.head_dim = out_features

        self.linear = nn.Linear(in_features, self.head_dim * heads, bias=False)
        self.attn_src = nn.Parameter(torch.zeros(1, heads, 1, self.head_dim))
        self.attn_dst = nn.Parameter(torch.zeros(1, heads, 1, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.attn_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.attn_dst.data, gain=1.414)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, N, in_features]
            adj: Adjacency structure mask [N, N] (1.0 for connected, 0.0 for disconnected)
        Returns:
            out: Tensor of shape [B, N, out_features]
        """
        B, N, _ = x.shape
        # Linear projection: [B, N, heads * head_dim] -> [B, heads, N, head_dim]
        h = self.linear(x).view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores for source and target nodes
        attn_src = torch.matmul(h, self.attn_src.transpose(-1, -2))  # [B, heads, N, 1]
        attn_dst = torch.matmul(h, self.attn_dst.transpose(-1, -2))  # [B, heads, N, 1]

        # Pairwise attention scores e_ij: [B, heads, N, N]
        e = self.leaky_relu(attn_src + attn_dst.transpose(-1, -2))

        # Mask out non-connected pairs (where adj == 0) with a large negative value
        mask = (adj == 0).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        e = e.masked_fill(mask, -1e9)

        # Softmax over neighborhood
        alpha = F.softmax(e, dim=-1)  # [B, heads, N, N]
        alpha = self.dropout(alpha)

        # Aggregate neighbor features weighted by dynamic attention alpha: [B, heads, N, head_dim]
        out = torch.matmul(alpha, h)

        if self.concat:
            # Concatenate heads: [B, N, heads * head_dim] = [B, N, out_features]
            out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.heads * self.head_dim)
        else:
            # Average heads: [B, N, head_dim]
            out = out.mean(dim=1)

        return out


class SoftArgmax2D(nn.Module):
    """
    Differentiable 2D Soft-Argmax coordinate regression layer.
    Computes spatial expectation over normalized coordinate grids [0, 1] x [0, 1]
    weighted by a temperature-scaled spatial softmax over the full heatmap H x W.
    Guarantees global differentiability so coordinate losses (WingLoss, GraphLoss)
    can pull heatmap peaks across the entire image space.

    Operates directly on unnormalized heatmap logits (or converts sigmoid heatmaps
    back to logits) to guarantee wide dynamic range, eliminating the background
    sum centroid bias that occurs when softmax is applied to post-sigmoid [0, 1] values.
    """
    def __init__(self, num_landmarks: int = 13, init_temperature: float = 0.1):
        super().__init__()
        import math
        self.num_landmarks = num_landmarks
        init_log_temp = torch.full((1, num_landmarks, 1, 1), math.log(init_temperature))
        self.log_temperature = nn.Parameter(init_log_temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, N, H, W] containing unnormalized logits or heatmaps
        Returns:
            coords: Tensor of shape [B, N, 2] in normalized scale [0, 1]
        """
        B, N, H, W = x.shape
        device = x.device

        # If input has already been compressed to [0, 1] (Gaussian heatmaps), convert to log-space
        # to ensure wide dynamic range and preserve parabolic peak shape: log(exp(-d^2/(2*s^2))) = -d^2/(2*s^2)
        if (x >= 0.0).all() and (x <= 1.0).all() and x.max() <= 1.0 and x.min() >= 0.0:
            x_logits = torch.log(x.clamp(min=1e-8))
        else:
            x_logits = x

        # Temperature scaling with numerical stability clamp
        temperature = torch.exp(self.log_temperature).clamp(min=1e-3, max=1.0)
        scaled_hm = x_logits / temperature

        # Spatial softmax across the full (H, W) spatial domain
        flat_hm = scaled_hm.view(B, N, -1)
        softmax_hm = F.softmax(flat_hm, dim=-1).view(B, N, H, W)

        # Build coordinate grid matching standard pixel scale normalization convention:
        # pixel index i corresponds to normalized coordinate i / W (and i / H).
        # When multiplied by W (e.g. 640) during evaluation/inference, this recovers pixel index i exactly.
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=x.dtype) / float(H),
            torch.arange(W, device=device, dtype=x.dtype) / float(W),
            indexing="ij",
        )
        grid_x = grid_x.view(1, 1, H, W)
        grid_y = grid_y.view(1, 1, H, W)

        exp_x = torch.sum(softmax_hm * grid_x, dim=(-2, -1))
        exp_y = torch.sum(softmax_hm * grid_y, dim=(-2, -1))

        return torch.stack([exp_x, exp_y], dim=-1).clamp(0.0, 1.0)


class UNetUpBlock(nn.Module):
    """
    U-Net Decoder Upsampling Block with Skip Connections.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        total_in = out_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CephalometricSwinGCN(nn.Module):
    """
    Upgraded Cephalometric Swin-GCN Network featuring:
    1. Multi-scale Swin Backbone with U-Net feature pyramid skip-connections.
       - Backbone resolution is strictly aligned to a multiple of 224 (32 * window_size_7),
         eliminating internal zero-padding boundary distortions across all shifted window stages.
    2. Sigmoid normalized heatmaps [0, 1] matching ground-truth Gaussian targets.
    3. Differentiable Soft-Argmax coordinate regression directly from unnormalized logits
       with learnable per-landmark temperature, eliminating background centroid drift.
    4. Graph Loss structural alignment via anatomical adjacency buffering.
    """
    def __init__(
        self,
        num_landmarks: int = 13,
        pretrained: bool = True,
        img_size: int = 640,
        init_temperature: float = 0.1,
        window_radius: int | None = None,  # Kept for backward compatibility
        use_stem_skip: bool = False,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.img_size = img_size
        self.use_stem_skip = use_stem_skip

        # Unit = 32 * window_size_7 = 224. Align backbone img_size to the nearest multiple of 224
        # (e.g. 640 -> 672, 672 -> 672) to ensure all 4 Swin stages have dimensions evenly divisible by 7.
        self.window_unit = 224
        self.aligned_size = ((img_size + self.window_unit - 1) // self.window_unit) * self.window_unit

        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            img_size=(self.aligned_size, self.aligned_size),
            strict_img_size=False,
        )

        # Optional high-resolution convolutional stem for thin cortical edges at stride 2 and stride 1
        if self.use_stem_skip:
            self.stem_half = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.up0 = UNetUpBlock(128, 32, 64)
            self.stem_full = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            )
            self.up_final = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.head_conv = nn.Sequential(
                nn.Conv2d(32 + 16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, num_landmarks, kernel_size=1),
            )
        if not self.use_stem_skip:
            self.up3 = UNetUpBlock(1024, 512, 512)
            self.up2 = UNetUpBlock(512, 256, 256)
            self.up1 = UNetUpBlock(256, 128, 128)
            self.up0 = UNetUpBlock(128, 0, 64)
            self.up_final = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, num_landmarks, kernel_size=1),
            )
        else:
            self.up3 = UNetUpBlock(1024, 512, 512)
            self.up2 = UNetUpBlock(512, 256, 256)
            self.up1 = UNetUpBlock(256, 128, 128)

        # Global Differentiable Soft-Argmax layer with learnable per-landmark temperature
        self.soft_argmax = SoftArgmax2D(
            num_landmarks=num_landmarks, init_temperature=init_temperature
        )

        self.register_buffer("adj_matrix", self._build_adjacency())

    def _build_adjacency(self) -> torch.Tensor:
        adj = torch.eye(13)
        # C2 (0: C2_PI, 1: C2_IC, 2: C2_AI)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[1, 2] = adj[2, 1] = 1.0

        # C3 (3: C3_PS, 4: C3_AS, 5: C3_PI, 6: C3_IC, 7: C3_AI)
        adj[3, 4] = adj[4, 3] = 1.0  # Superior edge (PS <-> AS)
        adj[4, 7] = adj[7, 4] = 1.0  # Anterior edge (AS <-> AI)
        adj[5, 6] = adj[6, 5] = 1.0  # Inferior edge (PI <-> IC)
        adj[6, 7] = adj[7, 6] = 1.0  # Inferior edge (IC <-> AI)
        adj[3, 5] = adj[5, 3] = 1.0  # Posterior edge (PS <-> PI)

        # C4 (8: C4_PS, 9: C4_AS, 10: C4_PI, 11: C4_IC, 12: C4_AI)
        adj[8, 9] = adj[9, 8] = 1.0    # Superior edge (PS <-> AS)
        adj[9, 12] = adj[12, 9] = 1.0  # Anterior edge (AS <-> AI)
        adj[10, 11] = adj[11, 10] = 1.0 # Inferior edge (PI <-> IC)
        adj[11, 12] = adj[12, 11] = 1.0 # Inferior edge (IC <-> AI)
        adj[8, 10] = adj[10, 8] = 1.0  # Posterior edge (PS <-> PI)

        # Inter-vertebral connections (Spinal column alignment)
        adj[0, 3] = adj[3, 0] = 1.0  # Posterior spine: C2_PI <-> C3_PS
        adj[5, 8] = adj[8, 5] = 1.0  # Posterior spine: C3_PI <-> C4_PS
        adj[2, 4] = adj[4, 2] = 1.0  # Anterior spine: C2_AI <-> C3_AS
        adj[7, 9] = adj[9, 7] = 1.0  # Anterior spine: C3_AI <-> C4_AS

        return adj

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        pad_h = (self.window_unit - H % self.window_unit) % self.window_unit
        pad_w = (self.window_unit - W % self.window_unit) % self.window_unit

        # Pad input to nearest multiple of 224 using replicate padding to prevent boundary edge artifacts
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_in = x

        features = self.backbone(x_in)
        # Permute (B, H, W, C) -> (B, C, H, W) for timm Swin outputs
        f0 = features[0].permute(0, 3, 1, 2).contiguous()
        f1 = features[1].permute(0, 3, 1, 2).contiguous()
        f2 = features[2].permute(0, 3, 1, 2).contiguous()
        f3 = features[3].permute(0, 3, 1, 2).contiguous()

        # U-Net Pyramid Decoding with Skip Connections
        d3 = self.up3(f3, f2)
        d2 = self.up2(d3, f1)
        d1 = self.up1(d2, f0)
        if self.use_stem_skip:
            s_half = self.stem_half(x_in)
            d0 = self.up0(d1, s_half)
            d_final = self.up_final(d0)
            s_full = self.stem_full(x_in)
            logits = self.head_conv(torch.cat([d_final, s_full], dim=1))
        else:
            d0 = self.up0(d1, None)
            logits = self.up_final(d0)

        # Slice back to original resolution (H, W) if input was padded
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :H, :W].contiguous()
        elif logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        # Heatmap Normalization: strictly bound predictions to [0, 1] matching ground-truth Gaussian targets
        heatmaps = torch.sigmoid(logits)

        # Global Differentiable Soft-Argmax Coordinates directly from unnormalized logits
        coords = self.soft_argmax(logits)  # [B, 13, 2] in [0, 1]

        return heatmaps, coords


