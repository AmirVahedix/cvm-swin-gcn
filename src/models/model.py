import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GraphConvolution(nn.Module):
    """
    Standard Graph Convolutional layer.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        return torch.matmul(adj, support)


class SoftArgmax2D(nn.Module):
    """
    Differentiable Soft-Argmax 2D coordinate extraction with learnable
    per-landmark temperature scaling for sharp sub-pixel localization.
    """
    def __init__(self, num_landmarks: int = 13, init_temperature: float = 0.1):
        super().__init__()
        import math
        init_log_temp = torch.full((1, num_landmarks, 1, 1), math.log(init_temperature))
        self.log_temperature = nn.Parameter(init_log_temp)

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmaps: Tensor of shape [B, N, H, W]
        Returns:
            coords: Tensor of shape [B, N, 2] in normalized scale [0, 1]
        """
        B, N, H, W = heatmaps.shape
        # Clamp temperature to range [0.001, 1.0] for numerical stability
        temperature = torch.exp(self.log_temperature).clamp(min=1e-3, max=1.0)

        scaled_hm = heatmaps / temperature
        flat_hm = scaled_hm.view(B, N, -1)
        softmax_hm = F.softmax(flat_hm, dim=-1).view(B, N, H, W)

        device = heatmaps.device
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, H, device=device),
            torch.linspace(0.0, 1.0, W, device=device),
            indexing="ij",
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        exp_x = torch.sum(softmax_hm * grid_x, dim=(-2, -1))
        exp_y = torch.sum(softmax_hm * grid_y, dim=(-2, -1))

        return torch.stack([exp_x, exp_y], dim=-1)  # [B, N, 2]



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
    2. Continuous Soft-Argmax heatmap coordinate extraction.
    3. High-resolution landmark feature attention pooling for GCN structural modeling.
    4. GCN structural residual offset head for anatomical graph alignment.
    """
    def __init__(self, num_landmarks: int = 13, pretrained: bool = True):
        super().__init__()
        self.num_landmarks = num_landmarks

        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            img_size=(640, 640),
        )

        # U-Net Pyramid Decoder Blocks
        # Swin channels: Stage 0: 128 (160x160), Stage 1: 256 (80x80), Stage 2: 512 (40x40), Stage 3: 1024 (20x20)
        self.up3 = UNetUpBlock(1024, 512, 512)  # 20x20 -> 40x40
        self.up2 = UNetUpBlock(512, 256, 256)   # 40x40 -> 80x80
        self.up1 = UNetUpBlock(256, 128, 128)   # 80x80 -> 160x160
        self.up0 = UNetUpBlock(128, 0, 64)      # 160x160 -> 320x320
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 320x320 -> 640x640
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_landmarks, kernel_size=1),
        )

        # Soft-Argmax layer with learnable per-landmark temperature
        self.soft_argmax = SoftArgmax2D(num_landmarks=num_landmarks, init_temperature=0.1)

        # GCN Structural Residual Refinement Head
        self.gcn1 = GraphConvolution(128, 128)
        self.gcn2 = GraphConvolution(128, 64)
        self.offset_head = nn.Linear(64, 2)

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

        # Row normalize
        rowsum = adj.sum(dim=1, keepdim=True)
        adj = adj / rowsum
        return adj

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        # Permute (B, H, W, C) -> (B, C, H, W) for timm Swin outputs
        f0 = features[0].permute(0, 3, 1, 2).contiguous()  # 128, 160x160
        f1 = features[1].permute(0, 3, 1, 2).contiguous()  # 256, 80x80
        f2 = features[2].permute(0, 3, 1, 2).contiguous()  # 512, 40x40
        f3 = features[3].permute(0, 3, 1, 2).contiguous()  # 1024, 20x20

        # U-Net Pyramid Decoding with Skip Connections
        d3 = self.up3(f3, f2)  # 512, 40x40
        d2 = self.up2(d3, f1)  # 256, 80x80
        d1 = self.up1(d2, f0)  # 128, 160x160
        d0 = self.up0(d1, None)  # 64, 320x320
        heatmaps = self.up_final(d0)  # 13, 640x640

        # 1. Soft-Argmax Differentiable Base Coordinate Extraction
        soft_coords = self.soft_argmax(heatmaps)  # [B, 13, 2]

        # 2. Extract High-Resolution Landmark Features (from 160x160 feature level)
        B, N, H, W = heatmaps.shape
        flat_attn = F.softmax(heatmaps.flatten(2), dim=-1).view(B, N, H, W)
        attn_160 = F.adaptive_avg_pool2d(flat_attn, d1.shape[-2:])
        attn_sum = attn_160.flatten(2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_160 = (attn_160.flatten(2) / attn_sum).view_as(attn_160)

        node_features = torch.einsum("bnij,bcij->bnc", attn_160, d1)  # [B, 13, 128]

        # 3. GCN Graph Residual Coordinate Offset Prediction
        gcn_feat = F.relu(self.gcn1(node_features, self.adj_matrix))
        gcn_feat = F.relu(self.gcn2(gcn_feat, self.adj_matrix))
        coord_offset = 0.05 * torch.tanh(self.offset_head(gcn_feat))  # Bounded offset [-0.05, 0.05]

        # Final Refined Coordinates
        coords = torch.clamp(soft_coords + coord_offset, 0.0, 1.0)

        return heatmaps, coords
