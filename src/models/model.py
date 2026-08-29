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


class LocalWindowSoftArgmax2D(nn.Module):
    """
    Differentiable Local-Window Soft-Argmax 2D coordinate extraction.
    Computes spatial argmax to locate the integer peak, extracts a localized
    (2R+1) x (2R+1) neighborhood window around the peak, and performs temperature-scaled
    expectation within the window.
    
    This eliminates 100% of global background noise/bias on large spatial heatmaps (640x640)
    while providing high-precision sub-pixel differentiability.
    """
    def __init__(self, num_landmarks: int = 13, radius: int = 7, init_temperature: float = 0.1):
        super().__init__()
        import math
        self.num_landmarks = num_landmarks
        self.radius = radius
        init_log_temp = torch.full((1, num_landmarks, 1, 1), math.log(init_temperature))
        self.log_temperature = nn.Parameter(init_log_temp)

        # Coordinate grid offsets relative to the window center [-R, R]
        dy = torch.arange(-radius, radius + 1, dtype=torch.float32)
        dx = torch.arange(-radius, radius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(dy, dx, indexing="ij")
        self.register_buffer("grid_x", grid_x.unsqueeze(0).unsqueeze(0))  # [1, 1, 2R+1, 2R+1]
        self.register_buffer("grid_y", grid_y.unsqueeze(0).unsqueeze(0))  # [1, 1, 2R+1, 2R+1]

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmaps: Tensor of shape [B, N, H, W]
        Returns:
            coords: Tensor of shape [B, N, 2] in normalized scale [0, 1]
        """
        B, N, H, W = heatmaps.shape
        R = self.radius

        # 1. Integer Peak Location via Spatial Argmax
        flat_hm = heatmaps.view(B, N, -1)
        max_idx = torch.argmax(flat_hm, dim=-1)  # [B, N]
        py = max_idx // W                        # [B, N]
        px = max_idx % W                         # [B, N]

        # 2. Extract localized patches with replicate padding to safely handle image boundaries
        padded = F.pad(heatmaps, (R, R, R, R), mode="replicate")  # [B, N, H + 2R, W + 2R]
        W_pad = W + 2 * R

        cy = py + R  # Center Y in padded coordinate system
        cx = px + R  # Center X in padded coordinate system

        # Build absolute index grid for gathering local patches across [B, N, 2R+1, 2R+1]
        sample_y = cy.unsqueeze(-1).unsqueeze(-1) + self.grid_y.long()  # [B, N, 2R+1, 2R+1]
        sample_x = cx.unsqueeze(-1).unsqueeze(-1) + self.grid_x.long()  # [B, N, 2R+1, 2R+1]
        flat_patch_idx = (sample_y * W_pad + sample_x).view(B, N, -1)  # [B, N, (2R+1)^2]

        flat_padded = padded.view(B, N, -1)
        patches = torch.gather(flat_padded, dim=2, index=flat_patch_idx).view(B, N, 2 * R + 1, 2 * R + 1)

        # 3. Temperature-scaled softmax expectation within local patch
        temperature = torch.exp(self.log_temperature).clamp(min=1e-3, max=1.0)
        scaled_patches = patches / temperature
        patch_softmax = F.softmax(scaled_patches.view(B, N, -1), dim=-1).view(B, N, 2 * R + 1, 2 * R + 1)

        # Sub-pixel coordinate offsets in [-R, R]
        delta_x = torch.sum(patch_softmax * self.grid_x, dim=(-2, -1))  # [B, N]
        delta_y = torch.sum(patch_softmax * self.grid_y, dim=(-2, -1))  # [B, N]

        # Continuous sub-pixel coordinates in [0, 1] normalized space
        coord_x = (px.float() + delta_x) / float(W)
        coord_y = (py.float() + delta_y) / float(H)

        return torch.stack([coord_x, coord_y], dim=-1).clamp(0.0, 1.0)  # [B, N, 2]


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
    Upgraded Cephalometric Swin-GAT Network featuring:
    1. Multi-scale Swin Backbone with U-Net feature pyramid skip-connections.
    2. Local Windowed Soft-Argmax heatmap coordinate extraction (zero background bias, sharp sub-pixel precision).
    3. Continuous landmark feature sampling via bilinear grid_sample on 160x160 pyramid features.
    4. Dynamic Multi-Head Graph Attention (GAT) residual offset head for anatomical graph alignment.
    """
    def __init__(self, num_landmarks: int = 13, pretrained: bool = True, window_radius: int = 7):
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

        # Local-Window Soft-Argmax layer with learnable per-landmark temperature
        self.soft_argmax = LocalWindowSoftArgmax2D(
            num_landmarks=num_landmarks, radius=window_radius, init_temperature=0.1
        )

        # Dynamic Multi-Head Graph Attention (GAT) Structural Residual Refinement Head
        self.gat1 = GraphAttentionLayer(128, 128, heads=4, concat=True)
        self.gat2 = GraphAttentionLayer(128, 64, heads=4, concat=True)
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

        # 1. Local-Window Soft-Argmax Sub-Pixel Base Coordinates
        base_coords = self.soft_argmax(heatmaps)  # [B, 13, 2]

        # 2. Extract High-Resolution Landmark Features via Bilinear Grid Sampling on d1 (160x160)
        # grid_sample expects coordinates in [-1, 1] range: (x, y) -> 2 * norm - 1
        B, N, _ = base_coords.shape
        grid_sample_coords = (base_coords * 2.0 - 1.0).unsqueeze(2)  # [B, 13, 1, 2]
        sampled_feats = F.grid_sample(
            d1, grid_sample_coords, mode="bilinear", padding_mode="border", align_corners=True
        )  # [B, 128, 13, 1]
        node_features = sampled_feats.squeeze(-1).permute(0, 2, 1).contiguous()  # [B, 13, 128]

        # 3. Dynamic GAT Graph Residual Coordinate Offset Prediction
        gat_feat = F.elu(self.gat1(node_features, self.adj_matrix))
        gat_feat = F.elu(self.gat2(gat_feat, self.adj_matrix))
        coord_offset = 0.03 * torch.tanh(self.offset_head(gat_feat))  # Bounded offset [-0.03, 0.03]

        # Final Refined Coordinates
        coords = torch.clamp(base_coords + coord_offset, 0.0, 1.0)

        return heatmaps, coords


