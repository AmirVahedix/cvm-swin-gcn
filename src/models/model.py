import torch.nn as nn
import torch
import timm
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        return torch.matmul(adj, support)


class CephalometricSwinGCN(nn.Module):
    def __init__(self, num_landmarks=13):
        super().__init__()

        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            features_only=True,
            img_size=(640, 640),
        )

        # Note: Adjust these transposed convolutions based on the exact
        # spatial dimensions of your chosen Swin backbone's output to reach 640x640
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_landmarks, kernel_size=1),
        )

        self.gcn1 = GraphConvolution(1024, 256)
        self.gcn2 = GraphConvolution(256, 128)
        self.coord_head = nn.Linear(128, 2)

        self.register_buffer("adj_matrix", self._build_adjacency())

    def _build_adjacency(self):
        adj = torch.eye(13)
        # C2 (0,1,2)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[1, 2] = adj[2, 1] = 1.0
        # C3 (3,4,5,6,7)
        adj[3, 4] = adj[4, 3] = 1.0
        adj[4, 5] = adj[5, 4] = 1.0
        adj[5, 6] = adj[6, 5] = 1.0
        adj[6, 7] = adj[7, 6] = 1.0
        # C4 (8,9,10,11,12)
        adj[8, 9] = adj[9, 8] = 1.0
        adj[9, 10] = adj[10, 9] = 1.0
        adj[10, 11] = adj[11, 10] = 1.0
        adj[11, 12] = adj[12, 11] = 1.0

        # Row normalize
        rowsum = adj.sum(dim=1, keepdim=True)
        adj = adj / rowsum
        return adj

    def forward(self, x):
        features = self.backbone(x)
        deepest_features = features[-1]  # Shape: [8, 20, 20, 1024]

        # 1. Fix channel ordering: (B, H, W, C) -> (B, C, H, W)
        deepest_features = deepest_features.permute(0, 3, 1, 2).contiguous()

        # 2. Generate Heatmaps
        heatmaps = self.heatmap_head(deepest_features)  # Shape: [8, 13, 640, 640]
        B, N, H, W = heatmaps.shape

        # 3. SAFE ATTENTION CALCULATION:
        # Flatten only the spatial dimensions (H and W) into a single dimension (H*W)
        # Shape changes: [8, 13, 640, 640] -> [8, 13, 409600]
        flat_heatmaps = heatmaps.flatten(2)

        # Apply Softmax over the flattened spatial dimension (dim=2)
        flat_attention = F.softmax(flat_heatmaps, dim=2)

        # Reshape cleanly back to spatial layout: [8, 13, 640, 640]
        attention = flat_attention.view(B, N, H, W)

        # 4. Pool attention map down to match backbone spatial dimensions (20, 20)
        attn_down = F.adaptive_avg_pool2d(
            attention, deepest_features.shape[-2:]
        )  # Shape: [8, 13, 20, 20]

        # 5. Extract landmark features via Einstein summation
        node_features = torch.einsum(
            "bnij,bcij->bnc", attn_down, deepest_features
        )  # Shape: [8, 13, 1024]

        # 6. Graph Layers & Coordinate Prediction
        x_gcn = F.relu(self.gcn1(node_features, self.adj_matrix))
        x_gcn = F.relu(self.gcn2(x_gcn, self.adj_matrix))
        coords = torch.sigmoid(self.coord_head(x_gcn))

        return heatmaps, coords
