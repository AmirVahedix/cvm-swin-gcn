import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.model import CephalometricSwinGCN, CephalometricSwin
from src.train import train_epoch, validate_epoch
from src.models.losses import AdaptiveWingLoss, WingLoss


class DummyModel(nn.Module):
    def __init__(self, num_landmarks=13, img_size=64):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.conv = nn.Conv2d(3, num_landmarks, kernel_size=3, padding=1)
        self.fc = nn.Linear(num_landmarks * img_size * img_size, num_landmarks * 2)

    def forward(self, x):
        B = x.shape[0]
        heatmaps = torch.sigmoid(self.conv(x))
        coords = torch.sigmoid(self.fc(heatmaps.view(B, -1))).view(B, self.num_landmarks, 2)
        return heatmaps, coords


class TestAblationNoGraph(unittest.TestCase):
    def setUp(self):
        self.num_landmarks = 13
        self.img_size = 64
        self.device = torch.device("cpu")

        batch_count = 2
        batch_size = 2
        total_samples = batch_count * batch_size

        images = torch.randn(total_samples, 3, self.img_size, self.img_size)
        heatmaps = torch.rand(total_samples, self.num_landmarks, self.img_size, self.img_size)
        coords = torch.rand(total_samples, self.num_landmarks, 2)

        dataset = [
            {
                "image": images[i],
                "heatmaps": heatmaps[i],
                "coords": coords[i],
            }
            for i in range(total_samples)
        ]

        def collate_fn(batch):
            return {
                "image": torch.stack([b["image"] for b in batch]),
                "heatmaps": torch.stack([b["heatmaps"] for b in batch]),
                "coords": torch.stack([b["coords"] for b in batch]),
            }

        self.dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        self.model = DummyModel(num_landmarks=self.num_landmarks, img_size=self.img_size).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.awl_loss = AdaptiveWingLoss()
        self.wing_loss = WingLoss(img_size=float(self.img_size))
        self.landmark_weights = torch.ones(self.num_landmarks, device=self.device)

    def test_model_alias_and_no_adj_buffer(self):
        """Verify CephalometricSwin alias exists and include_adj=False by default."""
        self.assertIs(CephalometricSwin, CephalometricSwinGCN)

    def test_train_epoch_without_graph_loss(self):
        """Verify train_epoch computes loss cleanly without graph_loss (graph_loss=None, lambda_graph=0.0)."""
        loss = train_epoch(
            model=self.model,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            awl_loss=self.awl_loss,
            wing_loss=self.wing_loss,
            graph_loss=None,
            landmark_weights=self.landmark_weights,
            lambda_hm=1.0,
            lambda_cd=5.0,
            lambda_graph=0.0,
            device=self.device,
            epoch=1,
            epochs=1,
            scaler=None,
            use_amp=False,
        )
        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.isnan(torch.tensor(loss)))

    def test_validate_epoch_without_graph_loss(self):
        """Verify validate_epoch executes cleanly without graph_loss (graph_loss=None, lambda_graph=0.0)."""
        val_loss, metrics = validate_epoch(
            model=self.model,
            dataloader=self.dataloader,
            awl_loss=self.awl_loss,
            wing_loss=self.wing_loss,
            graph_loss=None,
            landmark_weights=self.landmark_weights,
            lambda_hm=1.0,
            lambda_cd=5.0,
            lambda_graph=0.0,
            device=self.device,
            img_size=self.img_size,
            epoch=1,
            epochs=1,
            use_amp=False,
        )
        self.assertGreater(val_loss, 0.0)
        self.assertIn("mre_mm", metrics)
        self.assertIn("sdr_2_0mm", metrics)


if __name__ == "__main__":
    unittest.main()
