import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.train import train_epoch, validate_epoch, get_coord_loss_warmup_factor
from src.models.losses import AdaptiveWingLoss, WingLoss, AnatomicalGraphLoss


class TinyModel(nn.Module):
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


class TestAblationNoAWL(unittest.TestCase):
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
        self.model = TinyModel(num_landmarks=self.num_landmarks, img_size=self.img_size).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.wing_loss = WingLoss(img_size=float(self.img_size))
        adj_matrix = torch.eye(self.num_landmarks)
        self.graph_loss = AnatomicalGraphLoss(adj_matrix)
        self.landmark_weights = torch.ones(self.num_landmarks, device=self.device)

    def test_coord_warmup_factor_with_zero_lambda_hm(self):
        """Verify get_coord_loss_warmup_factor returns 1.0 immediately when lambda_hm=0.0."""
        factor_epoch_1 = get_coord_loss_warmup_factor(epoch=1, warmup_epochs=5, lambda_hm=0.0)
        self.assertEqual(factor_epoch_1, 1.0)

        factor_standard = get_coord_loss_warmup_factor(epoch=1, warmup_epochs=5, lambda_hm=1.0)
        self.assertEqual(factor_standard, 0.0)

    def test_train_epoch_without_awl(self):
        """Verify train_epoch executes cleanly and computes loss without Adaptive Wing Loss."""
        loss = train_epoch(
            model=self.model,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            awl_loss=None,
            wing_loss=self.wing_loss,
            graph_loss=self.graph_loss,
            landmark_weights=self.landmark_weights,
            lambda_hm=0.0,
            lambda_cd=5.0,
            lambda_graph=1.0,
            device=self.device,
            epoch=1,
            epochs=1,
            scaler=None,
            use_amp=False,
        )
        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.isnan(torch.tensor(loss)))

    def test_validate_epoch_without_awl(self):
        """Verify validate_epoch executes cleanly and returns valid metrics without Adaptive Wing Loss."""
        val_loss, metrics = validate_epoch(
            model=self.model,
            dataloader=self.dataloader,
            awl_loss=None,
            wing_loss=self.wing_loss,
            graph_loss=self.graph_loss,
            landmark_weights=self.landmark_weights,
            lambda_hm=0.0,
            lambda_cd=5.0,
            lambda_graph=1.0,
            device=self.device,
            img_size=self.img_size,
            epoch=1,
            epochs=1,
            use_amp=False,
        )
        self.assertGreater(val_loss, 0.0)
        self.assertFalse(torch.isnan(torch.tensor(val_loss)))
        self.assertIn("mre_mm", metrics)
        self.assertIn("sdr_2_0mm", metrics)

    def test_end_to_end_gradient_flow_without_awl(self):
        """Verify gradients flow all the way to convolutional feature layers without explicit heatmap loss."""
        self.optimizer.zero_grad()
        images = torch.randn(2, 3, self.img_size, self.img_size)
        gt_coords = torch.rand(2, self.num_landmarks, 2)

        pred_heatmaps, pred_coords = self.model(images)
        loss_cd = self.wing_loss(pred_coords, gt_coords, landmark_weights=self.landmark_weights)
        loss_g = self.graph_loss(pred_coords, gt_coords)
        total_loss = 5.0 * loss_cd + 1.0 * loss_g
        total_loss.backward()

        self.assertIsNotNone(self.model.conv.weight.grad)
        self.assertGreater(torch.norm(self.model.conv.weight.grad).item(), 0.0)

    def test_end_to_end_gradient_flow_with_awl(self):
        """Verify full multi-task gradient flow (AWL + WingLoss + GraphLoss) without gradient loss."""
        self.optimizer.zero_grad()
        images = torch.randn(2, 3, self.img_size, self.img_size)
        gt_heatmaps = torch.rand(2, self.num_landmarks, self.img_size, self.img_size)
        gt_coords = torch.rand(2, self.num_landmarks, 2)

        pred_heatmaps, pred_coords = self.model(images)
        awl = AdaptiveWingLoss()
        loss_hm = awl(pred_heatmaps, gt_heatmaps, landmark_weights=self.landmark_weights)
        loss_cd = self.wing_loss(pred_coords, gt_coords, landmark_weights=self.landmark_weights)
        loss_g = self.graph_loss(pred_coords, gt_coords)

        total_loss = 1.0 * loss_hm + 5.0 * loss_cd + 1.0 * loss_g
        total_loss.backward()

        # Gradients must successfully propagate to both conv and fc layers
        self.assertIsNotNone(self.model.conv.weight.grad)
        self.assertGreater(torch.norm(self.model.conv.weight.grad).item(), 0.0)
        self.assertFalse(torch.isnan(self.model.conv.weight.grad).any())

        self.assertIsNotNone(self.model.fc.weight.grad)
        self.assertGreater(torch.norm(self.model.fc.weight.grad).item(), 0.0)
        self.assertFalse(torch.isnan(self.model.fc.weight.grad).any())

    def test_train_epoch_with_awl_enabled(self):
        """Verify train_epoch computes combined loss with awl_loss active and gradients intact."""
        awl = AdaptiveWingLoss()
        loss = train_epoch(
            model=self.model,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            awl_loss=awl,
            wing_loss=self.wing_loss,
            graph_loss=self.graph_loss,
            landmark_weights=self.landmark_weights,
            lambda_hm=1.0,
            lambda_cd=5.0,
            lambda_graph=1.0,
            device=self.device,
            epoch=1,
            epochs=1,
            scaler=None,
            use_amp=False,
        )
        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.isnan(torch.tensor(loss)))


if __name__ == "__main__":
    unittest.main()
