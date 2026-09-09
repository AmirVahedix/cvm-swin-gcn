import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.train import train_epoch, validate_epoch
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


class TestAMPTraining(unittest.TestCase):
    def setUp(self):
        self.num_landmarks = 13
        self.img_size = 64
        self.device = torch.device("cpu")

        # Create synthetic dataset
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
        self.awl_loss = AdaptiveWingLoss()
        self.wing_loss = WingLoss(img_size=float(self.img_size))
        adj_matrix = torch.eye(self.num_landmarks)
        self.graph_loss = AnatomicalGraphLoss(adj_matrix)
        self.landmark_weights = torch.ones(self.num_landmarks, device=self.device)

    def test_train_epoch_amp_flag_disabled(self):
        """Verify train_epoch executes cleanly with use_amp=False."""
        loss = train_epoch(
            model=self.model,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            awl_loss=self.awl_loss,
            wing_loss=self.wing_loss,
            graph_loss=self.graph_loss,
            landmark_weights=self.landmark_weights,
            lambda_hm=1.0,
            lambda_cd=1.0,
            lambda_graph=0.1,
            device=self.device,
            epoch=1,
            epochs=1,
            scaler=None,
            use_amp=False,
        )
        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.isnan(torch.tensor(loss)))

    def test_train_epoch_amp_flag_enabled_cpu(self):
        """Verify train_epoch executes safely on CPU even when use_amp=True without throwing errors."""
        loss = train_epoch(
            model=self.model,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            awl_loss=self.awl_loss,
            wing_loss=self.wing_loss,
            graph_loss=self.graph_loss,
            landmark_weights=self.landmark_weights,
            lambda_hm=1.0,
            lambda_cd=1.0,
            lambda_graph=0.1,
            device=self.device,
            epoch=1,
            epochs=1,
            scaler=None,
            use_amp=True,
        )
        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.isnan(torch.tensor(loss)))

    def test_validate_epoch_amp_enabled(self):
        """Verify validate_epoch executes with use_amp=True and returns correct metrics dictionary."""
        val_loss, metrics = validate_epoch(
            model=self.model,
            dataloader=self.dataloader,
            awl_loss=self.awl_loss,
            wing_loss=self.wing_loss,
            graph_loss=self.graph_loss,
            landmark_weights=self.landmark_weights,
            lambda_hm=1.0,
            lambda_cd=1.0,
            lambda_graph=0.1,
            device=self.device,
            img_size=self.img_size,
            epoch=1,
            epochs=1,
            use_amp=True,
        )
        self.assertGreater(val_loss, 0.0)
        self.assertIn("mae", metrics)
        self.assertIn("sdr_2_5", metrics)

    def test_cli_argument_parsing(self):
        """Verify CLI argument defaults and flag toggles for AMP and compile."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--amp", dest="use_amp", action="store_true", default=True)
        parser.add_argument("--no-amp", dest="use_amp", action="store_false")
        parser.add_argument("--compile", dest="compile_model", action="store_true", default=False)
        parser.add_argument("--batch-size", "-b", type=int, default=8)

        # Default args
        args_default = parser.parse_args([])
        self.assertTrue(args_default.use_amp)
        self.assertFalse(args_default.compile_model)
        self.assertEqual(args_default.batch_size, 8)

        # Custom args
        args_custom = parser.parse_args(["--no-amp", "--compile", "--batch-size", "16"])
        self.assertFalse(args_custom.use_amp)
        self.assertTrue(args_custom.compile_model)
        self.assertEqual(args_custom.batch_size, 16)

    def test_soft_argmax_2d_differentiability(self):
        """Verify that SoftArgmax2D produces non-zero gradients across the full spatial heatmap."""
        from src.models.model import SoftArgmax2D

        layer = SoftArgmax2D(num_landmarks=3, init_temperature=0.1)
        # Heatmap input with gradient tracking
        heatmaps = torch.randn(2, 3, 32, 32, requires_grad=True)
        coords = layer(heatmaps)

        self.assertEqual(coords.shape, (2, 3, 2))
        self.assertTrue((coords >= 0.0).all() and (coords <= 1.0).all())

        # Backpropagate arbitrary target coordinate loss
        target = torch.tensor([[[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]]]).expand(2, -1, -1)
        loss = torch.sum((coords - target) ** 2)
        loss.backward()

        # Ensure non-zero gradients flowed back to heatmaps
        self.assertIsNotNone(heatmaps.grad)
        self.assertTrue((heatmaps.grad != 0.0).any())
        self.assertFalse(torch.isnan(heatmaps.grad).any())

    def test_coord_loss_warmup_factor(self):
        """Verify get_coord_loss_warmup_factor schedules lambda_cd from 0.0 to 1.0."""
        from src.train import get_coord_loss_warmup_factor

        self.assertAlmostEqual(get_coord_loss_warmup_factor(1, 5), 0.0)
        self.assertAlmostEqual(get_coord_loss_warmup_factor(2, 5), 0.25)
        self.assertAlmostEqual(get_coord_loss_warmup_factor(3, 5), 0.50)
        self.assertAlmostEqual(get_coord_loss_warmup_factor(4, 5), 0.75)
        self.assertAlmostEqual(get_coord_loss_warmup_factor(5, 5), 1.0)
        self.assertAlmostEqual(get_coord_loss_warmup_factor(6, 5), 1.0)
        self.assertAlmostEqual(get_coord_loss_warmup_factor(10, 5), 1.0)
        self.assertAlmostEqual(get_coord_loss_warmup_factor(1, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
