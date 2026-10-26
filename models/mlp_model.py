"""MLP classifier on fixed-size protein embeddings.

The input is one vector per protein domain, for example the mean of PST's per-residue
representations (1280 values for ``pst_t33_so``, as written by ``scripts/extract_embeddings.py``).
Batches are ``(embeddings, labels)`` pairs:

    >>> import torch
    >>> model = MLPClassifier(input_dim=1280, num_classes=10)
    >>> model(torch.randn(4, 1280)).shape
    torch.Size([4, 10])
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MLPClassifier(pl.LightningModule):
    """Three linear layers with ReLU and dropout (dropout is also applied to the input embedding)."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def _step(self, batch, stage: str) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
