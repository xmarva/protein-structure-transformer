import torch
import pytorch_lightning as pl
from torch import nn

class MLPTrainer(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLPTrainer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch  # Unpack all three elements
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch  # Unpack all three elements
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch  # Unpack all three elements
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': lr_scheduler, 'monitor': 'val_loss'}
        }
