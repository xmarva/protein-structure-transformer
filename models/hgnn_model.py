import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from models.hgp.layers import GCN, HGPSLPool
from torch_geometric.nn import GCNConv

class HierarchicalGNN(pl.LightningModule):
    def __init__(self, num_features, nhid, num_classes, pooling_ratio, dropout_ratio, sample_neighbor, sparse_attention, structure_learning, lamb):
        super(HierarchicalGNN, self).__init__()
        
        # Store the arguments
        self.save_hyperparameters()

        # Define the layers
        self.conv1 = GCNConv(num_features, nhid)
        self.conv2 = GCN(nhid, nhid)
        self.conv3 = GCN(nhid, nhid)

        self.pool1 = HGPSLPool(nhid, pooling_ratio, sample_neighbor, sparse_attention, structure_learning, lamb)
        self.pool2 = HGPSLPool(nhid, pooling_ratio, sample_neighbor, sparse_attention, structure_learning, lamb)

        self.lin1 = nn.Linear(nhid * 2, nhid)
        self.lin2 = nn.Linear(nhid, nhid // 2)
        self.lin3 = nn.Linear(nhid // 2, num_classes)

        self.dropout_ratio = dropout_ratio
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin3(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
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