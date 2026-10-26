"""Graph classifier with hierarchical pooling (HGP-SL) for protein graphs.

The architecture is the HGP-SL model of Zhang et al. (2020): a GCN layer, two rounds of
"pool, then convolve" with HGPSLPool, a max+mean readout after each stage, and a three-layer
MLP on the sum of the three readouts.

For proteins, a graph is one domain: nodes are residues, edges join residues whose C-alpha atoms
are closer than a cutoff (8 Å in the PST data pipeline), and node features are per-residue
vectors such as per-residue PST embeddings. The training batch is a PyG ``Batch`` whose ``y``
holds one class index per graph. This repository has no script that builds such graphs yet; the
model has been checked on toy inputs only:

    >>> import torch
    >>> from torch_geometric.data import Batch, Data
    >>> def chain(n, dim=16):  # a path graph with random node features
    ...     i = torch.arange(n - 1)
    ...     edges = torch.cat([torch.stack([i, i + 1]), torch.stack([i + 1, i])], dim=1)
    ...     return Data(x=torch.randn(n, dim), edge_index=edges, y=torch.tensor([0]))
    >>> model = HierarchicalGNN(num_features=16, nhid=32, num_classes=10)
    >>> model(Batch.from_data_list([chain(12), chain(30)])).shape
    torch.Size([2, 10])
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from models.hgp.layers import GCN, HGPSLPool


class HierarchicalGNN(pl.LightningModule):
    """HGP-SL graph classifier as a LightningModule.

    The defaults follow the original HGP-SL code.

    Args:
        num_features: size of the node features.
        nhid: hidden size.
        num_classes: number of classes.
        pooling_ratio: fraction of nodes kept by each pooling layer.
        dropout_ratio: dropout in the MLP head.
        sample_neighbor: learn pooled edges only between nearby nodes (see HGPSLPool).
        sparse_attention: normalise pooled edge weights with sparsemax instead of softmax.
        structure_learning: learn the pooled graph's edges instead of keeping the induced subgraph.
        lamb: weight of the input edges in the structure learning scores.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
    """

    def __init__(
        self,
        num_features: int,
        nhid: int = 128,
        num_classes: int = 10,
        pooling_ratio: float = 0.5,
        dropout_ratio: float = 0.0,
        sample_neighbor: bool = True,
        sparse_attention: bool = True,
        structure_learning: bool = True,
        lamb: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        pool_args = {
            "ratio": pooling_ratio,
            "sample": sample_neighbor,
            "sparse": sparse_attention,
            "sl": structure_learning,
            "lamb": lamb,
        }
        self.conv1 = GCNConv(num_features, nhid)
        self.pool1 = HGPSLPool(nhid, **pool_args)
        self.conv2 = GCN(nhid, nhid)
        self.pool2 = HGPSLPool(nhid, **pool_args)
        self.conv3 = GCN(nhid, nhid)

        self.head = nn.Sequential(
            nn.Linear(2 * nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(nhid, nhid // 2),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(nhid // 2, num_classes),
        )

    def forward(self, data) -> Tensor:
        """Return class logits of shape (num_graphs, num_classes)."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        return self.head(F.relu(x1) + F.relu(x2) + F.relu(x3))

    def _step(self, batch, stage: str) -> Tensor:
        logits = self(batch)
        loss = F.cross_entropy(logits, batch.y)
        acc = (logits.argmax(dim=1) == batch.y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_acc", acc, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
