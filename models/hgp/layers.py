"""Layers of HGP-SL, Hierarchical Graph Pooling with Structure Learning.

Zhang et al., "Hierarchical Graph Pooling with Structure Learning", AAAI 2020 (arXiv:1911.05954).
Adapted from the authors' code at https://github.com/cszhangzhen/HGP-SL. Differences from it:

- per-graph top-k node selection and edge filtering live here, because the PyTorch Geometric
  helpers the original imported have moved between PyG releases;
- torch_scatter and torch_sparse calls are replaced with their PyTorch Geometric and PyTorch
  equivalents, so the layers only need ``torch`` and ``torch_geometric``;
- the learned graph is built from edge lists instead of a dense (nodes x nodes) matrix.

All layers take the usual PyG inputs: node features ``x`` of shape (num_nodes, channels),
``edge_index`` of shape (2, num_edges), optional 1-D edge weights, and ``batch``, the graph id of
every node, sorted as PyG batching produces it.
"""

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, coalesce, scatter, softmax

from models.hgp.sparse_softmax import Sparsemax


def topk(score: Tensor, ratio: float, batch: Tensor) -> Tensor:
    """Indices of the ceil(ratio * n_i) highest-scoring nodes of every graph i, in ascending order.

    Ascending order keeps the nodes of each graph contiguous and in their original (residue) order.
    """
    num_nodes = scatter(torch.ones_like(batch), batch, dim=0, reduce="sum")
    k = torch.ceil(ratio * num_nodes.to(score.dtype)).long()

    # Order nodes by graph, and by decreasing score inside each graph.
    order = torch.argsort(score, descending=True)
    order = order[torch.sort(batch[order], stable=True).indices]
    start = num_nodes.cumsum(0) - num_nodes
    rank = torch.arange(score.numel(), device=score.device) - start[batch[order]]

    keep = order[rank < k[batch[order]]]
    return keep.sort().values


def filter_adj(edge_index: Tensor, edge_attr: Tensor | None, perm: Tensor, num_nodes: int):
    """Keep the edges between the nodes in ``perm`` and renumber them to positions in ``perm``."""
    new_id = edge_index.new_full((num_nodes,), -1)
    new_id[perm] = torch.arange(perm.numel(), device=perm.device)
    row, col = new_id[edge_index[0]], new_id[edge_index[1]]
    keep = (row >= 0) & (col >= 0)
    edge_attr = edge_attr[keep] if edge_attr is not None else None
    return torch.stack([row[keep], col[keep]]), edge_attr


def add_two_hop_edges(edge_index: Tensor, edge_attr: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    """Connect every pair of nodes that are two hops apart.

    Existing edges keep their weight; new edges get weight 0.
    """
    ones = torch.ones(edge_index.size(1), device=edge_index.device)
    adj = torch.sparse_coo_tensor(edge_index, ones, (num_nodes, num_nodes))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
        two_hop = torch.sparse.mm(adj, adj).coalesce().indices()

    edge_index = torch.cat([edge_index, two_hop], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr.new_full((two_hop.size(1),), float("inf"))])
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, reduce="min")
    edge_attr = edge_attr.masked_fill(torch.isinf(edge_attr), 0.0)
    return edge_index, edge_attr


def _normalised_weights(edge_index: Tensor, edge_weight: Tensor | None, num_nodes: int, dtype) -> Tensor:
    """Edge weights of D^-1/2 A D^-1/2, with D the weighted out-degree."""
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
    row, col = edge_index
    deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = deg_inv_sqrt.masked_fill(torch.isinf(deg_inv_sqrt), 0.0)
    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCN(MessagePassing):
    """Graph convolution D^-1/2 A D^-1/2 X W + b without added self-loops.

    Used after a pooling layer, whose learned graph already has self-loops and edge weights.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        x = x @ self.weight
        norm = _normalised_weights(edge_index, edge_weight, x.size(0), x.dtype)
        out = self.propagate(edge_index, x=x, norm=norm)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class NodeInformationScore(MessagePassing):
    """Node information of HGP-SL, (I - D^-1/2 A D^-1/2) X.

    A node that its neighbours describe well carries little information of its own, so the
    pooling layer drops the nodes with the smallest score.
    """

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        norm = _normalised_weights(edge_index, edge_weight, x.size(0), x.dtype)
        return x - self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class HGPSLPool(nn.Module):
    """HGP-SL pooling: keep the most informative nodes of every graph and learn new edges between them.

    Args:
        in_channels: node feature size.
        ratio: fraction of nodes kept in every graph.
        sample: learn edge weights only between kept nodes that were within a few hops of each other
            in the input graph (cheaper on large graphs) instead of between all pairs in a graph.
        sparse: normalise the learned edge weights with sparsemax instead of softmax.
        sl: learn the pooled graph's edges. If False, return the subgraph induced by the kept nodes.
        lamb: weight of the input edges added to the learned attention scores.
        negative_slope: slope of the LeakyReLU applied to the attention scores.

    Returns ``(x, edge_index, edge_weight, batch)`` of the pooled graphs.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.8,
        sample: bool = False,
        sparse: bool = False,
        sl: bool = True,
        lamb: float = 1.0,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.lamb = lamb
        self.negative_slope = negative_slope

        self.att = Parameter(torch.empty(1, 2 * in_channels))
        nn.init.xavier_uniform_(self.att)
        self.sparsemax = Sparsemax()
        self.information_score = NodeInformationScore()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None, batch: Tensor | None = None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        num_nodes = x.size(0)

        score = self.information_score(x, edge_index, edge_attr).abs().sum(dim=1)
        perm = topk(score, self.ratio, batch)
        x, batch = x[perm], batch[perm]
        induced_index, induced_attr = filter_adj(edge_index, edge_attr, perm, num_nodes)

        if not self.sl:
            return x, induced_index, induced_attr, batch

        if self.sample:
            # Candidate edges: pairs of kept nodes within a few hops in the input graph
            # (two rounds of two-hop augmentation, as in the original code).
            if edge_attr is None:
                edge_attr = torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device)
            hop_index, hop_attr = edge_index, edge_attr
            for _ in range(2):
                hop_index, hop_attr = add_two_hop_edges(hop_index, hop_attr, num_nodes)
            new_index, new_attr = filter_adj(hop_index, hop_attr, perm, num_nodes)
            new_index, new_attr = add_remaining_self_loops(new_index, new_attr, 0.0, x.size(0))
            weights = self._attention(x, new_index) + new_attr * self.lamb
            new_index, weights = coalesce(new_index, weights, x.size(0))
        else:
            # Candidate edges: all pairs of kept nodes in the same graph, self-pairs included.
            row, col = (batch.unsqueeze(0) == batch.unsqueeze(1)).nonzero(as_tuple=True)
            pair_index = torch.stack([row, col])
            if induced_attr is None:
                induced_attr = torch.ones(induced_index.size(1), dtype=x.dtype, device=x.device)
            new_index = torch.cat([pair_index, induced_index], dim=1)
            weights = torch.cat([self._attention(x, pair_index), induced_attr * self.lamb])
            new_index, weights = coalesce(new_index, weights, x.size(0), reduce="sum")

        row = new_index[0]
        if self.sparse:
            new_attr = self.sparsemax(weights, row)
        else:
            new_attr = softmax(weights, row, num_nodes=x.size(0))
        keep = new_attr != 0
        return x, new_index[:, keep], new_attr[keep], batch

    def _attention(self, x: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        scores = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
        return F.leaky_relu(scores, self.negative_slope)
