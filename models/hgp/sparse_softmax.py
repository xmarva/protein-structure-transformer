"""Sparsemax normalised within groups of a 1-D tensor.

Sparsemax (Martins & Astudillo, "From Softmax to Sparsemax", ICML 2016) is a replacement for
softmax that can assign exactly zero weight. Here it is applied per group, the way
``torch_geometric.utils.softmax`` normalises the weights of each node's outgoing edges.

Adapted from HGP-SL (https://github.com/cszhangzhen/HGP-SL), which builds on the OpenNMT-py
implementation. The torch_scatter calls of the original are replaced with
``torch_geometric.utils.scatter``.

Example:
    >>> x = torch.tensor([2.0, 1.0, -1.0, 1.5, 0.0, 0.5, -0.5])
    >>> index = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    >>> Sparsemax()(x, index)
    tensor([0.7500, 0.0000, 0.0000, 0.2500, 0.2500, 0.7500, 0.0000])
"""

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch_geometric.utils import scatter


def _group_sizes(index: Tensor) -> Tensor:
    return scatter(torch.ones_like(index), index, dim=0, reduce="sum")


def _sort_within_groups(x: Tensor, index: Tensor, fill_value: float = -1e16) -> tuple[Tensor, Tensor]:
    """Sort ``x`` in descending order inside every group; return the sorted values and their running sums."""
    sizes = _group_sizes(index)
    num_groups, max_size = sizes.numel(), int(sizes.max())
    start = sizes.cumsum(0) - sizes

    # Place every group in a row of a padded (num_groups, max_size) matrix and sort the rows.
    position = torch.arange(index.numel(), device=x.device) - start[index] + index * max_size
    dense = x.new_full((num_groups * max_size,), fill_value)
    dense[position] = x
    dense = dense.view(num_groups, max_size).sort(dim=-1, descending=True).values
    cumsum = dense.cumsum(dim=-1).view(-1)
    dense = dense.view(-1)

    valid = dense != fill_value
    return dense[valid], cumsum[valid]


def _threshold(x: Tensor, index: Tensor) -> tuple[Tensor, Tensor]:
    """Sparsemax threshold tau and support size of every group."""
    sizes = _group_sizes(index)
    start = sizes.cumsum(0) - sizes

    sorted_x, cumsum = _sort_within_groups(x, index)
    cumsum = cumsum - 1.0
    rank = (torch.arange(index.numel(), device=x.device) - start[index] + 1).to(x.dtype)
    in_support = rank * sorted_x > cumsum

    support_size = scatter(in_support.long(), index, dim=0, dim_size=sizes.numel(), reduce="sum")
    last = (start + support_size - 1).clamp(min=0)
    tau = cumsum.gather(0, last) / support_size.clamp(min=1).to(x.dtype)
    return tau, support_size


class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, index: Tensor) -> Tensor:
        x = x - scatter(x, index, dim=0, reduce="max")[index]  # numerical stability
        tau, support_size = _threshold(x, index)
        output = torch.clamp(x - tau[index], min=0)
        ctx.save_for_backward(support_size, output, index)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        support_size, output, index = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = scatter(grad_input, index, dim=0, dim_size=support_size.numel(), reduce="sum")
        v_hat = v_hat / support_size.clamp(min=1).to(output.dtype)
        grad_input = torch.where(output != 0, grad_input - v_hat[index], grad_input)
        return grad_input, None


class Sparsemax(nn.Module):
    """Sparsemax of ``x`` within the groups given by ``index``.

    ``index`` must be sorted (the entries of each group contiguous), as the source nodes of a
    coalesced ``edge_index`` are.
    """

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        if index.numel() > 1 and bool((index[1:] < index[:-1]).any()):
            raise ValueError("Sparsemax expects a sorted group index")
        return SparsemaxFunction.apply(x, index)
