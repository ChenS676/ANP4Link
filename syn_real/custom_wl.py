
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import (
    degree,
    is_sparse,
    scatter,
    sort_edge_index,
    to_edge_index,
    from_networkx
)
from torch import Tensor
from scipy.sparse.linalg import eigsh
from typing import Optional
from torch_geometric.typing import Adj
from torch_geometric.utils import train_test_split_edges, to_undirected



class WLConv(torch.nn.Module):
    r"""The Weisfeiler Lehman (WL) operator from the `"A Reduction of a Graph
    to a Canonical Form and an Algebra Arising During this Reduction"
    <https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf>`_ paper.

    :class:`WLConv` iteratively refines node colorings according to:

    .. math::
        \mathbf{x}^{\prime}_i = \textrm{hash} \left( \mathbf{x}_i, \{
        \mathbf{x}_j \colon j \in \mathcal{N}(i) \} \right)

    Shapes:
        - **input:**
          node coloring :math:`(|\mathcal{V}|, F_{in})` *(one-hot encodings)*
          or :math:`(|\mathcal{V}|)` *(integer-based)*,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node coloring :math:`(|\mathcal{V}|)` *(integer-based)*
    """
    def __init__(self):
        super().__init__()
        self.hashmap = {}

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.hashmap = {}

    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        r"""Runs the forward pass of the module."""
        if x.dim() > 1:
            assert (x.sum(dim=-1) == 1).sum() == x.size(0)
            x = x.argmax(dim=-1)  # one-hot -> integer.
        assert x.dtype == torch.long

        if is_sparse(edge_index):
            col_and_row, _ = to_edge_index(edge_index)
            col = col_and_row[0]
            row = col_and_row[1]
        else:
            edge_index = sort_edge_index(edge_index, num_nodes=x.size(0),
                                         sort_by_row=False)
            row, col = edge_index[0], edge_index[1]

        # `col` is sorted, so we can use it to `split` neighbors to groups:
        deg = degree(col, x.size(0), dtype=torch.long).tolist()

        out = []
        for node, neighbors in zip(x.tolist(), x[row].split(deg)):
            idx = hash(tuple([node] + neighbors.sort()[0].tolist()))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap[idx])

        return torch.tensor(out, device=x.device)

    def histogram(self, x: Tensor, batch: Optional[Tensor] = None,
                  norm: bool = False) -> Tensor:
        r"""Given a node coloring :obj:`x`, computes the color histograms of
        the respective graphs (separated by :obj:`batch`).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_colors = len(self.hashmap)
        batch_size = int(batch.max()) + 1

        index = batch * num_colors + x
        out = scatter(torch.ones_like(index), index, dim=0,
                      dim_size=num_colors * batch_size, reduce='sum')
        out = out.view(batch_size, num_colors)

        if norm:
            out = out.to(torch.float)
            out /= out.norm(dim=-1, keepdim=True)

        return out



class WLConvMultiFeature(torch.nn.Module):
    def __init__(self):
        """Weisfeiler-Lehman convolution supporting multi-dimensional node features."""
        super().__init__()
        self.hashmap = {}

    def reset_parameters(self):
        """Resets hash storage."""
        self.hashmap = {}
    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """
        Runs the forward pass of the Weisfeiler-Lehman update step.

        Args:
            x (Tensor): Node feature matrix of shape (num_nodes, num_features).
            edge_index (Tensor): Edge index tensor of shape (2, num_edges).

        Returns:
            Tensor: Updated node hash labels of shape (num_nodes,).
        """
        if x.dim() > 1:
            # Convert multi-dimensional features into hashable form
            x = [tuple(row.tolist()) for row in x]  # Convert each feature row to a tuple
        
        if is_sparse(edge_index):
            col_and_row, _ = to_edge_index(edge_index)
            col, row = col_and_row[0], col_and_row[1]
        else:
            edge_index = sort_edge_index(edge_index, num_nodes=len(x), sort_by_row=False)
            row, col = edge_index[0], edge_index[1]
        # Compute node degree
        deg = degree(col, len(x), dtype=torch.long).tolist()
        out = []
        for node, neighbors in zip(x, [x[row] for row in row.split(deg)]):
            # Hash the node's feature and its sorted neighbor features
            idx = hash((node, tuple(sorted(neighbors))))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap[idx])
        return torch.tensor(out, device=edge_index.device, dtype=torch.long)



class WLConvMultiFeature(torch.nn.Module):
    def __init__(self):
        """Weisfeiler-Lehman convolution supporting multi-dimensional node features."""
        super().__init__()
        self.hashmap = {}

    def reset_parameters(self):
        """Resets hash storage."""
        self.hashmap = {}

    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """
        Runs the forward pass of the Weisfeiler-Lehman update step.

        Args:
            x (Tensor): Node feature matrix of shape (num_nodes, num_features).
            edge_index (Tensor): Edge index tensor of shape (2, num_edges).

        Returns:
            Tensor: Updated node hash labels of shape (num_nodes,).
        """
        if x.ndim > 1:
            # Convert multi-dimensional features into hashable form (tuple per node)
            x = [tuple(row.tolist()) for row in x]
        if is_sparse(edge_index):
            col_and_row, _ = to_edge_index(edge_index)
            col, row = col_and_row[0], col_and_row[1]
        else:
            edge_index = sort_edge_index(edge_index, num_nodes=len(x), sort_by_row=False)
            row, col = edge_index[0], edge_index[1]
        # Compute node degree
        deg = degree(col, len(x), dtype=torch.long).tolist()
        # Corrected neighbor feature extraction
        neighbors_per_node = [[] for _ in range(len(x))]
        for src, dst in zip(row.tolist(), col.tolist()):
            neighbors_per_node[dst].append(x[src])  # Collect features of neighbors
        out = []
        for node, neighbors in zip(x, neighbors_per_node): # O(N^avg_deg)
            # Hash the node's feature and its sorted neighbor features
            idx = hash((node, tuple(sorted(neighbors))))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap[idx])
        return torch.tensor(out, device=edge_index.device, dtype=torch.long)



class WLConvOptimized(torch.nn.Module):
    def __init__(self):
        """Weisfeiler-Lehman convolution optimized for multi-dimensional node features."""
        super().__init__()
        self.hashmap = {}

    def reset_parameters(self):
        """Resets hash storage."""
        self.hashmap = {}

    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """
        Runs the forward pass of the Weisfeiler-Lehman update step (optimized).

        Args:
            x (Tensor): Node feature matrix of shape (num_nodes, num_features).
            edge_index (Tensor): Edge index tensor of shape (2, num_edges).

        Returns:
            Tensor: Updated node hash labels of shape (num_nodes,).
        """
        num_nodes = x.shape[0]
        if x.ndim > 1:
            # Convert multi-dimensional features into a hashable form (faster version)
            x = x.tolist()  # Avoid repeated tolist() calls
        else:
            x = x.tolist()
        # Convert edge_index to row, col format
        if is_sparse(edge_index):
            col_and_row, _ = to_edge_index(edge_index)
            col, row = col_and_row[0], col_and_row[1]
        else:
            edge_index = sort_edge_index(edge_index, num_nodes=num_nodes, sort_by_row=False)
            row, col = edge_index[0], edge_index[1]
        # Compute degree and initialize neighbor storage efficiently
        neighbors_per_node = [[] for _ in range(num_nodes)]
        # Use NumPy array for faster indexing
        x_array = np.array(x, dtype=object)  # Keeps features as objects for hashing
        # Efficient neighbor collection
        for src, dst in zip(row.cpu().numpy(), col.cpu().numpy()):
            neighbors_per_node[dst].append(x_array[src])
        # Faster hashing using NumPy broadcasting and tuple encoding
        out = np.empty(num_nodes, dtype=int)
        for i in range(num_nodes):
            # Hash the node's feature and its sorted neighbor features (avoiding Python loops)
            idx = hash((tuple([x_array[i]]), tuple(sorted(neighbors_per_node[i]))))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out[i] = self.hashmap[idx]
        # Convert back to a tensor
        return torch.tensor(out, device=edge_index.device, dtype=torch.long)



def higher_order_wl(edge_index, num_nodes, k=2, num_iterations=1000):
    """
    Runs the Higher-Order Weisfeiler-Lehman (k-WL) test and groups nodes with similar hashed labels.

    Args:
        edge_index (Tensor): The edge index tensor (2, |E|) representing the graph.
        num_nodes (int): The number of nodes in the graph.
        k (int): Order of WL test (1 = standard WL, 2+ = higher-order WL).
        num_iterations (int): Number of WL iterations.

    Returns:
        node_groups (dict): Mapping from WL hashes to node sets.
        node_labels (Tensor): Final hashed labels for each node.
    """
    
    def hash_function(vals):
        """Simple multi-dimensional hash function."""
        return hash(tuple(sorted(vals)))

    node_labels = torch.arange(num_nodes, dtype=torch.long)
    edge_index = to_undirected(edge_index)

    for _ in range(num_iterations):
        new_labels = {}
        
        for node in range(num_nodes):
            neighbors = [node]
            for _ in range(k):
                neighbors = set(neighbors).union(edge_index[1][edge_index[0] == node].tolist())

            neighbor_labels = [node_labels[n].item() for n in neighbors]
            new_labels[node] = hash_function([node_labels[node].item()] + neighbor_labels)

        new_labels = torch.tensor([new_labels[n] for n in range(num_nodes)], dtype=torch.long)

        if torch.equal(new_labels, node_labels):
            break
        node_labels = new_labels

    unique_labels, inverse_indices = torch.unique(node_labels, return_inverse=True)
    node_groups = {label.item(): (inverse_indices == i).nonzero(as_tuple=True)[0].tolist()
                   for i, label in enumerate(unique_labels)}

    return node_groups, node_labels

