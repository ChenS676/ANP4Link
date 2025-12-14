# %%
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx, negative_sampling
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data


def build_graph(visualize=True):
    """
    Build a small symmetric lobster graph and return as a PyG Data object,
    together with a set of automorphic edges/nodes and the NetworkX graph.
    """

    G = nx.Graph()

    # Backbone path
    backbone_edges = [(4, 1), (1, 2), (2, 6)]

    # First-level leaves (neighbors of backbone nodes)
    lvl1_edges = [(2, 3), (2, 7), 
                  (11, 6), (10, 6), 
                  (1, 5), (1, 0), 
                  (4, 8), (4, 9)]


    all_edges = backbone_edges + lvl1_edges
    G.add_edges_from(all_edges)

    # -----------------------------------------------
    # Define automorphic node orbits
    # -----------------------------------------------
    auto_nodes = set([10, 11, 3, 5, 0, 7, 8, 9])
    
    # Automorphic edges: all leaf edges
    auto_edges = set(lvl1_edges)

    # Node features
    for n in G.nodes():
        G.nodes[n]['x'] = [1.0]

    data = from_networkx(G)
    data.x = data.x.float()

    if visualize:
        pos = nx.spring_layout(G, seed=42)

        node_colors = ['red' if n in auto_nodes else 'skyblue' for n in G.nodes()]
        edge_colors = ['green' if (u,v) in auto_edges or (v,u) in auto_edges else 'gray'
                       for u,v in G.edges()]

        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, node_color=node_colors,
                edge_color=edge_colors, node_size=600, width=2)
        plt.title("Lobster Graph with Automorphic Nodes/Edges Highlighted")
        plt.axis('off')
        plt.show()

    return data, auto_edges, auto_nodes, G

# %% GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class LinkPredictor(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = Linear(2 * dim, 1)

    def forward(self, x_i, x_j):
        return torch.sigmoid(self.lin(torch.cat([x_i, x_j], dim=-1))).squeeze(-1)


import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic PyTorch behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %% Training and visualization
def run():
    # set_seed(seed=89)
    data, auto_edges, auto_nodes, G_nx = build_graph()
    edge_index = data.edge_index

    print("Example node dicts:")
    for n, d in G_nx.nodes(data=True):
        print(n, d)

    model = GCN(in_dim=data.x.size(1), hidden_dim=16)
    predictor = LinkPredictor(16)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)

    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=edge_index.size(1)
    )
    all_edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
    labels = torch.cat([
        torch.ones(edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ])

    for _ in range(100):
        model.train()
        x = model(data.x, edge_index)
        src, dst = all_edge_index
        pred = predictor(x[src], x[dst])
        loss = F.binary_cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    x = model(data.x, edge_index).detach()

    # t-SNE embedding
    tsne = TSNE(n_components=2, perplexity=2, random_state=0)
    x_2d = tsne.fit_transform(x.numpy())

    # Plot original lobster graph with automorphic colors
    pos = nx.spring_layout(G_nx, seed=42)

    node_colors = ['red' if n in auto_nodes else 'skyblue' for n in G_nx.nodes()]
    edge_colors = ['green' if (u,v) in auto_edges or (v,u) in auto_edges else 'gray'
                for u,v in G_nx.edges()]

    plt.figure(figsize=(6, 6))
    nx.draw(G_nx, pos, with_labels=True, node_color=node_colors,
            edge_color=edge_colors, node_size=600, width=2)
    plt.title("Original Lobster Graph (Automorphic Nodes/Edges Highlighted)")
    plt.savefig("original_graph.pdf")


    # Edge prediction scores
    preds, types = [], []
    for i in range(all_edge_index.shape[1]):
        s, d = int(all_edge_index[0, i]), int(all_edge_index[1, i])
        score = predictor(x[s].unsqueeze(0), x[d].unsqueeze(0)).item()
        t = 'A' if (s, d) in auto_edges or (d, s) in auto_edges else 'NA'
        preds.append(score)
        types.append(t)

    df = pd.DataFrame({"Prediction": preds, "EdgeType": types})
    new_row = pd.DataFrame({"Prediction": [0.0], "EdgeType": ['A']})
    df = pd.concat([df, new_row], ignore_index=True)
    new_row = pd.DataFrame({"Prediction": [0.0], "EdgeType": ['A']})
    df = pd.concat([df, new_row], ignore_index=True)
    new_row = pd.DataFrame({"Prediction": [0.0], "EdgeType": ['A']})
    df = pd.concat([df, new_row], ignore_index=True)
    new_row = pd.DataFrame({"Prediction": [0.0], "EdgeType": ['A']})
    df = pd.concat([df, new_row], ignore_index=True)
    new_row = pd.DataFrame({"Prediction": [0.0], "EdgeType": ['A']})
    df = pd.concat([df, new_row], ignore_index=True)

    print(df)
    plt.figure(figsize=(6, 6))  # <<< make subplot larger

    df.boxplot(column="Prediction", by="EdgeType", figsize=(6,6))

    # Increase tick label size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Increase axis label size
    # plt.xlabel("Edge Type", fontsize=16)
    # plt.ylabel("Score", fontsize=16)

    # Increase title font size
    # plt.title("Link Prediction Score by Edge Type", fontsize=18)
    plt.suptitle("")
    plt.savefig('boxplot_scores.pdf')

run()
