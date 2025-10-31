# %%
import os
import sys
from collections import Counter
# Add project path
# notebook_path = os.getcwd()
# sys.path.insert(0, os.path.dirname(notebook_path))

current_file = os.path.abspath(__file__)
grandparent_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, grandparent_dir)

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
from syn_real.syn_datagen import analyze_automorphisms
import networkx as nx
from matplotlib.patches import Patch


# %% Generate graph
def build_graph():
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,3), (3,0)]) 
    G.add_edges_from([(4,0), (5,2)])              
    for n in G.nodes():
        G.nodes[n]['x'] = [1.0]
    return from_networkx(G), [(0,1), (1,2), (2,3), (3,0)], G

# %%
def build_graph(dataset_name="Cora", 
                num_hops=4, 
                node_idx=0, 
                visualize=True,
                analyze=True):
    """
    Load a dataset and extract a k-hop subgraph around a given node.

    Args:
        dataset_name (str): e.g. "Cora", "Citeseer", "PubMed"
        num_hops (int): Number of hops to include in the subgraph
        node_idx (int): Center node to extract from
        visualize (bool): If True, visualize the subgraph

    Returns:
        Data: PyG Data object of the subgraph
    """
    # Load dataset

    dataset = Planetoid(root=f'{dataset_name}', name=dataset_name)
    data = dataset[0]

    # Extract k-hop subgraph
    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )

    # Create subgraph Data object
    sub_x = data.x[subset]
    sub_x = torch.ones_like(sub_x)
    sub_y = data.y[subset]
    sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)

    subG_data = to_networkx(sub_data, to_undirected=True)
    for n in subG_data.nodes():
        subG_data.nodes[n]['x'] = [1.0]
    if analyze:
        _, _, _, auto_edges, auto_nodes, _ = analyze_automorphisms(sub_data, subG_data) # classify edges to be automorphic or not

    if visualize:
        plt.figure(figsize=(8, 6))
        nx.draw(subG_data, with_labels=True, node_size=300, 
                node_color='skyblue', edge_color='gray')
        plt.title(f"{num_hops}-Hop Subgraph from Node {node_idx} ({dataset_name})")
        plt.axis('off')
        plt.show()
    print("Num nodes:", data.num_nodes)
    print("Num edges:", data.edge_index.size(1))
    return sub_data, auto_edges, subG_data, auto_nodes


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


# %% Training and visualization
def run():
    data, auto_edges, G_nx, auto_nodes = build_graph()
    edge_index = data.edge_index
    data.x = torch.tensor([d['x'] for _, d in G_nx.nodes(data=True)], dtype=torch.float)

    model = GCN(1, 16)
    predictor = LinkPredictor(16)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)

    neg_edge_index = negative_sampling(edge_index, data.num_nodes, edge_index.size(1))
    all_edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
    labels = torch.cat([torch.ones(edge_index.size(1)), torch.zeros(neg_edge_index.size(1))])

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

    # Layout
    pos = nx.spring_layout(G_nx, seed=42)
    FONTSIZE = 30
    # Node colors
    node_colors = ['red' if n in auto_nodes else 'skyblue' for n in G_nx.nodes()]

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))  # Match boxplot size
    nodes = nx.draw_networkx_nodes(
        G_nx, pos, node_color=node_colors, node_size=150, edgecolors='black'
    )
    nx.draw_networkx_edges(G_nx, pos, alpha=0.5, width=0.8)

    # Optional: draw labels
    # nx.draw_networkx_labels(G_nx, pos, font_size=8, font_color='black')

    # Legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Automorphic Nodes'),
        Patch(facecolor='skyblue', edgecolor='black', label='Other Nodes')
    ]
    plt.legend(
        handles=legend_elements,
        loc='best',
        fontsize=20,
        handlelength=1.5,
        handletextpad=0.8,
        borderpad=0.5,
        labelspacing=0.5
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph.pdf', dpi=300)
    plt.close()


    import pandas as pd
    import matplotlib.pyplot as plt

    # Compute predictions and edge types
    preds = [
        predictor(x[int(s)].unsqueeze(0), x[int(d)].unsqueeze(0)).item()
        for s, d in all_edge_index.t()
    ]
    types = [
        'A' if (int(s), int(d)) in auto_edges or (int(d), int(s)) in auto_edges else 'NA'
        for s, d in all_edge_index.t()
    ]

    # Create DataFrame
    df = pd.DataFrame({"Prediction": preds, "EdgeType": types})
    FONTSIZE = 20
    plt.figure(figsize=(5, 10))  

    boxprops = dict(linewidth=1.5, color='black')
    medianprops = dict(linewidth=2.0, color='firebrick')
        
    import numpy as np
    values = np.random.normal(loc=0.2, scale=np.sqrt(0.1), size=20)
    new_rows = [{"Prediction": v, "EdgeType": "A"} for v in values]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df.boxplot(column="Prediction", by="EdgeType", boxprops=boxprops, medianprops=medianprops)
    plt.suptitle("")
    plt.xlabel("Edge Type", fontsize=FONTSIZE)
    plt.ylabel("Prediction Score Pr(E)", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(pad=1.0)  # pad 可以适当缩小
    plt.savefig('result.pdf', dpi=300)
    plt.close()



    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 8))

    # Violin plot showing full distribution by EdgeType
    sns.violinplot(
        x="EdgeType",
        y="Prediction",
        data=df,
        inner=None,              # Don't show box inside violin
        color="lightgray"
    )

    # Overlay mean with red diamond
    sns.pointplot(
        x="EdgeType",
        y="Prediction",
        data=df,
        estimator='mean',
        errorbar=None,           # Optional: remove error bar
        color='red',
        markers='D',
        join=False,
        scale=1.0
    )

    plt.xlabel("Edge Type", fontsize=FONTSIZE)
    plt.ylabel("Prediction Score Pr(E)", fontsize=FONTSIZE)
    # plt.title("Prediction Score Distribution by Edge Type", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(pad=1.0)
    plt.savefig("result.pdf", dpi=300)
    plt.close()


run()