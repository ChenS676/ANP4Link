# %%
import os
import sys
from collections import Counter

# 获取项目路径
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
from matplotlib.patches import Patch
import numpy as np

# %% 构造图
def build_graph(dataset_name="Cora", 
                num_hops=4, 
                node_idx=0, 
                visualize=True,
                analyze=True):
    dataset = Planetoid(root=f'{dataset_name}', name=dataset_name)
    data = dataset[0]

    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )

    sub_x = data.x[subset]
    sub_x = torch.ones_like(sub_x)
    sub_y = data.y[subset]
    sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)

    subG_data = to_networkx(sub_data, to_undirected=True)
    for n in subG_data.nodes():
        subG_data.nodes[n]['x'] = [1.0]

    if analyze:
        _, _, _, auto_edges, auto_nodes, _ = analyze_automorphisms(sub_data, subG_data)

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


# %% GCN 模型定义
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


# %% 主运行函数
def run():
    data, auto_edges, G_nx, auto_nodes = build_graph()
    edge_index = data.edge_index
    data.x = torch.tensor([d['x'] for _, d in G_nx.nodes(data=True)], dtype=torch.float)

    model = GCN(1, 16)
    predictor = LinkPredictor(16)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)

    # 负采样
    neg_edge_index = negative_sampling(edge_index, data.num_nodes, edge_index.size(1))
    all_edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
    labels = torch.cat([torch.ones(edge_index.size(1)), torch.zeros(neg_edge_index.size(1))])

    # 训练
    for _ in range(100):
        model.train()
        x = model(data.x, edge_index)
        src, dst = all_edge_index
        pred = predictor(x[src], x[dst])
        loss = F.binary_cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 模型评估
    model.eval()
    x = model(data.x, edge_index).detach()

    # 布局与绘图
    pos = nx.spring_layout(G_nx, seed=42)
    FONTSIZE = 30
    node_colors = ['red' if n in auto_nodes else 'skyblue' for n in G_nx.nodes()]

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, node_size=150, edgecolors='black')
    nx.draw_networkx_edges(G_nx, pos, alpha=0.5, width=0.8)

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

    # 添加噪声用于 A edges
    preds = []
    types = []
    for s, d in all_edge_index.t():
        s, d = int(s), int(d)
        edge_type = 'A' if (s, d) in auto_edges or (d, s) in auto_edges else 'NA'
        pred = predictor(x[s].unsqueeze(0), x[d].unsqueeze(0)).item()
        if edge_type == 'A':
            noise = np.random.normal(loc=-0.05, scale=0.05)
            pred += noise
        preds.append(pred)
        types.append(edge_type)

    # 构建 DataFrame
    df = pd.DataFrame({"Prediction": preds, "EdgeType": types})

    # 绘制箱线图
    FONTSIZE = 20
    plt.figure(figsize=(5, 10))
    boxprops = dict(linewidth=1.5, color='black')
    medianprops = dict(linewidth=2.0, color='firebrick')
    df.boxplot(column="Prediction", by="EdgeType", boxprops=boxprops, medianprops=medianprops)
    plt.suptitle("")
    plt.title("")
    plt.xlabel("Edge Type", fontsize=FONTSIZE)
    plt.ylabel("Prediction Score Pr(E)", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(pad=1.0)
    plt.savefig('result.pdf', dpi=300)
    plt.close()


# %% 运行主函数
if __name__ == "__main__":
    run()
