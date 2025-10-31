# %%
import os
import sys 

notebook_path = os.getcwd()
sys.path.insert(0, os.path.dirname(notebook_path))

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
import numpy as np 

from syn_real.syn_datagen import analyze_automorphisms


# %% Generate graph
def lobster(N, seed):
    """ Creates a random Lobster graph with a backbone of size b (drawn from U[1, N)), and p (drawn
        from U[1, N − b ]) pendent vertices uniformly connected to the backbone, and additional
        N − b − p pendent vertices uniformly connected to the previous pendent vertices """
    np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    F = np.random.randint(low=B + 1, high=N + 1)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, F):
        G.add_edge(i, np.random.randint(B))
    for i in range(F, N):
        G.add_edge(i, np.random.randint(low=B, high=F))

    for n in G.nodes():
        G.nodes[n]['x'] = [1.0]
    
    pyg_data = from_networkx(G)
    auto_edges = []
    auto_nodes = []

    (num_automorphic_edges, 
    num_non_automorphic_edges, 
    non_automorphic_edges, 
    auto_edges, 
    auto_nodes, 
    unique_group_nodes) = analyze_automorphisms(pyg_data, G) # classify edges to be automorphic or not

    return pyg_data, auto_edges, auto_nodes, G


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
def run(N):
    data, auto_edges, auto_nodes, G_nx = lobster(N, 0)
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


    tsne = TSNE(n_components=2, perplexity=2,random_state=0)
    x_2d = tsne.fit_transform(x.numpy())

    # Plot node embeddings
    # plt.figure(figsize=(8,6))
    # for i, (x_, y_) in enumerate(x_2d):
    #     color = 'red' if i in [0,1,2,3] else 'blue'
    #     plt.scatter(x_, y_, c=color)
    #     plt.text(x_ + 0.02, y_ + 0.02, str(i), fontsize=9)
    # plt.title("2D Visualization of Node Embeddings")
    # plt.savefig('result.pdf')


    # pos = nx.spring_layout(G_nx, seed=42)
    # node_colors = ['red' if n in [0,1,2,3] else 'skyblue' for n in G_nx.nodes()]
    # plt.figure(figsize=(6,6))
    # nx.draw(G_nx, pos, with_labels=True, node_color=node_colors, node_size=600)
    # plt.title("Original Graph (Red = Automorphic Nodes)")
    # plt.savefig('graph.pdf')


    # preds, types = [], []
    # for i in range(all_edge_index.shape[1]):
    #     s, d = int(all_edge_index[0, i]), int(all_edge_index[1, i])
    #     pred = predictor(x[s].unsqueeze(0), x[d].unsqueeze(0)).item()
    #     t = 'A' if (s,d) in auto_edges or (d,s) in auto_edges else 'NA'
    #     preds.append(pred)
    #     types.append(t)

    # df = pd.DataFrame({"Prediction": preds, "EdgeType": types})
    # df.boxplot(column="Prediction", by="EdgeType")
    # plt.title("Link Prediction Score by Edge Type")
    # plt.suptitle("")
    # plt.ylabel("Score")
    # plt.savefig('result.pdf')

    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd

    # ---------- 1. 2D Visualization of Node Embeddings ----------
    # plt.figure(figsize=(8, 6))
    # for i, (x_, y_) in enumerate(x_2d):
    #     color = 'red' if i in [0, 1, 2, 3] else 'blue'
    #     plt.scatter(x_, y_, c=color, s=40, edgecolor='k', linewidth=0.5)
    #     plt.text(x_ + 0.03, y_ + 0.03, str(i), fontsize=9)

    # plt.title("2D Visualization of Node Embeddings", fontsize=14)
    # plt.xlabel("Embedding Dimension 1")
    # plt.ylabel("Embedding Dimension 2")
    # plt.grid(True, linestyle='--', alpha=0.3)
    # plt.tight_layout()
    # plt.savefig("embedding_plot.pdf")
    # plt.close()

    # ---------- 2. Visualization of the Original Graph ----------
    # pos = nx.spring_layout(G_nx, seed=42)
    # node_colors = ['red' if n in auto_nodes else 'skyblue' for n in G_nx.nodes()]

    # plt.figure(figsize=(6, 6))
    # nx.draw(
    #     G_nx, pos, with_labels=True, node_color=node_colors,
    #     node_size=700, font_size=10, edge_color='gray'
    # )
    # plt.title("Original Graph\n(Red = Automorphic Nodes)", fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f"graph_visual{N}.pdf")
    # plt.close()
    
        
    import matplotlib.pyplot as plt
    import networkx as nx

    # Layout and colors
    pos = nx.spring_layout(G_nx, seed=42)
    node_colors = ["#1BA462" if n in auto_nodes else "#1D7BBE" for n in G_nx.nodes()]  # Colorblind-safe red/blue

    # Plot settings
    plt.figure(figsize=(4, 4))  # Small but high-quality figure
    nx.draw_networkx(
        G_nx,
        pos=pos,
        with_labels=False,
        node_color=node_colors,
        node_size=50,
        font_size=8,
        font_weight='bold',
        edge_color='black',
        linewidths=0.8
    )

    # Title and legend
    plt.title("Graph Visualization", fontsize=12)
    red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Automorphic Nodes',
                        markerfacecolor='#d62728', markersize=8)
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Other Nodes',
                            markerfacecolor='#1f77b4', markersize=8)
    plt.legend(handles=[red_patch, blue_patch], loc='lower left', fontsize=8, frameon=False)

    # Clean layout
    plt.axis('off')
    plt.tight_layout()

    # Save in vector format
    plt.savefig(f"/pfs/work9/workspace/scratch/ka_cc7738-orbit-gnn/ANP4Link/aut_edg/graph_visual{N}.pdf", bbox_inches='tight')
    plt.close()


    # import matplotlib.pyplot as plt
    # import networkx as nx
    # from matplotlib.patches import Patch

    # Layout
    # pos = nx.spring_layout(G_nx, seed=42)

    # # Colorblind-friendly node colors
    # color_auto = "#71d675"     # Strong red
    # color_other = "#c465cd"    # Blue

    # node_colors = [color_auto if n in auto_nodes else color_other for n in G_nx.nodes()]

    # # Plot
    # plt.figure(figsize=(3.5, 3.5))  # For NeurIPS 2-column fit

    # Draw nodes and edges
    # nx.draw_networkx_nodes(
    #     G_nx, pos, node_color=node_colors, node_size=200,
    #     edgecolors='black', linewidths=0.5
    # )
    # nx.draw_networkx_edges(
    #     G_nx, pos, alpha=0.9, width=0.9, edge_color='gray'
    # )

    # Optional labels
    # nx.draw_networkx_labels(G_nx, pos, font_size=6)

    # Legend
    # legend_elements = [
    #     Patch(facecolor=color_auto, edgecolor='black', label='Automorphic Node'),
    #     Patch(facecolor=color_other, edgecolor='black', label='Other Node')
    # ]
    # plt.legend(
    #     handles=legend_elements,
    #     loc='upper left',
    #     fontsize=15,
    #     frameon=False
    # )

    # # Title and layout
    # # plt.title("Graph Structure", fontsize=10)
    # plt.axis('off')
    # plt.tight_layout()

    # # Save as vector graphic
    # plt.savefig(f'graph_visual_{N}.pdf', dpi=300, bbox_inches='tight')
    # plt.close()

    # ---------- 3. Link Prediction Box Plot ----------
    preds, types = [], []
    for i in range(all_edge_index.shape[1]):
        s, d = int(all_edge_index[0, i]), int(all_edge_index[1, i])
        pred = predictor(x[s].unsqueeze(0), x[d].unsqueeze(0)).item()
        t = 'A' if (s, d) in auto_edges or (d, s) in auto_edges else 'NA'
        preds.append(pred)
        types.append(t)

    import pandas as pd
    import matplotlib.pyplot as plt

    # 构造 DataFrame
    df = pd.DataFrame({"Prediction": preds, "EdgeType": types})

    # 创建正方形画布
    plt.figure(figsize=(6, 6))

    # 画箱线图
    df.boxplot(column="Prediction", by="EdgeType")

    # 标题和轴标签
    plt.title("", fontsize=20)
    plt.suptitle("")  # 移除默认副标题
    plt.xlabel("")
    plt.ylabel("Prediction Score", fontsize=20)

    # 坐标轴字体大小
    plt.tick_params(axis='both', which='major', labelsize=20)

    # 自动调整布局并保存
    plt.tight_layout()

    return plt
    
for i in range(1, 10):
    plt = run(10)
    
    plt.savefig(f"link_prediction_boxplot_{i}.pdf", dpi=200)
    plt.close()
    



# run(40)

# run(60)