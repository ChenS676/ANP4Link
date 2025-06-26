# %%
import os
import sys
sys.path.insert(0, "/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/")
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.typing import Adj
import itertools
from torch_geometric.utils import (
    degree,
    is_sparse,
    scatter,
    sort_edge_index,
    to_edge_index,
    from_networkx,
    to_networkx,
    train_test_split_edges,
    to_undirected
)
from torch_geometric.data import Data


from syn_real.custom_wl import (get_graph_orbits,
                                run_wl_test_and_group_nodes)
from syn_real.plotting import plot_graph_with_orbits
from syn_real.measure import (hash_links_by_orbit, 
                              compute_automorphism_metrics,
                              count_automorphic_edges
                            )
from syn_real.auto_operation import (create_disjoint_graph,
                            add_random_edges)
from collections import Counter

# %%

def get_k_hop_subgraph_from_dataset(dataset_name="Cora", num_hops=4, node_idx=0, visualize=True):
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
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
    data = dataset[0]

    # Extract k-hop subgraph
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )

    # Create subgraph Data object
    sub_x = data.x[subset]
    sub_y = data.y[subset]
    sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)

    # Optional visualization
    if visualize:
        G_sub = to_networkx(sub_data, to_undirected=True)
        plt.figure(figsize=(8, 6))
        nx.draw(G_sub, with_labels=True, node_size=300, node_color='skyblue', edge_color='gray')
        plt.title(f"{num_hops}-Hop Subgraph from Node {node_idx} ({dataset_name})")
        plt.axis('off')
        plt.show()

    return sub_data



def plot_orbit_dist(node_groups):
    node_dist = {}
    for node, label in enumerate(node_groups):
        if label not in node_dist:
            node_dist[label] = []
        node_dist[label].append(node)
        
    group_sizes = np.array([len(group) for group in node_dist.values()])
    

    plt.figure(figsize=(6, 4))
    plt.hist(group_sizes, bins=range(min(group_sizes), max(group_sizes) + 2), align='left', rwidth=0.8)
    plt.xlabel('Orbit Label')
    plt.ylabel('Frequency')
    plt.title('Histogram of Orbit Size (Dist of Dist)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    # plt.savefig(f"{name}_distribution_d{depth}.pdf")
    plt.close()



def plot_orbit(orbits):
    plt.figure(figsize=(6, 4))
    plt.hist(orbits, bins=range(min(orbits), max(orbits) + 2), align='left', rwidth=0.8)
    plt.xlabel('Orbit Label')
    plt.ylabel('Frequency')
    plt.title('Orbit Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    # plt.savefig(f"{name}_distribution_d{depth}.pdf")
    plt.close()
    
    
    
def analyze_automorphisms(G):

    _, node_labels, orbits = run_wl_test_and_group_nodes(data.edge_index, num_nodes=data.num_nodes, num_iterations=100)
    orbits, num_orbit = get_graph_orbits(G)
    # print(f"Number of orbits: {num_orbit}")

    custom_labels = {}
    for i, ov in zip(G.nodes(), orbits):
            custom_labels[i] = f"{ov}"

    count_automorphic_edges(G, orbits)
    
    metrics, num_nodes, group_sizes = compute_automorphism_metrics(orbits, G.number_of_nodes())
    
    plot_orbit_dist(orbits)
    plot_orbit(orbits)
    hash_links_by_orbit(G, orbits)
    plot_graph_with_orbits(G, 
                           None, 
                           orbits, 
                           custom_labels=custom_labels, 
                           figsize=(8, 6), cmap='tab20b')
    return metrics, num_nodes, group_sizes, orbits


# %%
import random

def rewire_edges_regularly(data, keep_prob=0.7):
    edge_list = data.edge_index.T.tolist()
    num_edges = len(edge_list)
    new_edges = []
    for _ in range(num_edges):
        if random.random() < keep_prob:
            new_edges.append(random.choice(edge_list))
        else:
            u, v = random.sample(range(data.num_nodes), 2)
            new_edges.append([u, v])
    new_edge_index = torch.tensor(new_edges, dtype=torch.long).T
    return Data(edge_index=new_edge_index, num_nodes=data.num_nodes, x=data.x)


if __name__ == '__main__':

    # Example usage
    subgraph = get_k_hop_subgraph_from_dataset(dataset_name="Cora", num_hops=3, node_idx=0, visualize=True)
    print(subgraph)
    print("Subgraph edge index:", subgraph.edge_index)
    print("Subgraph node features:", subgraph.x.shape)
    print("Subgraph labels:", subgraph.y.shape)
    
    # %%
    G = to_networkx(subgraph, to_undirected=True)

    # Process Graph with WL Test
    data = from_networkx(G)
    print(f"Number of nodes in the graph: {data.num_nodes}")
    analyze_automorphisms(G)
    # mdata, mG = create_disjoint_graph(data)
    
    inter_ratio = 1
    intra_ratio = 1
    total_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    interval = 1
    for edges in total_edges:
        if inter_ratio != 0 and intra_ratio != 0 and total_edges != 0:
            updated_graph_data, new_edges = add_random_edges(data, inter_ratio=inter_ratio, total_edges=edges*interval)
            # rewired_data = rewire_edges_regularly(data, keep_prob=0.6)
        else:
            updated_graph_data = data
        
        G = to_networkx(updated_graph_data, to_undirected=True)
        analyze_automorphisms(G)
