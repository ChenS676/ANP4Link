# %%
import os
import sys
sys.path.insert(0, "/hkfs/work/workspace/scratch/cc7738-2025_whole/ANP4Link/")
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import (
    degree,
    is_sparse,
    scatter,
    sort_edge_index,
    to_edge_index,
    from_networkx
)
from collections import defaultdict
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import train_test_split_edges, to_undirected
import random
from networkx import random_regular_graph
from syn_graph.graph_generation import GraphType 
from syn_graph.syn_random import RegularTilling
from collections import Counter
from syn_graph.syn_random import init_regular_tilling
from syn_graph.graph_generation import (generate_graph, 
                                        plot_triangular_graph)

# %%

def plot_graph_with_orbits(G, pos, orbits, custom_labels=None, figsize=(8, 6), cmap='tab20b'):
    """
    Plots a NetworkX graph with node coloring based on orbit labels.
    
    Parameters:
        G (nx.Graph): The graph to plot.
        pos (dict): Node positions.
        orbits (list or array): Orbit label for each node.
        custom_labels (dict, optional): Dictionary of node labels.
        figsize (tuple): Figure size for the plot.
        cmap (str): Matplotlib colormap for nodes.
    """
    plt.figure(figsize=figsize)
    node_colors = [orbits[node] for node in G.nodes()]
    nx.draw(
        G,
        pos=pos,
        labels=custom_labels,
        node_color=node_colors,
        cmap=cmap,
        node_size=500,
        font_weight='bold',
        edgecolors='black'
    )
    plt.title("Graph Colored by Orbit Labels")
    plt.show()
    plt.close()


# %%
def plot_orbit_dist(node_groups):
    group_sizes = np.array([len(group) for group in node_groups.values()])
    plt.figure(figsize=(6, 4))
    plt.hist(group_sizes, bins=range(min(group_sizes), max(group_sizes) + 2), align='left', rwidth=0.8)
    plt.xlabel('Orbit Label')
    plt.ylabel('Frequency')
    plt.title('Histogram of Orbit Labels')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    # plt.savefig(f"{name}_distribution_d{depth}.pdf")
    plt.close()



def count_orbit_edges(G, node_groups):
    # Create a mapping: node -> orbit_id

    intra_orbit_edges = 0
    inter_orbit_edges = 0

    try:
        for u, v in G.edges():
            if node_groups[u] == node_groups[v]:
                intra_orbit_edges += 1
            else:
                inter_orbit_edges += 1
    except:
    
        for links in G.edges():
            for u, v in links:
                if node_groups[u] == node_groups[v]:
                    intra_orbit_edges += 1
                else:
                    inter_orbit_edges += 1
    # print(f"Intra-orbit edges: {intra_orbit_edges}, Inter-orbit edges: {inter_orbit_edges}")
    print(f"{inter_orbit_edges/ (intra_orbit_edges + inter_orbit_edges) * 100}, of edges are inter-orbit")
    return intra_orbit_edges, inter_orbit_edges


def process_graph(N, graph_type, pos=None, is_grid=False, label="graph"):
    print(f"Processing graph of type {graph_type} with {N} nodes")
    if graph_type == RegularTilling.SQUARE_GRID:
        G, _, _, pos = init_regular_tilling(N, RegularTilling.SQUARE_GRID, seed=None)
    elif graph_type == 'GraphType.COMPLETE':
        graph_type = 'GraphType.COMPLETE'
        G = nx.complete_graph(N)
    else:
        G = generate_graph(N, graph_type, seed=0)
    
    # Process Graph with WL Test
    data = from_networkx(G)
    edge_index = data.edge_index
    node_groups, node_labels, orbits = run_wl_test_and_group_nodes(edge_index, num_nodes=G.number_of_nodes(), num_iterations=100)
    _, _, _ = compute_automorphism_metrics(node_groups, G.number_of_nodes())
    count_orbit_edges(G, node_labels)
    
    try:
        custom_labels = {}
        for i, ov in zip(G.nodes(), orbits):
                custom_labels[i] = f"{ov}"

        plot_graph_with_orbits(G, pos, orbits, custom_labels=custom_labels, figsize=(8, 6), cmap='tab20b')
        plot_orbit_dist(node_groups)
        plot_orbit_histogram(orbits)
    except:
        # Visualiz  e with WL-based coloring
        plot_triangular_graph(G, orbits, custom_labels=node_labels, figsize=(8, 6), cmap='tab20b')
        plot_orbit_dist(node_groups)
        plot_orbit_histogram(orbits)
    # save_metrics(metrics, f"{graph_type}_{N}", csv_path='summary.csv'



if __name__ == "__main__":    
    # %%
    process_graph(10, GraphType.BARABASI_ALBERT)


    # %%
    process_graph(20, GraphType.BARABASI_ALBERT)


    # %%
    process_graph(40, GraphType.BARABASI_ALBERT)


    # %%
    process_graph(60, GraphType.BARABASI_ALBERT)


    # %%
    process_graph(34, GraphType.TREE)

    # %%
    process_graph(10, GraphType.TREE)

    # %%
    process_graph(10, RegularTilling.SQUARE_GRID)


    process_graph(20, RegularTilling.SQUARE_GRID)


    process_graph(80, RegularTilling.SQUARE_GRID)


    process_graph(10, GraphType.TRIANGULAR)


    process_graph(20, GraphType.TRIANGULAR)


    process_graph(30, GraphType.TRIANGULAR)

