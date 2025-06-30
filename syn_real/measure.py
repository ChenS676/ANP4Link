# %%
import os
import sys
sys.path.insert(0, "/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/")

# === Standard Library ===
import random
from collections import defaultdict, Counter

# === Third-Party Libraries ===
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    from_networkx,
    to_networkx,
)
import pynauty

# === Project-Specific Modules ===
from syn_graph.graph_generation import GraphType, generate_graph
from syn_graph.syn_random import RegularTilling, init_regular_tilling


def get_regular_orbit_labels(G: nx.Graph):
    """
    Given a NetworkX graph, compute the automorphism orbits using pynauty.
    
    Args:
        G (networkx.Graph): Input undirected graph.
        
    Returns:
        orbits (List[int]): List of orbit IDs indexed by node ID.
        orbit_labels (Dict[node, str]): Mapping from node to its orbit label (as a string).
    """
    # Map original node labels to integers
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    # Build adjacency dict for pynauty
    adj_dict = {
        node_mapping[node]: [
            node_mapping[neighbor] for neighbor in G.neighbors(node)
        ]
        for node in G.nodes()
    }
    
    # Construct pynauty graph and compute automorphism group
    n = len(G.nodes())
    G_pynauty = pynauty.Graph(number_of_vertices=n, adjacency_dict=adj_dict, directed=False)
    _, _, _, orbits, _ = pynauty.autgrp(G_pynauty)

    # Relabel orbits to compact IDs
    new_orbit_ids = {orbit: idx for idx, orbit in enumerate(sorted(set(orbits)))}
    orbits_mapped = [new_orbit_ids[orbit] for orbit in orbits]

    # Map back to original node labels
    orbit_labels = {
        reverse_mapping[i]: str(orbit_id) for i, orbit_id in enumerate(orbits_mapped)
    }

    return orbit_labels, len(set(orbits_mapped))

# %%

def create_disjoint_graph(data):
    """
    Creates two disjoint copies of a real-world graph (e.g., Cora).
    Args:
        data (Data): PyG Data object representing the original graph.
    Returns:
        Data: PyG Data object representing the new merged graph.
    """
    num_nodes = data.num_nodes
    G = to_networkx(data, to_undirected=True)
    G2 = nx.relabel_nodes(G, lambda x: x + num_nodes)
    merged_graph = nx.compose(G, G2)
    merged_data = from_networkx(merged_graph)
    merged_data.edge_index = torch.cat([merged_data.edge_index, torch.tensor([num_nodes-1, num_nodes-1 + num_nodes]).unsqueeze(dim=1)], dim=1)
    if hasattr(data, 'x'):
        data.x = torch.ones((data.num_nodes, 16))
    merged_data.x = torch.cat([data.x, data.x], dim=0)
    merged_graph = to_networkx(merged_data, to_undirected=True)
    return merged_data, merged_graph



# %%
def semi_syn_graph(G, pos=None):
    # print(f"Processing graph of type {graph_type} with {N} nodes")
    # if graph_type == RegularTilling.SQUARE_GRID:
    #     G, _, _, pos = init_regular_tilling(N, RegularTilling.SQUARE_GRID, seed=None)
    # elif graph_type == 'GraphType.COMPLETE':
    #     graph_type = 'GraphType.COMPLETE'
    #     G = nx.complete_graph(N)
    # else:
    #     G = generate_graph(N, graph_type, seed=0)
    
    # Process Graph with WL Test
    data = from_networkx(G)
    # print(f"Graph type: {graph_type}, Number of nodes: {data.num_nodes}, Number of edges: {data.num_edges}")
    # print(f"Number of nodes in the graph: {data.num_nodes}")
    mdata, mG = create_disjoint_graph(data)
    # print(f"Merged Graph type: {graph_type}, Number of nodes: {mdata.num_nodes}, Number of edges: {mdata.num_edges}")
    print(mdata)

    # check the WL distribution
    _, node_labels, orbits = run_wl_test_and_group_nodes(data.edge_index, num_nodes=data.num_nodes, num_iterations=100)
    custom_labels = {}
    for i, ov in zip(G.nodes(), orbits):
            custom_labels[i] = f"{ov}"
    plot_graph_with_orbits(G, None, orbits, custom_labels=custom_labels, figsize=(8, 6), cmap='tab20b')
    count_orbit_edges(G, node_labels)

    _, mnode_labels, morbits = run_wl_test_and_group_nodes(mdata.edge_index, num_nodes=mdata.num_nodes, num_iterations=100)
    # metrics, num_nodes, group_sizes = compute_automorphism_metrics(node_groups, G.number_of_nodes())

    custom_labels = {}
    for i, ov in zip(mG.nodes(), morbits):
            custom_labels[i] = f"{ov}"
    
    plot_graph_with_orbits(mG, None, morbits, custom_labels=custom_labels, figsize=(8, 6), cmap='tab20b')
    count_orbit_edges(mG, mnode_labels)
    return 





def add_random_edges(graph_data, 
                     inter_ratio=0.5, 
                     intra_ratio=0.5, 
                     total_edges=1000):
    """
    Adds random edges between and within two graph copies in a controlled way.

    Args:
        graph_data (Data): The graph structure (PyG format).
        inter_ratio (float): Fraction of edges to add **between** the two graph copies.
        total_edges (int): Total number of random edges to add.

    Returns:
        Data: Graph with additional edges.
    """
    num_nodes = graph_data.num_nodes // 2 
    inter_edges = int(total_edges * inter_ratio)
    intra_edges = int(total_edges * intra_ratio)
    inter_edges_list = [
        (random.randint(0, num_nodes - 1), random.randint(num_nodes, 2 * num_nodes - 1))
        for _ in range(inter_edges)
    ]
    intra_edges_list = []
    for _ in range(intra_edges):
        copy = random.choice([0, 1])  # sample first or second copy
        base_offset = num_nodes * copy 
        u, v = random.sample(range(base_offset, base_offset + num_nodes), 2)
        intra_edges_list.append((u, v))
        
    new_edges = torch.tensor(inter_edges_list + intra_edges_list, dtype=torch.long).T
    print(new_edges.shape)
    updated_edge_index = torch.cat([graph_data.edge_index, new_edges], dim=1)
    return Data(edge_index=updated_edge_index, num_nodes=graph_data.num_nodes, x=graph_data.x), updated_edge_index




def semi_autom_test(N, graph_type, pos=None):
    # print(f"Processing graph of type {graph_type} with {N} nodes")
    if graph_type == RegularTilling.SQUARE_GRID:
        G, _, _, pos = init_regular_tilling(N, RegularTilling.SQUARE_GRID, seed=None)
    elif graph_type == 'GraphType.COMPLETE':
        graph_type = 'GraphType.COMPLETE'
        G = nx.complete_graph(N)
    else:
        G = generate_graph(N, graph_type, seed=0)
    
    # Process Graph with WL Test
    data = from_networkx(G)
    # print(f"Graph type: {graph_type}, Number of nodes: {data.num_nodes}, Number of edges: {data.num_edges}")
    # print(f"Number of nodes in the graph: {data.num_nodes}")

    # check the WL distribution
    analyze_automorphisms(G)

    mdata, mG = create_disjoint_graph(data)
    analyze_automorphisms(mG)
    return 





def hash_links_by_orbit(G: nx.Graph, orbits: list ):
    """
    Group and count edges by the sorted orbit pairs they connect.

    Args:
        G: networkx.Graph
        node_groups: list of orbit IDs per node (e.g., from WL hashing)

    Returns:
        edge_class_counts: dict with keys (orbit_a, orbit_b), values are counts
        edge_classes: list of (orbit_a, orbit_b) tuples in the same order as G.edges()
    """
    if type(orbits) is torch.Tensor:
        orbits = orbits.tolist()
    node_to_orbit = orbits
    edge_class_counts = defaultdict(int)
    edge_classes = []

    try:
        for u, v in G.edges():
            orbit_u = node_to_orbit[u]
            orbit_v = node_to_orbit[v]
            key = tuple(sorted((orbit_u, orbit_v))) 
            edge_class_counts[key] += 1
            edge_classes.append(key)
    except:
        # regular graph
        for links in G.edges():
            for u, v in links:
                orbit_u = node_to_orbit[u]
                orbit_v = node_to_orbit[v]
                key = tuple(sorted((orbit_u, orbit_v)))
                edge_class_counts[key] += 1
                edge_classes.append(key)    
    edge_role_size = sorted(list(edge_class_counts.values()), reverse=True)
    return edge_class_counts, edge_role_size




    

import itertools
def count_automorphic_edges(G, node_groups):
    """
    Counts intra-orbit and inter-orbit edges in a graph G based on node_groups,
    excluding nodes that are the only ones in their group.

    Parameters:
        G (networkx.Graph): The input graph.
        node_groups (list): A list where the index is the node ID and the value is the orbit/group ID.

    Returns:
        tuple: (intra_orbit_edges, inter_orbit_edges)
    """
    group_counts = Counter(node_groups)
    valid_nodes = {i for i, group in enumerate(node_groups) if group_counts[group] == 1}

    # print(f"unique orbit node: {len(valid_nodes)}")
    intra_orbit_edges = 0
    inter_orbit_edges = 0
    for u, v in itertools.product(G.nodes(), G.nodes()):
        if u in valid_nodes and v in valid_nodes:
            continue 
        # print(f"u: {u}, v: {v}, group_u: {node_groups[u]}, group_v: {node_groups[v]}")
        if node_groups[u] == node_groups[v]:
            intra_orbit_edges += 1
        else:
            inter_orbit_edges += 1
    print(f"Intra-orbit edges: {intra_orbit_edges}, Inter-orbit edges: {inter_orbit_edges}")
    print(f"Non-distinguishable edges: {(intra_orbit_edges+inter_orbit_edges)}")
    return intra_orbit_edges, inter_orbit_edges


def compute_automorphism_metrics(orbits, num_nodes):
    """
    Computes numerical metrics for graph automorphism based on WL node grouping.
    Args:
        node_groups (dict): Dictionary mapping WL hash values to lists of node indices.
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        dict: Automorphism metrics {A_r1, C_auto, H_auto}
    """
    # Compute the size of each group (how many nodes share the same WL label)
    node_groups = {}
    for node, label in enumerate(orbits):
        if label not in node_groups:
            node_groups[label] = []
        node_groups[label].append(node)
        
    group_sizes = np.array([len(group) for group in node_groups.values()])

    A_r1 = np.sum(group_sizes**2) / num_nodes**2
    C_auto = len(node_groups)
    A_r_norm_1 = 1 + np.log(A_r1) / np.log(num_nodes) # lower is less automorphism
    A_r_norm_2 = np.log(np.sum(group_sizes**2)) / (2 * np.log(num_nodes)) 
    A_r_log = (np.log(np.sum(group_sizes**2)) - np.log(num_nodes**2)) / np.log(num_nodes)
    automorphism_score = (len(node_groups) / num_nodes)
    return {
        "Automorphism Ratio (A_r1)": A_r1,
        "A_r_norm_2": A_r_norm_2,
        "A_r_norm_1": A_r_norm_1,
        "Number of Unique Groups (C_auto)": C_auto,
        "Automorphism Ratio (A_r_log)": A_r_log,
        "num_nodes": num_nodes,
        "automorphism_score": automorphism_score
    }, num_nodes, group_sizes


