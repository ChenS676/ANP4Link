# %%
import os
import sys
current_file = os.path.abspath(__file__)
grandparent_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, grandparent_dir)

import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (
    to_networkx,
    from_networkx
)
import numpy as np
import torch
from syn_real.automorphism import (run_wl_test_and_group_nodes, 
                                   count_automorphic_edges, 
                                   compute_automorphism_metrics)
import random




# --- 2️⃣ Create Disjoint Graph Copies & Merge ---
def create_disjoint_graph(data: Data) -> Data:
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
    merged_edge_index = torch.tensor(list(merged_graph.edges)).mT

    if hasattr(data, "x") and data.x is not None:
        merged_x = torch.cat([data.x, data.x], dim=0)
    merged_data = Data(edge_index=merged_edge_index, num_nodes=2 * num_nodes)
    if hasattr(data, "x") and data.x is not None:
        merged_data.x = merged_x  
    print(merged_data.x)
    return merged_data


# --- 3️⃣ Add Controllable Random Edges ---
def add_random_edges(graph_data, 
                     inter_ratio=0.5, 
                     intra_ratio=0.5, 
                     total_edges=1000):
    """
    Adds random edges between and within two graph copies in a controlled way.

    Args:
        graph_data (Data): The graph structure (PyG format).
        inter_ratio (float): Fraction of edges to add **between** the two graph copies.
        intra_ratio (float): Fraction of edges to add **within** each graph copy.
        total_edges (int): Total number of random edges to add.

    Returns:
        Data: Graph with additional edges.
    """
    num_nodes = graph_data.num_nodes // 2 
    inter_edges = int(total_edges * inter_ratio)
    intra_edges = total_edges - inter_edges  
    inter_edges_list = [
        (random.randint(0, num_nodes - 1), random.randint(num_nodes, 2 * num_nodes - 1))
        for _ in range(inter_edges)
    ]
    intra_edges_list = []
    for _ in range(intra_edges):
        copy = random.choice([0, 1]) 
        base_offset = num_nodes * copy 
        u, v = random.sample(range(base_offset, base_offset + num_nodes), 2)
        intra_edges_list.append((u, v))
        
    new_edges = torch.tensor(inter_edges_list + intra_edges_list, dtype=torch.long).T
    updated_edge_index = torch.cat([graph_data.edge_index, new_edges], dim=1)
    return Data(edge_index=updated_edge_index, num_nodes=graph_data.num_nodes, x=graph_data.x)



# %% 
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




def semi_autom_expt(G, pos=None):

    # Process Graph with WL Test
    data = from_networkx(G)
    print(f"Number of nodes in the graph: {data.num_nodes}")
    analyze_automorphisms(G)
    mdata, mG = create_disjoint_graph(data)
    
    inter_ratio = 0.9
    intra_ratio = 0.5
    total_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*10
    
    for edges in total_edges:
        print(edges)
        if inter_ratio != 0 and intra_ratio != 0 and total_edges != 0:
            updated_graph_data, new_edges = add_random_edges(mdata, inter_ratio=inter_ratio, total_edges=edges)
            rewired_data = rewire_edges_regularly(data, keep_prob=0.6)
        else:
            updated_graph_data = mdata
        
        G = to_networkx(updated_graph_data, to_undirected=True)
        analyze_automorphisms(G)
    return 


# %%

if '__name__' == "__main__":
    parser = argparse.ArgumentParser(description="Semi-automorphism experiment on synthetic graphs")
    parser.add_argument("--graph_type", type=str, default="GraphType.BARABASI_ALBERT", help="Type of graph to generate")
    parser.add_argument("--N", type=int, default=10, help="Number of nodes in the graph")
    args = parser.parse_args()

    graph_type = args.graph_type
    N = args.N

    graph_type =  GraphType.ERDOS_RENYI
    N = 10

    if graph_type == RegularTilling.SQUARE_GRID:
        G, _, _, pos = init_regular_tilling(N, RegularTilling.SQUARE_GRID, seed=None)
    elif graph_type == 'GraphType.COMPLETE':
        graph_type = 'GraphType.COMPLETE'
        G = nx.complete_graph(N)
    else:
       # The code is generating a graph `G` with `N` nodes and a specified `graph_type` using a seed
       # value of 0 for random number generation. The function `generate_graph` is being called with
       # the specified parameters to create the graph.
        G = generate_graph(N, graph_type, seed=0)

    semi_autom_expt(G)

    # %%
    # semi_syn_graph(10, GraphType.BARABASI_ALBERT)

    # %%
    # semi_autom_test(10, GraphType.BARABASI_ALBERT)



