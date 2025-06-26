
# %%
import networkx as nx
import matplotlib.pyplot as plt

def generate_watts_strogatz(n, k, p):
    """
    Generate a Watts–Strogatz small-world network.

    Parameters:
    - n (int): number of nodes
    - k (int): each node is joined with its k nearest neighbors in ring topology (k must be even)
    - p (float): probability of rewiring each edge

    Returns:
    - G (networkx.Graph): the generated WS graph
    """
    return nx.watts_strogatz_graph(n, k, p)

def plot_graph(G, title="Watts–Strogatz Graph"):
    """
    Plot the provided NetworkX graph.

    Parameters:
    - G (networkx.Graph): the graph to plot
    - title (str): title of the plot
    """
    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=8)
    plt.title(title)
    plt.show()

def compute_metrics(G):
    """
    Compute and return some standard small-world metrics:
    - average clustering coefficient
    - average shortest path length (only if the graph is connected)

    Returns:
    - metrics (dict)
    """
    clustering = nx.average_clustering(G)
    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G)
    else:
        # take the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        path_length = nx.average_shortest_path_length(subG)
    return {
        "avg_clustering": clustering,
        "avg_shortest_path": path_length
    }

if __name__ == "__main__":
    # Example parameters
    n = 200   # number of nodes
    k = 4   # each node connected to 6 nearest neighbors
    p = 0.05    # rewiring probability

    # Generate the WS small-world graph
    G = generate_watts_strogatz(n, k, p)

    # Compute metrics
    metrics = compute_metrics(G)
    print(f"Average clustering coefficient: {metrics['avg_clustering']:.4f}")
    print(f"Average shortest path length:   {metrics['avg_shortest_path']:.4f}")

    # Plot the graph
    plot_graph(G, title=f"WS Graph (n={n}, k={k}, p={p})")
    
    plot_degree_distribution_stem(G)
    edge_index = from_networkx(G).edge_index
    node_groups, node_labels, orbits = run_wl_test_and_group_nodes(edge_index, num_nodes=G.number_of_nodes(), num_iterations=100)
    count_orbit_edges(G, orbits)
    
    # plot_orbit_dist(orbits)
    plot_orbit_histogram(orbits)
    custom_labels = {}
    for i, ov in zip(G.nodes(), orbits):
            custom_labels[i] = f"{ov}"
    hash_links_by_orbit(G, orbits)
    plot_graph_with_orbits(G, 
                           None, 
                           orbits, 
                           custom_labels=custom_labels, 
                           figsize=(8, 6), cmap='tab20b')

# %%
import networkx as nx
import matplotlib.pyplot as plt

def generate_chung_lu(degree_sequence):
    """
    Generate a Chung–Lu random graph with a given expected degree sequence.

    Parameters:
    - degree_sequence (list of floats): target expected degrees for each node

    Returns:
    - G (networkx.Graph): the generated Chung–Lu graph
    """
    # The expected_degree_graph function implements the Chung–Lu model
    G = nx.expected_degree_graph(degree_sequence, selfloops=False)
    return G

def compute_metrics(G):
    """
    Compute and return some standard graph metrics:
    - average clustering coefficient
    - average shortest path length (using the largest connected component if necessary)

    Returns:
    - metrics (dict)
    """
    clustering = nx.average_clustering(G)
    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G)
    else:
        # restrict to the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        path_length = nx.average_shortest_path_length(subG)
    return {
        "avg_clustering": clustering,
        "avg_shortest_path": path_length
    }

def plot_graph(G, title="Chung–Lu Random Graph"):
    """
    Plot the provided NetworkX graph.

    Parameters:
    - G (networkx.Graph): the graph to plot
    - title (str): title of the plot
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=100, font_size=8)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Example expected degree sequence
    n = 50
    import numpy as np
    seq = np.random.zipf(a=2.0, size=n)
    degree_sequence = [min(d, n-1) for d in seq]

    # Generate the Chung–Lu graph
    G = generate_chung_lu(degree_sequence)

    # Compute and print metrics
    metrics = compute_metrics(G)
    print(f"Average clustering coefficient: {metrics['avg_clustering']:.4f}")
    print(f"Average shortest path length:   {metrics['avg_shortest_path']:.4f}")

    # Plot the graph
    plot_graph(G, title=f"Chung–Lu Graph (n={n})")

    
    plot_degree_distribution_stem(G)
    edge_index = from_networkx(G).edge_index
    node_groups, node_labels, orbits = run_wl_test_and_group_nodes(edge_index, num_nodes=G.number_of_nodes(), num_iterations=100)
    count_orbit_edges(G, orbits)
    
    # plot_orbit_dist(orbits)
    plot_orbit_histogram(orbits)
    custom_labels = {}
    for i, ov in zip(G.nodes(), orbits):
            custom_labels[i] = f"{ov}"
    hash_links_by_orbit(G, orbits)
    plot_graph_with_orbits(G, 
                           None, 
                           orbits, 
                           custom_labels=custom_labels, 
                           figsize=(8, 6), cmap='tab20b')




