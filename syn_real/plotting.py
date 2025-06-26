import matplotlib.pyplot as plt 
import networkx as nx

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
def plot_orbit_dist(orbits):
    node_groups = {}
    for node, label in enumerate(orbits):
        if label not in node_groups:
            node_groups[label] = []
        node_groups[label].append(node)
        
    group_sizes = np.array([len(group) for group in node_groups.values()])
    plt.figure(figsize=(6, 4))
    plt.hist(group_sizes, bins=range(min(group_sizes), max(group_sizes) + 2), align='left', rwidth=0.8)
    plt.xlabel('Orbit Size')
    plt.ylabel('Frequency of Orbit Sizes')
    plt.title('Histogram of Orbit Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    # plt.savefig(f"{name}_distribution_d{depth}.pdf")
    plt.close()



def plot_orbit_histogram(orbits):
    plt.figure(figsize=(6, 4))
    plt.hist(orbits, bins=range(min(orbits), max(orbits) + 2), align='left', rwidth=0.8)
    plt.xlabel('Orbit Label')
    plt.ylabel('Frequency')
    plt.title('Orbit Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    # plt.savefig(f"{name}_distribution_d{depth}.pdf")
    plt.close()


# %%
def plot_triangular_graph(G, 
                          orbits, 
                          custom_labels=None, 
                          figsize=(8, 6), 
                          cmap='Blues'):
    """
    Plots a triangular lattice graph with node coloring based on orbit labels.

    Parameters:
        G (nx.Graph): The triangular lattice graph to plot.
        orbits (list or dict): Orbit label for each node.
        custom_labels (dict, optional): Custom labels for nodes.
        figsize (tuple): Size of the figure.
        cmap (str): Matplotlib colormap name.
    """

    plt.figure(figsize=(6, 6))
    labels = {}
    for i, node in enumerate(G.nodes()):
        labels.update({node: orbits[i].item()})
    nx.draw(
        G,
        # node_color=orbits, 
        with_labels=True,
        labels=labels,
        node_size=1500,       # Increase node size (default is ~300–600)
        edgecolors='black',
        font_weight='bold',
        font_size=14          
    )

    plt.title("Graph Colored by Float Values", fontsize=16) 
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    plt.close()




def plot_unique_edge_class(edge_class_counts: dict):
    unique_orbit_seq = sorted(edge_class_counts.values(), reverse=True)
    print(f"Edge class counts: {unique_orbit_seq[:10]}")
    print(f"Unique Edge classes: {len(edge_class_counts.keys())}")

    plt.figure()
    markerline, stemlines, _ = plt.stem(
        unique_orbit_seq,
        markerfmt='bo',
        basefmt=' '
    )
    markerline.set_markerfacecolor('none')
    stemlines.set_linewidth(0.5)
    plt.title('Automorphic Edge Distribution')
    plt.xlabel("Automorphism Edge Classes")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    


# %%
def plot_orbit_histogram(orbits):
    degrees = [d for _, d in G.degree()]
    degrees = sorted(degrees, reverse=True)

    plt.figure()
    markerline, stemlines, _ = plt.stem(
        degrees,
        markerfmt='bo',
        basefmt=' '
    )
    markerline.set_markerfacecolor('none')
    stemlines.set_linewidth(0.5)

    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    return 
    plt.figure(figsize=(6, 4))
    plt.hist(orbits, bins=range(min(orbits), max(orbits) + 2), align='left', rwidth=0.8)
    plt.xlabel('Orbit Label')
    plt.ylabel('Frequency')
    plt.title('Histogram of Orbit Labels')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    # plt.savefig(f"{name}_distribution_d{depth}.pdf")
    plt.close()



def plot_sorted_orbit_hist(orbits):
    degrees = [d for _, d in G.degree()]
    degrees = sorted(degrees, reverse=True)

    plt.figure()
    markerline, stemlines, _ = plt.stem(
        degrees,
        markerfmt='bo',
        basefmt=' '
    )
    markerline.set_markerfacecolor('none')
    stemlines.set_linewidth(0.5)

    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()