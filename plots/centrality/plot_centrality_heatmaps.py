"""
Centrality Heatmap Visualizations.

Generates heatmap plots showing centrality scores across a structured graph
that highlights differences between Degree, Closeness, Betweenness, and PageRank.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import networkx as nx
import random

from centrality.degree_centrality import degree_centrality
from centrality.harmonic_closeness import harmonic_closeness
from centrality.betweenness_centrality import brandes_betweenness
from centrality.pagerank import pagerank


def generate_showcase_graph():
    """
    Generate a structured graph ideal for demonstrating differences
    across Degree, Closeness, Betweenness, and PageRank centralities.

    Structure:
    - Community A: nodes 0-9 (dense)
    - Community B: nodes 10-19 (dense)
    - Bridge: 5 <-> 12
    - Hub: node 0 connects to all nodes in A + leaf nodes
    - Peripheral chain: 20-23 attached to node 5
    - Leaf nodes: 24-27 attached to node 0
    """
    G = nx.Graph()

    # Community A (0–9)
    for i in range(10):
        for j in range(i + 1, 10):
            if random.random() < 0.5:
                G.add_edge(i, j)

    # Community B (10–19)
    for i in range(10, 20):
        for j in range(i + 1, 20):
            if random.random() < 0.5:
                G.add_edge(i, j)

    # Bridge
    G.add_edge(5, 12)

    # Hub node inside community A
    for u in range(1, 10):
        G.add_edge(0, u)

    # Peripheral chain (20–23)
    G.add_edge(5, 20)
    G.add_edge(20, 21)
    G.add_edge(21, 22)
    G.add_edge(22, 23)

    # Leaves on hub node
    for leaf in [24, 25, 26, 27]:
        G.add_edge(0, leaf)

    return G


def normalize(scores):
    """Normalize centrality scores to [0,1] for color mapping."""
    vals = list(scores.values())
    low, high = min(vals), max(vals)

    if high == low:
        return {k: 0.5 for k in scores}

    return {k: (v - low) / (high - low) for k, v in scores.items()}


def plot_heatmap(G, pos, scores, title, outpath):
    """Plot a single centrality heatmap using proper mappable nodes."""
    norm_scores = normalize(scores)
    values = [norm_scores[n] for n in G.nodes()]

    plt.figure(figsize=(7, 7))

    # Nodes
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=values,
        node_size=260,
        cmap="Reds"
    )

    # Edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.4, width=1)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=7)

    # Colorbar
    plt.colorbar(nodes, label="Normalized score")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def generate_centrality_heatmaps(output_dir="plots/centrality/images"):
    """Generate heatmaps for all four centrality measures on the same graph."""
    random.seed(123)

    G = generate_showcase_graph()
    adj_list = {node: list(G.neighbors(node)) for node in G.nodes()}

    # Fix layout for consistency
    pos = nx.spring_layout(G, seed=123)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Compute centralities
    print("Computing centrality measures...")
    degree_scores = degree_centrality(adj_list)
    harmonic_scores = harmonic_closeness(adj_list)
    betweenness_scores = brandes_betweenness(adj_list)
    pagerank_scores = pagerank(adj_list)

    # Save four separate heatmaps
    print("Generating individual heatmaps...")
    plot_heatmap(G, pos, degree_scores,
                 "Degree Centrality Heatmap",
                 f"{output_dir}/degree_heatmap.png")

    plot_heatmap(G, pos, harmonic_scores,
                 "Harmonic Closeness Heatmap",
                 f"{output_dir}/harmonic_heatmap.png")

    plot_heatmap(G, pos, betweenness_scores,
                 "Betweenness Centrality Heatmap",
                 f"{output_dir}/betweenness_heatmap.png")

    plot_heatmap(G, pos, pagerank_scores,
                 "PageRank Heatmap",
                 f"{output_dir}/pagerank_heatmap.png")

    # Combined 2x2 grid
    print("Generating combined heatmap grid...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    pairs = [
        ("Degree", degree_scores),
        ("Harmonic Closeness", harmonic_scores),
        ("Betweenness", betweenness_scores),
        ("PageRank", pagerank_scores)
    ]

    for (title, scores), axis in zip(pairs, axes.flatten()):
        norm = normalize(scores)
        vals = [norm[n] for n in G.nodes()]

        nodes = nx.draw_networkx_nodes(
            G, pos, node_color=vals, node_size=220, cmap="Reds", ax=axis
        )
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=1, alpha=0.4, ax=axis)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=axis)

        axis.set_title(title)
        axis.set_axis_off()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/centrality_heatmap_grid.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    print(f"All heatmaps saved to {output_dir}/")


if __name__ == "__main__":
    generate_centrality_heatmaps()
