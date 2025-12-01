"""
Random Graph Generator using Erdős-Rényi G(n, p) model.

The Erdős-Rényi model generates a random graph where each possible edge
is included independently with probability p.
"""

import random


def generate_random_graph(n: int, p: float = 0.5, seed: int = None) -> dict:
    """
    Generate a random undirected graph using the Erdős-Rényi G(n, p) model.

    In this model, we start with n isolated nodes and connect each pair of 
    nodes independently with probability p.

    Parameters
    ----------
    n : int
        Number of nodes in the graph. Nodes are labeled 0 to n-1.

    p : float, optional (default=0.5)
        Probability of edge creation between any pair of nodes.
        - p = 0 produces an empty graph (no edges)
        - p = 1 produces a complete graph (all possible edges)
        - p ≈ ln(n)/n is the threshold for graph connectivity

    seed : int, optional
        Random seed for reproducibility. If None, uses system randomness.

    Returns
    -------
    dict
        Adjacency list representation of the graph.
        Example: {0: [1, 2], 1: [0], 2: [0]} means:
            - Node 0 is connected to nodes 1 and 2
            - Node 1 is connected to node 0
            - Node 2 is connected to node 0

    Examples
    --------
    >>> graph = generate_random_graph(5, p=0.5, seed=42)
    >>> len(graph)
    5

    Notes
    -----
    Time Complexity: O(n²) - we consider all possible edges
    Space Complexity: O(n + m) where m is the number of edges
    
    Expected number of edges: n*(n-1)/2 * p
    """
    if seed is not None:
        random.seed(seed)

    # Initialize adjacency list with empty lists for each node
    adjacency = {i: [] for i in range(n)}

    # Consider each possible edge (i, j) where i < j
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                # Add undirected edge: both i->j and j->i
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency


def generate_random_graph_matrix(n: int, p: float = 0.5, seed: int = None) -> list:
    """
    Generate a random undirected graph as an adjacency matrix.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.

    p : float, optional (default=0.5)
        Edge probability.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[list[int]]
        n x n adjacency matrix where matrix[i][j] = 1 if edge exists.
        The matrix is symmetric for undirected graphs.
    """
    if seed is not None:
        random.seed(seed)

    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                matrix[i][j] = 1
                matrix[j][i] = 1

    return matrix


if __name__ == "__main__":
    # Demo
    graph = generate_random_graph(5, p=0.5, seed=42)
    print("Random Graph (adjacency list):")
    for node, neighbors in graph.items():
        print(f"  {node}: {neighbors}")
    
    # Count edges
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    print(f"\nNodes: 5, Edges: {edge_count}")
