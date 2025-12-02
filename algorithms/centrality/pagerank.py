"""
PageRank Algorithm.

PageRank measures the importance of nodes in a graph based on the structure
of incoming links. Originally developed by Larry Page and Sergey Brin at Google
to rank web pages.

The core idea: A node is important if it's linked to by other important nodes.

Mathematical Definition:
    PR(v) = (1 - d)/N + d * Σ (PR(u) / out_degree(u))
                            u ∈ in_neighbors(v)

where:
    - d = damping factor (typically 0.85)
    - N = total number of nodes
    - The sum is over all nodes u that link to v

The damping factor models a "random surfer" who:
    - With probability d, follows a link
    - With probability (1-d), jumps to a random page
"""

from typing import Dict, List


def pagerank(
    graph: Dict[int, List[int]], 
    damping: float = 0.85, 
    tolerance: float = 1e-6, 
    max_iterations: int = 100
) -> Dict[int, float]:
    """
    Compute PageRank scores for all nodes using power iteration.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation of the graph.
        For undirected graphs, each edge appears in both directions.

    damping : float, optional (default=0.85)
        Damping factor (probability of following a link vs random jump).
        Standard value is 0.85.

    tolerance : float, optional (default=1e-6)
        Convergence tolerance. Algorithm stops when L1 difference
        between iterations falls below this threshold.

    max_iterations : int, optional (default=100)
        Maximum number of iterations before stopping.

    Returns
    -------
    dict[int, float]
        Dictionary mapping each node to its PageRank score.
        Scores sum to approximately 1.

    Time Complexity
    ---------------
    O(k * (V + E)) where k = number of iterations until convergence

    Space Complexity
    ----------------
    O(V)

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0], 2: [0, 1]}
    >>> scores = pagerank(graph)
    >>> sum(scores.values())  # Should be ~1.0
    1.0

    Notes
    -----
    **The Random Surfer Model:**
    
    Imagine a person randomly surfing the web:
    1. At each step, with probability d, they click a random link on the page
    2. With probability (1-d), they get bored and jump to a random page
    
    PageRank represents the probability of being at each page after
    infinite random steps.

    **Handling Sink Nodes:**
    
    Nodes with no outgoing edges (sinks) would cause probability to "leak".
    We handle this by distributing a sink node's PageRank equally to all nodes
    (as if the surfer teleports when hitting a dead end).

    **Convergence:**
    
    The algorithm uses power iteration and is guaranteed to converge
    because the transition matrix is stochastic (rows sum to 1) and
    the random jump ensures ergodicity (all states reachable).
    """
    nodes = list(graph.keys())
    n = len(nodes)

    if n == 0:
        return {}

    # Initialize: uniform probability distribution
    rank = {node: 1.0 / n for node in nodes}

    # Precompute out-degrees
    out_degree = {node: len(graph[node]) for node in nodes}

    # Power iteration
    for iteration in range(max_iterations):
        # New rank starts with teleportation probability
        new_rank = {node: (1 - damping) / n for node in nodes}

        # Accumulate rank contributions
        for node in nodes:
            if out_degree[node] == 0:
                # Sink node: distribute rank to all nodes
                contribution = damping * rank[node] / n
                for target in nodes:
                    new_rank[target] += contribution
            else:
                # Normal node: distribute rank along edges
                contribution = damping * rank[node] / out_degree[node]
                for neighbor in graph[node]:
                    new_rank[neighbor] += contribution

        # Check convergence (L1 norm)
        diff = sum(abs(new_rank[node] - rank[node]) for node in nodes)

        if diff < tolerance:
            return new_rank

        rank = new_rank

    # Return current state even if not fully converged
    return rank


def pagerank_top_k(
    graph: Dict[int, List[int]], 
    k: int = 10,
    **kwargs
) -> List[tuple]:
    """
    Get the top-k nodes by PageRank score.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    k : int, optional (default=10)
        Number of top nodes to return.

    **kwargs
        Additional arguments passed to pagerank().

    Returns
    -------
    list[tuple]
        List of (node, score) tuples, sorted by score descending.

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
    >>> top = pagerank_top_k(graph, k=2)
    >>> len(top)
    2
    """
    scores = pagerank(graph, **kwargs)
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:k]


if __name__ == "__main__":
    # Demo
    print("=== PageRank Demo ===\n")

    # Example graph
    #     0 <-> 1
    #     ^     ^
    #     |     |
    #     v     v
    #     2 <-> 3
    graph = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [1, 2]
    }

    print("Graph structure:")
    for node, neighbors in graph.items():
        print(f"  {node} -> {neighbors}")

    # Compute PageRank
    print("\n--- PageRank Scores ---")
    scores = pagerank(graph)
    for node, score in sorted(scores.items()):
        print(f"  Node {node}: {score:.6f}")

    print(f"\nSum of scores: {sum(scores.values()):.6f}")

    # Top-k
    print("\n--- Top Nodes ---")
    top = pagerank_top_k(graph, k=2)
    for node, score in top:
        print(f"  Node {node}: {score:.6f}")

    # Example with a "hub" node
    print("\n--- Hub Node Example ---")
    hub_graph = {
        0: [1, 2, 3, 4],  # Hub connected to all
        1: [0],
        2: [0],
        3: [0],
        4: [0]
    }
    hub_scores = pagerank(hub_graph)
    print("Graph: Star with node 0 as hub")
    for node, score in sorted(hub_scores.items()):
        print(f"  Node {node}: {score:.6f}")
    print("(Node 0 should have highest score as the central hub)")

    # Example with sink node
    print("\n--- Sink Node Example ---")
    sink_graph = {
        0: [1],
        1: [2],
        2: [],  # Sink: no outgoing edges
    }
    sink_scores = pagerank(sink_graph)
    print("Graph: 0 -> 1 -> 2 (sink)")
    for node, score in sorted(sink_scores.items()):
        print(f"  Node {node}: {score:.6f}")
