def pagerank(graph, d=0.85, tol=1e-6, max_iter=100):
    """
    Compute PageRank scores for all nodes in an undirected (or directed) graph.

    PageRank is defined by the recurrence:
        PR(v) = (1 - d)/N  +  d * (sum_{u -> v} (PR(u) / outdeg(u)) + sum_{u is sink} (PR(u) / N))

    where:
        d        = damping factor (default 0.85)
        N        = number of nodes
        u -> v   = an edge from u to v
        outdeg(u)= number of outgoing edges from u

    PageRank models a "random surfer":
        - With probability d, follow an outgoing link.
        - With probability (1-d), teleport uniformly to any node.
    This guarantees convergence even in graphs with sinks or disconnected components.

    Parameters
    ----------
    graph : dict
        Adjacency list representation of a graph.

    d : float, optional
        Damping factor. Standard choice is 0.85.

    tol : float, optional
        Convergence tolerance. If L1 difference between iterations falls below tol, the algorithm assumes convergence.

    max_iter : int, optional
        Maximum number of iterations before giving up on convergence.

    Returns
    -------
    rank : dict
        Dictionary mapping each node to its PageRank score.
        The scores sum to 1 (approximately).

    Notes
    -----
    • Sink nodes (outdegree = 0) distribute their PageRank equally to all nodes.
      This is necessary for convergence.

    • This implementation is standard "power iteration" PageRank.

    • Convergence is checked using L1 norm: sum |new - old|.
    """

    nodes = list(graph.keys())
    n = len(nodes)

    # Initial uniform distribution
    rank = {node: 1.0 / n for node in nodes}

    # Precompute outdegrees
    outdeg = {node: len(graph[node]) for node in nodes}

    for _ in range(max_iter):

        # Start with teleportation term for all nodes
        new_rank = {node: (1 - d) / n for node in nodes}

        # Distribute PageRank from each node
        for node in nodes:
            if outdeg[node] == 0:
                # Sink node: distribute PR evenly to all nodes
                share = d * rank[node] / n
                for target in nodes:
                    new_rank[target] += share

            else:
                # Distribute PR along outgoing edges
                share = d * rank[node] / outdeg[node]
                for neighbor in graph[node]:
                    new_rank[neighbor] += share

        # Check convergence via L1 difference
        diff = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        if diff < tol:
            return new_rank

        rank = new_rank

    # Return even if not fully converged
    return rank
