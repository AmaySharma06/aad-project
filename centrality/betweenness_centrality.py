from collections import deque

def brandes_betweenness(graph):
    """
    Compute betweenness centrality for all nodes using Brandes' algorithm.

    Betweenness centrality of a node v is defined as:
        C_B(v) = sum over all s ≠ v ≠ t of ( n_st(v) / n_st )

        where:
            n_st      = number of shortest paths from s to t
            n_st(v)   = number of those paths that pass through v

    Brandes' algorithm reduces the computation from O(n^3) to O(nm) for unweighted graphs using:
        - a BFS from each source s
        - predecessor tracking
        - dependency accumulation

    Parameters
    ----------
    graph : dict
        Adjacency list representation of an unweighted graph.

    Returns
    -------
    betweenness : dict
        Dictionary mapping each node to its normalized betweenness value.
        Values lie in [0, 1].

    Notes
    -----
    - The normalization factor 1 / ((n-1)(n-2)/2)
      ensures values are scale-free and comparable across graph sizes.

    - Unreachable nodes between (s, t) pairs naturally do not contribute,
      because Brandes' algorithm counts only reachable shortest paths.

    - This implementation is for undirected graphs.
      For directed graphs, the normalization constant changes.
    """

    nodes = list(graph.keys())
    betweenness = {v: 0.0 for v in nodes}

    # Iterate over all source nodes
    for s in nodes:
        stack = []
        pred = {v: [] for v in nodes}       # predecessors of v on shortest paths
        sigma = {v: 0 for v in nodes}       # number of shortest paths from s to v
        dist  = {v: -1 for v in nodes}      # BFS distance from s

        sigma[s] = 1
        dist[s] = 0

        # BFS
        queue = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)

            for w in graph[v]:

                # First time visiting w
                if dist[w] < 0:  
                    dist[w] = dist[v] + 1
                    queue.append(w)

                # Shortest path to w through v
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Dependency accumulation
        delta = {v: 0 for v in nodes}

        # Process nodes in reverse BFS order
        while stack:
            w = stack.pop()

            # Accumulate dependencies of predecessors
            for v in pred[w]:
                if sigma[w] > 0: # avoid division by zero
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])

            # Exclude the source node itself
            if w != s:
                betweenness[w] += delta[w]

    # Normalization
    n = len(nodes)
    if n > 2:
        scale = 1 / (((n - 1) * (n - 2)) / 2)
        for v in betweenness:
            betweenness[v] *= scale

    return betweenness
