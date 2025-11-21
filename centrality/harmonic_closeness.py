from collections import deque

def bfs_shortest_paths(graph, start):
    """
    Compute shortest-path distances from a single source node using BFS.

    Parameters
    ----------
    graph : dict
        Adjacency list representation of an unweighted graph.

    start : int
        The source node from which to compute shortest paths.

    Returns
    -------
    dist : dict
        Dictionary mapping each node -> shortest distance from `start`.
        Unreachable nodes have distance = float('inf').
    """
    # Initialize all distances as Infinity
    dist = {node: float('inf') for node in graph}
    dist[start] = 0

    # Standard BFS queue
    queue = deque([start])

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            # Visit node v for the first time
            if dist[v] == float('inf'):
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist


def harmonic_closeness(graph: dict) -> dict:
    """
    Compute Harmonic Closeness Centrality for every node in the graph.

    Harmonic Closeness for a node v is defined as:
        Harmonic sum: H(v) = summation(1/d(v,u)) for all nodes u != v
        
        where:
            d(v,u) is the shortest-path distance from v to u.

    - If u is unreachable from v, it contributes 0.
    - Unlike classical closeness, harmonic closeness works correctly
      even for disconnected graphs.

    Parameters
    ----------
    graph : dict
        Adjacency list representation of an unweighted graph.

    Returns
    -------
    H : dict
        Dictionary mapping each node -> harmonic closeness score.
        Higher value = more central.
    """
    H = {}

    for node in graph:
        # Compute distances from this node
        dist = bfs_shortest_paths(graph, node)

        # Harmonic sum: H(v) = summation(1/d(v,u)) for all reachable nodes u != v
        harmonic_sum = 0.0
        for target, d in dist.items():
            if target == node:
                continue
            if d != float('inf'):
                harmonic_sum += 1.0 / d
            # unreachable nodes contribute 0

        H[node] = harmonic_sum

    return H
