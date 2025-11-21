def degree_centrality(graph: dict, normalize: bool = False) -> dict:
    """
    Compute the Degree Centrality for each node in an undirected graph.

    Parameters
    ----------
    graph : dict
        Adjacency list representation of the graph.
        Example:
            {
                0: [1, 2],
                1: [0],
                2: [0]
                ...
            }
        Each key is a node and the value is the list of neighboring nodes.

    normalize : bool, optional (default = False)
        If True, returns the normalized degree centrality,
        i.e., degree divided by (n - 1), producing values in [0, 1].
        Normalization is useful when comparing graphs of different sizes.

    Returns
    -------
    dict
        Dictionary mapping each node to its (normalized or raw) degree centrality.

    Notes
    -----
    - Degree centrality is the simplest centrality measure.
    - Raw degree = number of neighbors.
    - Normalized degree = raw_degree / (n - 1)
      This allows comparison across graphs with different sizes.
    """
    n = len(graph)

    centrality = {}

    for node in graph:
        raw_degree = len(graph[node])
        centrality[node] = raw_degree

    if normalize:
        # Avoid division by zero for trivial graphs
        if n > 1:
            for node in centrality:
                centrality[node] /= (n - 1)

    return centrality
