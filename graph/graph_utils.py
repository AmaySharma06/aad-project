"""
Graph Utility Functions.

Provides common operations on graphs:
- Format conversions (adjacency matrix ↔ adjacency list)
- BFS shortest paths (reusable by multiple algorithms)
- Connected component detection
- Graph validation
"""

from collections import deque
from typing import Dict, List, Set, Tuple


def adjacency_matrix_to_list(matrix: List[List[int]]) -> Dict[int, List[int]]:
    """
    Convert an adjacency matrix to an adjacency list.

    Parameters
    ----------
    matrix : list[list[int]]
        n x n adjacency matrix where matrix[i][j] = 1 indicates an edge.

    Returns
    -------
    dict[int, list[int]]
        Adjacency list representation.

    Examples
    --------
    >>> matrix = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    >>> adjacency_matrix_to_list(matrix)
    {0: [1, 2], 1: [0], 2: [0]}
    """
    n = len(matrix)
    adjacency = {}

    for i in range(n):
        neighbors = []
        for j in range(n):
            if matrix[i][j] == 1:
                neighbors.append(j)
        adjacency[i] = neighbors

    return adjacency


def adjacency_list_to_matrix(adjacency: Dict[int, List[int]]) -> List[List[int]]:
    """
    Convert an adjacency list to an adjacency matrix.

    Parameters
    ----------
    adjacency : dict[int, list[int]]
        Adjacency list representation.

    Returns
    -------
    list[list[int]]
        n x n adjacency matrix.

    Examples
    --------
    >>> adj = {0: [1, 2], 1: [0], 2: [0]}
    >>> adjacency_list_to_matrix(adj)
    [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    """
    n = len(adjacency)
    matrix = [[0] * n for _ in range(n)]

    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            matrix[node][neighbor] = 1

    return matrix


def bfs_shortest_paths(graph: Dict[int, List[int]], source: int) -> Dict[int, int]:
    """
    Compute shortest-path distances from a source node using BFS.

    BFS visits nodes level by level, guaranteeing shortest paths in
    unweighted graphs.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation of an unweighted graph.

    source : int
        The source node from which to compute shortest paths.

    Returns
    -------
    dict[int, int]
        Dictionary mapping each node to its shortest distance from source.
        Unreachable nodes have distance = float('inf').

    Time Complexity
    ---------------
    O(V + E) where V = number of vertices, E = number of edges.

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
    >>> bfs_shortest_paths(graph, 0)
    {0: 0, 1: 1, 2: 1, 3: 2}
    """
    # Initialize all distances as infinity
    dist = {node: float('inf') for node in graph}
    dist[source] = 0

    # BFS queue
    queue = deque([source])

    while queue:
        u = queue.popleft()

        for v in graph[u]:
            # First time visiting v
            if dist[v] == float('inf'):
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist


def bfs_shortest_paths_with_predecessors(
    graph: Dict[int, List[int]], 
    source: int
) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, int]]:
    """
    BFS that also tracks predecessors and path counts.

    This extended version is used by Brandes' betweenness algorithm
    and Girvan-Newman edge betweenness.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    source : int
        Source node.

    Returns
    -------
    tuple
        (dist, pred, sigma) where:
        - dist[v] = shortest distance from source to v
        - pred[v] = list of predecessors of v on shortest paths
        - sigma[v] = number of shortest paths from source to v
    """
    nodes = list(graph.keys())

    dist = {v: -1 for v in nodes}
    pred = {v: [] for v in nodes}
    sigma = {v: 0 for v in nodes}

    dist[source] = 0
    sigma[source] = 1

    queue = deque([source])

    while queue:
        v = queue.popleft()

        for w in graph[v]:
            # First time visiting w
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                queue.append(w)

            # w is on a shortest path from source through v
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                pred[w].append(v)

    return dist, pred, sigma


def get_connected_components(graph: Dict[int, List[int]]) -> List[Set[int]]:
    """
    Find all connected components in an undirected graph using BFS.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    Returns
    -------
    list[set[int]]
        List of sets, where each set contains nodes in one component.

    Time Complexity
    ---------------
    O(V + E)

    Examples
    --------
    >>> graph = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
    >>> components = get_connected_components(graph)
    >>> len(components)
    3
    """
    visited = set()
    components = []

    for start in graph:
        if start in visited:
            continue

        # BFS from this node
        component = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue

            visited.add(node)
            component.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        components.append(component)

    return components


def is_connected(graph: Dict[int, List[int]]) -> bool:
    """
    Check if the graph is connected.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    Returns
    -------
    bool
        True if all nodes are in one connected component.
    """
    if len(graph) == 0:
        return True

    components = get_connected_components(graph)
    return len(components) == 1


def validate_graph(graph: Dict[int, List[int]]) -> Tuple[bool, str]:
    """
    Validate that a graph is well-formed.

    Checks:
    1. All neighbors are valid nodes in the graph
    2. For undirected graphs: if u->v exists, v->u should exist
    3. No self-loops (optional, controlled by allow_self_loops)

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list to validate.

    Returns
    -------
    tuple
        (is_valid, error_message)
        If valid, returns (True, "").
        If invalid, returns (False, description of error).
    """
    nodes = set(graph.keys())

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            # Check neighbor exists
            if neighbor not in nodes:
                return False, f"Node {node} has neighbor {neighbor} which doesn't exist"

            # Check symmetry (undirected graph)
            if node not in graph[neighbor]:
                return False, f"Edge {node}->{neighbor} exists but {neighbor}->{node} doesn't"

            # Check for self-loops
            if neighbor == node:
                return False, f"Self-loop detected at node {node}"

    return True, ""


def get_edge_list(graph: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    """
    Convert adjacency list to edge list (no duplicates for undirected).

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    Returns
    -------
    list[tuple[int, int]]
        List of edges as (u, v) tuples where u < v.
    """
    edges = set()
    for u, neighbors in graph.items():
        for v in neighbors:
            edge = (min(u, v), max(u, v))
            edges.add(edge)
    return list(edges)


def add_edge(graph: Dict[int, List[int]], u: int, v: int) -> None:
    """
    Add an undirected edge to the graph (in-place).

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list (modified in place).

    u, v : int
        Nodes to connect.
    """
    if v not in graph[u]:
        graph[u].append(v)
    if u not in graph[v]:
        graph[v].append(u)


def remove_edge(graph: Dict[int, List[int]], u: int, v: int) -> None:
    """
    Remove an undirected edge from the graph (in-place).

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list (modified in place).

    u, v : int
        Nodes to disconnect.
    """
    if v in graph[u]:
        graph[u].remove(v)
    if u in graph[v]:
        graph[v].remove(u)


def copy_graph(graph: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    Create a deep copy of an adjacency list.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Original adjacency list.

    Returns
    -------
    dict[int, list[int]]
        New adjacency list with copied data.
    """
    return {node: neighbors.copy() for node, neighbors in graph.items()}


if __name__ == "__main__":
    # Demo
    print("=== Graph Utilities Demo ===\n")

    # Create a sample graph
    graph = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1],
        3: [1]
    }

    print("Original graph:", graph)

    # Validate
    is_valid, error = validate_graph(graph)
    print(f"Valid: {is_valid}")

    # BFS shortest paths from node 0
    distances = bfs_shortest_paths(graph, 0)
    print(f"Distances from node 0: {distances}")

    # Connected components
    components = get_connected_components(graph)
    print(f"Connected components: {components}")
    print(f"Is connected: {is_connected(graph)}")

    # Convert to matrix and back
    matrix = adjacency_list_to_matrix(graph)
    print(f"\nAdjacency matrix:")
    for row in matrix:
        print(f"  {row}")

    recovered = adjacency_matrix_to_list(matrix)
    print(f"\nRecovered adjacency list: {recovered}")
