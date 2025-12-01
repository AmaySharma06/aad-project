"""
Breadth-First Search (BFS) Algorithm.

BFS explores a graph level by level, visiting all neighbors at distance k
before visiting neighbors at distance k+1. This guarantees shortest paths
in unweighted graphs.

Applications:
- Finding shortest paths in unweighted graphs
- Connected component detection
- Level-order traversal
- Finding all nodes within a certain distance
"""

from collections import deque
from typing import Dict, List, Tuple, Set, Optional


def bfs_traversal(
    graph: Dict[int, List[int]], 
    start: int
) -> Tuple[List[int], Dict[int, int]]:
    """
    Perform BFS traversal from a starting node.

    Visits all reachable nodes in breadth-first order, returning the
    traversal order and distances from the start node.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation of the graph.

    start : int
        The starting node for traversal.

    Returns
    -------
    tuple
        (traversal_order, distances)
        - traversal_order: List of nodes in the order they were visited
        - distances: Dict mapping each node to its distance from start

    Time Complexity
    ---------------
    O(V + E) where V = vertices, E = edges

    Space Complexity
    ----------------
    O(V) for the queue and visited set

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
    >>> order, dist = bfs_traversal(graph, 0)
    >>> order
    [0, 1, 2, 3]
    >>> dist
    {0: 0, 1: 1, 2: 1, 3: 2}

    Notes
    -----
    BFS uses a queue (FIFO) to explore nodes. This ensures that:
    1. All nodes at distance k are visited before nodes at distance k+1
    2. The first time we reach a node, it's via a shortest path
    
    The algorithm:
    1. Initialize queue with start node
    2. While queue is not empty:
       a. Dequeue a node u
       b. For each unvisited neighbor v of u:
          - Mark v as visited
          - Record distance[v] = distance[u] + 1
          - Enqueue v
    """
    # Track visited nodes and distances
    visited = {start}
    distances = {start: 0}
    traversal_order = [start]

    # Queue for BFS (FIFO)
    queue = deque([start])

    while queue:
        current = queue.popleft()

        # Explore all neighbors
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current] + 1
                traversal_order.append(neighbor)
                queue.append(neighbor)

    return traversal_order, distances


def bfs_shortest_path(
    graph: Dict[int, List[int]], 
    start: int, 
    end: int
) -> Tuple[Optional[List[int]], int]:
    """
    Find the shortest path between two nodes using BFS.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    start : int
        Source node.

    end : int
        Destination node.

    Returns
    -------
    tuple
        (path, distance)
        - path: List of nodes from start to end (None if unreachable)
        - distance: Shortest distance (-1 if unreachable)

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1, 4], 4: [3]}
    >>> path, dist = bfs_shortest_path(graph, 0, 4)
    >>> path
    [0, 1, 3, 4]
    >>> dist
    3

    Notes
    -----
    Uses BFS with parent tracking to reconstruct the path.
    Since BFS explores by levels, the first path found is shortest.
    """
    if start == end:
        return [start], 0

    if start not in graph or end not in graph:
        return None, -1

    # Track visited and parent for path reconstruction
    visited = {start}
    parent = {start: None}

    queue = deque([start])

    while queue:
        current = queue.popleft()

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current

                # Found the destination
                if neighbor == end:
                    # Reconstruct path
                    path = []
                    node = end
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path, len(path) - 1

                queue.append(neighbor)

    # End not reachable
    return None, -1


def bfs_all_paths(
    graph: Dict[int, List[int]], 
    start: int
) -> Dict[int, int]:
    """
    Compute shortest path distances from start to all reachable nodes.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    start : int
        Source node.

    Returns
    -------
    dict[int, int]
        Mapping from each node to its shortest distance from start.
        Unreachable nodes have distance = infinity.

    Examples
    --------
    >>> graph = {0: [1], 1: [0, 2], 2: [1], 3: []}  # Node 3 is isolated
    >>> distances = bfs_all_paths(graph, 0)
    >>> distances[2]
    2
    >>> distances[3]
    inf
    """
    # Initialize all distances as infinity
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    visited = {start}
    queue = deque([start])

    while queue:
        current = queue.popleft()

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


def bfs_connected_components(graph: Dict[int, List[int]]) -> List[Set[int]]:
    """
    Find all connected components using BFS.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    Returns
    -------
    list[set[int]]
        List of sets, each containing nodes in one component.

    Time Complexity
    ---------------
    O(V + E)

    Examples
    --------
    >>> graph = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
    >>> components = bfs_connected_components(graph)
    >>> len(components)
    3
    """
    visited = set()
    components = []

    for node in graph:
        if node in visited:
            continue

        # BFS from this node to find its component
        component = set()
        queue = deque([node])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)
            component.add(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

        components.append(component)

    return components


def bfs_level_order(
    graph: Dict[int, List[int]], 
    start: int
) -> List[List[int]]:
    """
    Return nodes grouped by their BFS level (distance from start).

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    start : int
        Source node.

    Returns
    -------
    list[list[int]]
        levels[i] contains all nodes at distance i from start.

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2, 4], 4: [3]}
    >>> levels = bfs_level_order(graph, 0)
    >>> levels
    [[0], [1, 2], [3], [4]]
    """
    visited = {start}
    levels = [[start]]
    current_level = [start]

    while current_level:
        next_level = []

        for node in current_level:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.append(neighbor)

        if next_level:
            levels.append(next_level)

        current_level = next_level

    return levels


if __name__ == "__main__":
    # Demo
    print("=== BFS Algorithm Demo ===\n")

    # Sample graph
    #     0
    #    / \
    #   1   2
    #   |   |
    #   3---+
    #   |
    #   4
    graph = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [1, 2, 4],
        4: [3]
    }

    print("Graph structure:")
    for node, neighbors in graph.items():
        print(f"  {node} -> {neighbors}")

    # BFS traversal
    print("\n--- BFS Traversal from node 0 ---")
    order, distances = bfs_traversal(graph, 0)
    print(f"Traversal order: {order}")
    print(f"Distances: {distances}")

    # Shortest path
    print("\n--- Shortest Path from 0 to 4 ---")
    path, dist = bfs_shortest_path(graph, 0, 4)
    print(f"Path: {path}")
    print(f"Distance: {dist}")

    # Level order
    print("\n--- Level Order (BFS layers) ---")
    levels = bfs_level_order(graph, 0)
    for i, level in enumerate(levels):
        print(f"  Level {i}: {level}")

    # Connected components
    print("\n--- Connected Components ---")
    # Add an isolated node
    graph_disconnected = {
        0: [1],
        1: [0],
        2: [3],
        3: [2],
        4: []
    }
    components = bfs_connected_components(graph_disconnected)
    print(f"Graph: {graph_disconnected}")
    print(f"Components: {components}")
