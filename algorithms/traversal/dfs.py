"""
Depth-First Search (DFS) Algorithm.

DFS explores as far as possible along each branch before backtracking.
It uses a stack (or recursion) to remember which nodes to visit next.

Applications:
- Cycle detection
- Topological sorting
- Finding connected components
- Pathfinding in mazes
- Detecting bipartiteness
"""

from typing import Dict, List, Tuple, Set, Optional


def dfs_traversal(
    graph: Dict[int, List[int]], 
    start: int,
    visited: Set[int] = None
) -> List[int]:
    """
    Perform DFS traversal from a starting node (recursive implementation).

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation of the graph.

    start : int
        The starting node for traversal.

    visited : set[int], optional
        Set of already visited nodes (used internally for recursion).

    Returns
    -------
    list[int]
        List of nodes in the order they were visited (DFS order).

    Time Complexity
    ---------------
    O(V + E) where V = vertices, E = edges

    Space Complexity
    ----------------
    O(V) for the recursion stack and visited set

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
    >>> dfs_traversal(graph, 0)
    [0, 1, 3, 2]  # or another valid DFS order

    Notes
    -----
    DFS uses a stack (implicit via recursion) to explore nodes:
    1. Visit the current node
    2. Recursively visit each unvisited neighbor
    3. Backtrack when no unvisited neighbors remain
    
    Unlike BFS, DFS does NOT guarantee shortest paths, but it uses
    less memory for graphs with large branching factors.
    """
    if visited is None:
        visited = set()

    traversal = []

    def _dfs_recursive(node: int):
        if node in visited:
            return

        visited.add(node)
        traversal.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                _dfs_recursive(neighbor)

    _dfs_recursive(start)
    return traversal


def dfs_iterative(
    graph: Dict[int, List[int]], 
    start: int
) -> List[int]:
    """
    Perform DFS traversal using an explicit stack (iterative implementation).

    This avoids potential stack overflow for very deep graphs.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    start : int
        Starting node.

    Returns
    -------
    list[int]
        Nodes in DFS order.

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
    >>> dfs_iterative(graph, 0)
    [0, 2, 1, 3]  # Note: order may differ from recursive due to stack LIFO

    Notes
    -----
    The iterative version processes neighbors in reverse order compared to
    recursive DFS because we push all neighbors and pop the last one first.
    To match recursive behavior, reverse the neighbor list before pushing.
    """
    visited = set()
    traversal = []
    stack = [start]

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        traversal.append(node)

        # Add neighbors to stack (reverse to match recursive order)
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)

    return traversal


def detect_cycle(graph: Dict[int, List[int]]) -> Tuple[bool, Optional[List[int]]]:
    """
    Detect if an undirected graph contains a cycle.

    Uses DFS with parent tracking. A cycle exists if we encounter
    a visited node that isn't our parent.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation of an undirected graph.

    Returns
    -------
    tuple
        (has_cycle, cycle_path)
        - has_cycle: True if a cycle exists
        - cycle_path: List of nodes forming a cycle (None if no cycle)

    Time Complexity
    ---------------
    O(V + E)

    Examples
    --------
    >>> graph_with_cycle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    >>> has_cycle, path = detect_cycle(graph_with_cycle)
    >>> has_cycle
    True

    >>> graph_no_cycle = {0: [1], 1: [0, 2], 2: [1]}
    >>> has_cycle, path = detect_cycle(graph_no_cycle)
    >>> has_cycle
    False

    Notes
    -----
    For undirected graphs, we need to track the parent of each node.
    An edge back to a visited node (that isn't our parent) indicates a cycle.
    
    For directed graphs, the algorithm would need to track "in-progress"
    vs "completed" nodes (using colors: white, gray, black).
    """
    visited = set()
    parent = {}

    def _dfs_cycle(node: int, par: Optional[int]) -> Optional[List[int]]:
        visited.add(node)
        parent[node] = par

        for neighbor in graph[node]:
            if neighbor not in visited:
                # Recurse
                cycle = _dfs_cycle(neighbor, node)
                if cycle is not None:
                    return cycle

            elif neighbor != par:
                # Found a back edge -> cycle detected
                # Reconstruct the cycle
                cycle = [neighbor]
                current = node
                while current != neighbor:
                    cycle.append(current)
                    current = parent[current]
                cycle.append(neighbor)
                return cycle

        return None

    # Check each component
    for node in graph:
        if node not in visited:
            cycle = _dfs_cycle(node, None)
            if cycle is not None:
                return True, cycle

    return False, None


def dfs_connected_components(graph: Dict[int, List[int]]) -> List[Set[int]]:
    """
    Find all connected components using DFS.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    Returns
    -------
    list[set[int]]
        List of sets, each containing nodes in one component.

    Examples
    --------
    >>> graph = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
    >>> components = dfs_connected_components(graph)
    >>> len(components)
    3
    """
    visited = set()
    components = []

    for node in graph:
        if node not in visited:
            # DFS from this node
            component = set()
            stack = [node]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                component.add(current)

                for neighbor in graph[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            components.append(component)

    return components


def dfs_path(
    graph: Dict[int, List[int]], 
    start: int, 
    end: int
) -> Optional[List[int]]:
    """
    Find a path between two nodes using DFS.

    Note: This finds A path, not necessarily the shortest path.
    Use BFS for shortest paths.

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
    list[int] or None
        A path from start to end, or None if no path exists.

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1, 4], 4: [3]}
    >>> path = dfs_path(graph, 0, 4)
    >>> path is not None
    True
    >>> path[0] == 0 and path[-1] == 4
    True
    """
    if start == end:
        return [start]

    visited = set()
    parent = {start: None}
    stack = [start]

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        if node == end:
            # Reconstruct path
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path

        for neighbor in graph[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                stack.append(neighbor)

    return None  # No path found


def dfs_preorder_postorder(
    graph: Dict[int, List[int]], 
    start: int
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Compute pre-order and post-order numbers for DFS traversal.

    Pre-order number: When a node is first discovered
    Post-order number: When we finish exploring a node (backtrack)

    These are useful for:
    - Detecting back edges
    - Topological sorting
    - Finding strongly connected components

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    start : int
        Starting node.

    Returns
    -------
    tuple
        (pre_order, post_order)
        - pre_order[v] = time when v was first visited
        - post_order[v] = time when we finished exploring v
    """
    pre_order = {}
    post_order = {}
    time = [0]  # Use list to allow modification in nested function
    visited = set()

    def _dfs(node: int):
        visited.add(node)
        pre_order[node] = time[0]
        time[0] += 1

        for neighbor in graph[node]:
            if neighbor not in visited:
                _dfs(neighbor)

        post_order[node] = time[0]
        time[0] += 1

    _dfs(start)
    return pre_order, post_order


if __name__ == "__main__":
    # Demo
    print("=== DFS Algorithm Demo ===\n")

    # Sample graph
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

    # DFS traversal (recursive)
    print("\n--- DFS Traversal (recursive) from node 0 ---")
    order = dfs_traversal(graph, 0)
    print(f"Traversal order: {order}")

    # DFS traversal (iterative)
    print("\n--- DFS Traversal (iterative) from node 0 ---")
    order_iter = dfs_iterative(graph, 0)
    print(f"Traversal order: {order_iter}")

    # Path finding
    print("\n--- DFS Path from 0 to 4 ---")
    path = dfs_path(graph, 0, 4)
    print(f"Path found: {path}")

    # Cycle detection
    print("\n--- Cycle Detection ---")
    
    # Graph with cycle
    cycle_graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    has_cycle, cycle_path = detect_cycle(cycle_graph)
    print(f"Graph {cycle_graph}")
    print(f"Has cycle: {has_cycle}, Cycle: {cycle_path}")

    # Tree (no cycle)
    tree = {0: [1, 2], 1: [0], 2: [0, 3], 3: [2]}
    has_cycle, cycle_path = detect_cycle(tree)
    print(f"\nGraph {tree}")
    print(f"Has cycle: {has_cycle}")

    # Connected components
    print("\n--- Connected Components ---")
    disconnected = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
    components = dfs_connected_components(disconnected)
    print(f"Graph: {disconnected}")
    print(f"Components: {components}")

    # Pre/Post order
    print("\n--- Pre-order and Post-order Numbers ---")
    pre, post = dfs_preorder_postorder(graph, 0)
    print(f"Pre-order:  {pre}")
    print(f"Post-order: {post}")
