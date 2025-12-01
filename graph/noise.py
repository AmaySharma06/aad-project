"""
Graph Noise Injection.

Provides functions to add/remove random edges to test algorithm robustness.
Used to evaluate how algorithms perform when the graph has noise or errors.
"""

import random
from typing import Dict, List, Tuple
from graph.graph_utils import copy_graph, get_edge_list, add_edge, remove_edge


def add_random_edges(
    graph: Dict[int, List[int]],
    num_edges: int,
    seed: int = None,
    inplace: bool = False
) -> Dict[int, List[int]]:
    """
    Add random edges to a graph.

    Randomly selects pairs of non-adjacent nodes and connects them.
    Useful for testing how algorithms handle noise.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Original adjacency list.

    num_edges : int
        Number of random edges to add.

    seed : int, optional
        Random seed for reproducibility.

    inplace : bool, optional (default=False)
        If True, modify graph in place. Otherwise, return a copy.

    Returns
    -------
    dict[int, list[int]]
        Graph with added edges.

    Notes
    -----
    If num_edges exceeds the number of possible new edges, adds as many
    as possible.
    """
    if seed is not None:
        random.seed(seed)

    result = graph if inplace else copy_graph(graph)
    nodes = list(result.keys())
    n = len(nodes)

    # Find all non-edges
    existing_edges = set()
    for u in result:
        for v in result[u]:
            existing_edges.add((min(u, v), max(u, v)))

    # Generate list of possible new edges
    possible_new = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in existing_edges:
                possible_new.append((i, j))

    # Randomly select edges to add
    num_to_add = min(num_edges, len(possible_new))
    edges_to_add = random.sample(possible_new, num_to_add)

    for u, v in edges_to_add:
        add_edge(result, u, v)

    return result


def remove_random_edges(
    graph: Dict[int, List[int]],
    num_edges: int,
    seed: int = None,
    inplace: bool = False
) -> Dict[int, List[int]]:
    """
    Remove random edges from a graph.

    Randomly selects existing edges and removes them.
    Useful for testing robustness to missing data.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Original adjacency list.

    num_edges : int
        Number of edges to remove.

    seed : int, optional
        Random seed for reproducibility.

    inplace : bool, optional (default=False)
        If True, modify graph in place. Otherwise, return a copy.

    Returns
    -------
    dict[int, list[int]]
        Graph with removed edges.

    Notes
    -----
    If num_edges exceeds the number of existing edges, removes all edges.
    """
    if seed is not None:
        random.seed(seed)

    result = graph if inplace else copy_graph(graph)

    # Get list of all edges
    edge_list = get_edge_list(result)

    # Randomly select edges to remove
    num_to_remove = min(num_edges, len(edge_list))
    edges_to_remove = random.sample(edge_list, num_to_remove)

    for u, v in edges_to_remove:
        remove_edge(result, u, v)

    return result


def apply_noise(
    graph: Dict[int, List[int]],
    add_fraction: float = 0.05,
    remove_fraction: float = 0.05,
    seed: int = None,
    inplace: bool = False
) -> Dict[int, List[int]]:
    """
    Apply noise to a graph by adding and removing edges.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Original adjacency list.

    add_fraction : float, optional (default=0.05)
        Fraction of possible new edges to add (0.0 to 1.0).

    remove_fraction : float, optional (default=0.05)
        Fraction of existing edges to remove (0.0 to 1.0).

    seed : int, optional
        Random seed for reproducibility.

    inplace : bool, optional (default=False)
        If True, modify graph in place.

    Returns
    -------
    dict[int, list[int]]
        Noisy graph.

    Examples
    --------
    >>> graph = {0: [1], 1: [0, 2], 2: [1]}
    >>> noisy = apply_noise(graph, add_fraction=0.1, remove_fraction=0.1, seed=42)
    """
    if seed is not None:
        random.seed(seed)

    result = graph if inplace else copy_graph(graph)
    n = len(result)

    # Count current edges and possible new edges
    current_edges = sum(len(neighbors) for neighbors in result.values()) // 2
    max_possible_edges = n * (n - 1) // 2
    possible_new_edges = max_possible_edges - current_edges

    # Calculate number of edges to add/remove
    num_to_add = int(possible_new_edges * add_fraction)
    num_to_remove = int(current_edges * remove_fraction)

    # Apply noise
    result = remove_random_edges(result, num_to_remove, inplace=True)
    result = add_random_edges(result, num_to_add, inplace=True)

    return result


def split_edges_for_testing(
    graph: Dict[int, List[int]],
    test_fraction: float = 0.2,
    seed: int = None
) -> Tuple[Dict[int, List[int]], List[Tuple[int, int]]]:
    """
    Split graph into training graph and test edges.

    Used for evaluating link prediction / recommendation algorithms.
    Remove some edges (test set) and see if the algorithm can predict them.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Original adjacency list.

    test_fraction : float, optional (default=0.2)
        Fraction of edges to hold out for testing.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (train_graph, test_edges)
        - train_graph: Graph with test edges removed
        - test_edges: List of (u, v) edges that were removed

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    >>> train, test = split_edges_for_testing(graph, test_fraction=0.5, seed=42)
    >>> # test contains some edges that algorithms should try to predict
    """
    if seed is not None:
        random.seed(seed)

    edge_list = get_edge_list(graph)
    num_test = int(len(edge_list) * test_fraction)

    # Randomly select test edges
    test_edges = random.sample(edge_list, num_test)
    test_set = set(test_edges)

    # Create training graph (remove test edges)
    train_graph = copy_graph(graph)
    for u, v in test_edges:
        remove_edge(train_graph, u, v)

    return train_graph, test_edges


if __name__ == "__main__":
    # Demo
    print("=== Noise Injection Demo ===\n")

    # Original graph
    graph = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1],
        3: [1]
    }

    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    print(f"Original graph: {edge_count} edges")
    print(f"  {graph}\n")

    # Add random edges
    added = add_random_edges(graph, num_edges=2, seed=42)
    new_edge_count = sum(len(neighbors) for neighbors in added.values()) // 2
    print(f"After adding 2 edges: {new_edge_count} edges")
    print(f"  {added}\n")

    # Remove random edges
    removed = remove_random_edges(graph, num_edges=2, seed=42)
    removed_edge_count = sum(len(neighbors) for neighbors in removed.values()) // 2
    print(f"After removing 2 edges: {removed_edge_count} edges")
    print(f"  {removed}\n")

    # Apply mixed noise
    noisy = apply_noise(graph, add_fraction=0.2, remove_fraction=0.2, seed=42)
    noisy_edge_count = sum(len(neighbors) for neighbors in noisy.values()) // 2
    print(f"After mixed noise: {noisy_edge_count} edges")
    print(f"  {noisy}\n")

    # Split for testing
    train, test = split_edges_for_testing(graph, test_fraction=0.5, seed=42)
    train_edges = sum(len(neighbors) for neighbors in train.values()) // 2
    print(f"Train/test split: {train_edges} train edges, {len(test)} test edges")
    print(f"  Train: {train}")
    print(f"  Test edges: {test}")
