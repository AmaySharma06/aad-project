"""
Louvain Algorithm for Community Detection.

The Louvain algorithm is a greedy optimization method for detecting
communities by maximizing modularity. It's known for being fast and
producing high-quality partitions.

The algorithm has two phases that repeat until convergence:
1. Local optimization: Move nodes to neighboring communities if it improves modularity
2. Community aggregation: Collapse communities into super-nodes and repeat

Reference:
    Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
    Fast unfolding of communities in large networks.
"""

from typing import Dict, List, Set, Tuple
from algorithms.community.modularity import compute_modularity, get_communities_list
import random


def louvain_communities(
    graph: Dict[int, List[int]],
    resolution: float = 1.0,
    seed: int = None,
    max_iterations: int = 100
) -> Tuple[Dict[int, int], float]:
    """
    Detect communities using the Louvain algorithm.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    resolution : float, optional (default=1.0)
        Resolution parameter. Higher values produce more communities.
        1.0 is standard modularity.

    seed : int, optional
        Random seed for reproducibility.

    max_iterations : int, optional (default=100)
        Maximum iterations for the local optimization phase.

    Returns
    -------
    tuple
        (partition, modularity)
        - partition: dict mapping node -> community_id
        - modularity: final modularity score

    Time Complexity
    ---------------
    O(n * log(n)) average case for sparse graphs

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4, 5], 4: [3, 5], 5: [3, 4]}
    >>> partition, Q = louvain_communities(graph)
    >>> Q > 0.3  # Good community structure detected
    True

    Notes
    -----
    **Phase 1 (Local Optimization):**
    - Start with each node in its own community
    - For each node, try moving it to each neighbor's community
    - Accept the move that gives maximum modularity gain
    - Repeat until no more improvements

    **Phase 2 (Community Aggregation):**
    - Collapse each community into a single super-node
    - Build a new graph where edges represent inter-community connections
    - Return to Phase 1 on the reduced graph

    The algorithm naturally discovers hierarchical community structure.
    """
    if seed is not None:
        random.seed(seed)

    # Handle empty graph
    if not graph:
        return {}, 0.0

    # Initialize: each node in its own community
    partition = {node: node for node in graph}

    # Precompute useful values
    m = sum(len(neighbors) for neighbors in graph.values()) / 2
    if m == 0:
        return partition, 0.0

    degree = {node: len(neighbors) for node, neighbors in graph.items()}

    # Main loop
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Randomize node order to avoid deterministic bias
        nodes = list(graph.keys())
        random.shuffle(nodes)

        for node in nodes:
            current_community = partition[node]

            # Find neighboring communities
            neighbor_communities = set()
            for neighbor in graph[node]:
                neighbor_communities.add(partition[neighbor])

            # Also consider staying in current community
            neighbor_communities.add(current_community)

            # Calculate modularity gain for each possible move
            best_community = current_community
            best_gain = 0.0

            for target_community in neighbor_communities:
                if target_community == current_community:
                    continue

                gain = _modularity_gain(
                    graph, partition, node, target_community,
                    m, degree, resolution
                )

                if gain > best_gain:
                    best_gain = gain
                    best_community = target_community

            # Make the move if it improves modularity
            if best_community != current_community:
                partition[node] = best_community
                improved = True

    # Renumber communities to be consecutive
    partition = _renumber_communities(partition)

    # Compute final modularity
    modularity = compute_modularity(graph, partition)

    return partition, modularity


def _modularity_gain(
    graph: Dict[int, List[int]],
    partition: Dict[int, int],
    node: int,
    target_community: int,
    m: float,
    degree: Dict[int, int],
    resolution: float = 1.0
) -> float:
    """
    Calculate the modularity gain from moving a node to a target community.

    Uses the efficient formula:
        ΔQ = [k_{i,in} / m - resolution * (Σ_tot * k_i) / (2m²)]

    where:
        k_{i,in} = edges from node i to nodes in target community
        Σ_tot = sum of degrees of nodes in target community
        k_i = degree of node i
    """
    current_community = partition[node]

    # Edges from node to current community (excluding self)
    k_i_current = sum(1 for neighbor in graph[node] 
                      if partition[neighbor] == current_community and neighbor != node)

    # Edges from node to target community
    k_i_target = sum(1 for neighbor in graph[node] 
                     if partition[neighbor] == target_community)

    # Sum of degrees in current community (excluding node)
    sigma_current = sum(degree[n] for n, c in partition.items() 
                       if c == current_community and n != node)

    # Sum of degrees in target community
    sigma_target = sum(degree[n] for n, c in partition.items() 
                      if c == target_community)

    k_i = degree[node]

    # Gain from removing from current community
    remove_gain = -k_i_current / m + resolution * (sigma_current * k_i) / (2 * m * m)

    # Gain from adding to target community
    add_gain = k_i_target / m - resolution * (sigma_target * k_i) / (2 * m * m)

    return remove_gain + add_gain


def _renumber_communities(partition: Dict[int, int]) -> Dict[int, int]:
    """
    Renumber communities to be consecutive integers starting from 0.
    """
    unique_communities = sorted(set(partition.values()))
    mapping = {old: new for new, old in enumerate(unique_communities)}
    return {node: mapping[comm] for node, comm in partition.items()}


def louvain_hierarchical(
    graph: Dict[int, List[int]],
    resolution: float = 1.0,
    seed: int = None
) -> List[Dict[int, int]]:
    """
    Run Louvain algorithm and return hierarchical community structure.

    This runs the full Louvain algorithm with community aggregation,
    returning the partition at each level of the hierarchy.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list.

    resolution : float, optional (default=1.0)
        Resolution parameter.

    seed : int, optional
        Random seed.

    Returns
    -------
    list[dict[int, int]]
        List of partitions at each hierarchical level.
        Level 0 is the finest (most communities), higher levels are coarser.
    """
    if seed is not None:
        random.seed(seed)

    hierarchy = []
    current_graph = graph.copy()
    node_mapping = {node: node for node in graph}  # Original node -> super-node

    while True:
        # Run local optimization phase
        partition, Q = louvain_communities(current_graph, resolution, seed)

        # Map back to original nodes
        original_partition = {}
        for original_node, super_node in node_mapping.items():
            original_partition[original_node] = partition[super_node]

        hierarchy.append(original_partition)

        # Check if we can aggregate further
        communities = get_communities_list(partition)
        if len(communities) >= len(current_graph):
            # No aggregation happened, we're done
            break

        # Phase 2: Aggregate communities into super-nodes
        current_graph, node_mapping = _aggregate_communities(
            current_graph, partition, node_mapping
        )

    return hierarchy


def _aggregate_communities(
    graph: Dict[int, List[int]],
    partition: Dict[int, int],
    prev_mapping: Dict[int, int]
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Aggregate communities into super-nodes for the next Louvain iteration.
    """
    # Build new graph where each community is a node
    community_ids = set(partition.values())
    new_graph = {c: [] for c in community_ids}

    # Count edges between communities
    edge_count: Dict[Tuple[int, int], int] = {}

    for node, neighbors in graph.items():
        comm_node = partition[node]
        for neighbor in neighbors:
            comm_neighbor = partition[neighbor]
            if comm_node != comm_neighbor:
                edge = (min(comm_node, comm_neighbor), max(comm_node, comm_neighbor))
                edge_count[edge] = edge_count.get(edge, 0) + 1

    # Add edges to new graph (we lose multiplicity in simple adjacency list)
    for (c1, c2), count in edge_count.items():
        if c2 not in new_graph[c1]:
            new_graph[c1].append(c2)
        if c1 not in new_graph[c2]:
            new_graph[c2].append(c1)

    # Update node mapping: original node -> new super-node
    new_mapping = {}
    for original_node, super_node in prev_mapping.items():
        new_mapping[original_node] = partition[super_node]

    return new_graph, new_mapping


if __name__ == "__main__":
    # Demo
    print("=== Louvain Algorithm Demo ===\n")

    # Graph with clear community structure
    # Community A: 0, 1, 2 (densely connected)
    # Community B: 3, 4, 5 (densely connected)
    # Sparse connection between communities
    graph = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3],  # Bridge node
        3: [2, 4, 5],
        4: [3, 5],
        5: [3, 4]
    }

    print("Graph structure:")
    for node, neighbors in graph.items():
        print(f"  {node} -> {neighbors}")

    # Run Louvain
    partition, Q = louvain_communities(graph, seed=42)

    print(f"\n--- Detected Communities ---")
    print(f"Partition: {partition}")
    print(f"Modularity: {Q:.4f}")

    # Show communities
    communities = get_communities_list(partition)
    for i, comm in enumerate(communities):
        print(f"  Community {i}: {sorted(comm)}")

    # Larger example with more structure
    print("\n--- Larger Graph Example ---")

    # Create a graph with 4 clear communities
    large_graph = {i: [] for i in range(20)}

    # Community 0: nodes 0-4
    # Community 1: nodes 5-9
    # Community 2: nodes 10-14
    # Community 3: nodes 15-19

    random.seed(42)
    for c in range(4):
        # Dense intra-community edges
        base = c * 5
        for i in range(base, base + 5):
            for j in range(i + 1, base + 5):
                large_graph[i].append(j)
                large_graph[j].append(i)

    # Sparse inter-community edges
    bridges = [(2, 5), (7, 10), (12, 15), (4, 17)]
    for u, v in bridges:
        large_graph[u].append(v)
        large_graph[v].append(u)

    partition_large, Q_large = louvain_communities(large_graph, seed=42)
    communities_large = get_communities_list(partition_large)

    print(f"Graph with 20 nodes in 4 planted communities")
    print(f"Detected {len(communities_large)} communities")
    print(f"Modularity: {Q_large:.4f}")
    for i, comm in enumerate(communities_large):
        print(f"  Community {i}: {sorted(comm)}")
