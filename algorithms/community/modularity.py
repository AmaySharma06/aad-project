"""
Modularity Functions for Community Detection.

Modularity is a measure of the quality of a network partition into communities.
A high modularity indicates dense connections within communities and sparse
connections between communities.

Modularity Q is defined as:
    Q = (1/2m) * Σ [A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)

where:
    - A_ij = adjacency matrix entry (1 if edge, 0 otherwise)
    - k_i = degree of node i
    - m = total number of edges
    - c_i = community of node i
    - δ(c_i, c_j) = 1 if c_i == c_j, else 0

Modularity ranges from -0.5 to 1:
    - Q > 0.3 typically indicates significant community structure
    - Q ≈ 0 means random partition
    - Q < 0 means worse than random
"""

from typing import Dict, List, Set
import math


def compute_modularity(
    graph: Dict[int, List[int]],
    partition: Dict[int, int]
) -> float:
    """
    Compute the modularity score of a graph partition.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    partition : dict[int, int]
        Mapping from node -> community_id.

    Returns
    -------
    float
        Modularity score Q in range [-0.5, 1].

    Time Complexity
    ---------------
    O(n + m) where n = nodes, m = edges

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4], 4: [3]}
    >>> partition = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
    >>> Q = compute_modularity(graph, partition)
    >>> Q > 0.3  # Good community structure
    True

    Notes
    -----
    The formula is:
        Q = (1/2m) * Σ_ij [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)

    This can be rewritten for efficiency as:
        Q = Σ_c [ e_c - a_c² ]

    where:
        e_c = fraction of edges within community c
        a_c = fraction of edge endpoints in community c
    """
    # Calculate total edges (each edge counted once)
    m = sum(len(neighbors) for neighbors in graph.values()) / 2

    if m == 0:
        return 0.0

    # Calculate degree for each node
    degree = {node: len(neighbors) for node, neighbors in graph.items()}

    # Group nodes by community
    communities: Dict[int, Set[int]] = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = set()
        communities[comm_id].add(node)

    Q = 0.0

    for comm_id, members in communities.items():
        # e_c: edges within community (as fraction of total)
        edges_within = 0
        for node in members:
            for neighbor in graph[node]:
                if neighbor in members:
                    edges_within += 1
        # Each edge counted twice in undirected graph
        edges_within /= 2
        e_c = edges_within / m

        # a_c: sum of degrees in community (as fraction of 2m)
        degree_sum = sum(degree[node] for node in members)
        a_c = degree_sum / (2 * m)

        Q += e_c - a_c * a_c

    return Q


def compute_modularity_delta(
    graph: Dict[int, List[int]],
    partition: Dict[int, int],
    node: int,
    new_community: int,
    m: float = None,
    degree: Dict[int, int] = None
) -> float:
    """
    Compute the change in modularity from moving a node to a new community.

    This is more efficient than recomputing full modularity.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list.

    partition : dict[int, int]
        Current partition.

    node : int
        Node to move.

    new_community : int
        Target community.

    m : float, optional
        Total edges (precomputed for efficiency).

    degree : dict[int, int], optional
        Node degrees (precomputed for efficiency).

    Returns
    -------
    float
        Change in modularity (positive = improvement).

    Notes
    -----
    The modularity change formula for moving node i from community C to D:
        ΔQ = [k_{i,D} - k_{i,C}] / m - degree[i] * [Σ_D - Σ_C + degree[i]] / (2m²)

    where:
        k_{i,C} = edges from i to nodes in C
        Σ_C = sum of degrees in C
    """
    old_community = partition[node]

    if old_community == new_community:
        return 0.0

    # Precompute if not provided
    if m is None:
        m = sum(len(neighbors) for neighbors in graph.values()) / 2

    if degree is None:
        degree = {n: len(neighbors) for n, neighbors in graph.items()}

    if m == 0:
        return 0.0

    # Count edges from node to old and new community
    edges_to_old = 0
    edges_to_new = 0

    for neighbor in graph[node]:
        if partition[neighbor] == old_community and neighbor != node:
            edges_to_old += 1
        elif partition[neighbor] == new_community:
            edges_to_new += 1

    # Sum of degrees in each community (excluding the moving node)
    sigma_old = sum(degree[n] for n, c in partition.items() if c == old_community and n != node)
    sigma_new = sum(degree[n] for n, c in partition.items() if c == new_community)

    k_i = degree[node]

    # Modularity change formula
    delta_Q = (edges_to_new - edges_to_old) / m
    delta_Q -= k_i * (sigma_new - sigma_old) / (2 * m * m)

    return delta_Q


def get_communities_list(partition: Dict[int, int]) -> List[Set[int]]:
    """
    Convert partition dict to list of community sets.

    Parameters
    ----------
    partition : dict[int, int]
        Node -> community_id mapping.

    Returns
    -------
    list[set[int]]
        List of sets, each containing nodes in one community.
    """
    communities: Dict[int, Set[int]] = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = set()
        communities[comm_id].add(node)
    return list(communities.values())


def normalized_mutual_information(
    partition1: Dict[int, int],
    partition2: Dict[int, int]
) -> float:
    """
    Compute Normalized Mutual Information (NMI) between two partitions.

    NMI measures how similar two partitions are, useful for comparing
    detected communities against ground truth.

    Parameters
    ----------
    partition1 : dict[int, int]
        First partition (e.g., ground truth).

    partition2 : dict[int, int]
        Second partition (e.g., detected communities).

    Returns
    -------
    float
        NMI score in [0, 1].
        1 = identical partitions
        0 = independent partitions

    Examples
    --------
    >>> truth = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> detected = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> nmi = normalized_mutual_information(truth, detected)
    >>> nmi
    1.0

    Notes
    -----
    NMI = 2 * MI(P1, P2) / (H(P1) + H(P2))

    where MI = mutual information, H = entropy.
    """
    # Get all nodes
    nodes = set(partition1.keys()) & set(partition2.keys())
    n = len(nodes)

    if n == 0:
        return 0.0

    # Get communities
    comms1: Dict[int, Set[int]] = {}
    comms2: Dict[int, Set[int]] = {}

    for node in nodes:
        c1 = partition1[node]
        c2 = partition2[node]

        if c1 not in comms1:
            comms1[c1] = set()
        comms1[c1].add(node)

        if c2 not in comms2:
            comms2[c2] = set()
        comms2[c2].add(node)

    # Compute entropy H(P1)
    H1 = 0.0
    for comm in comms1.values():
        p = len(comm) / n
        if p > 0:
            H1 -= p * math.log(p)

    # Compute entropy H(P2)
    H2 = 0.0
    for comm in comms2.values():
        p = len(comm) / n
        if p > 0:
            H2 -= p * math.log(p)

    # Compute mutual information MI(P1, P2)
    MI = 0.0
    for c1_id, c1_nodes in comms1.items():
        for c2_id, c2_nodes in comms2.items():
            overlap = len(c1_nodes & c2_nodes)
            if overlap > 0:
                p_joint = overlap / n
                p1 = len(c1_nodes) / n
                p2 = len(c2_nodes) / n
                MI += p_joint * math.log(p_joint / (p1 * p2))

    # Normalized MI
    if H1 + H2 == 0:
        return 1.0  # Both partitions have single community

    return 2 * MI / (H1 + H2)


if __name__ == "__main__":
    # Demo
    print("=== Modularity Demo ===\n")

    # Graph with clear community structure
    #   Community 0: 0-1-2 (triangle)
    #   Community 1: 3-4-5 (triangle)
    #   One edge connecting communities: 2-3
    graph = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3],
        3: [2, 4, 5],
        4: [3, 5],
        5: [3, 4]
    }

    print("Graph with two communities:")
    for node, neighbors in graph.items():
        print(f"  {node}: {neighbors}")

    # Good partition (matches structure)
    good_partition = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    Q_good = compute_modularity(graph, good_partition)
    print(f"\nGood partition {good_partition}")
    print(f"  Modularity Q = {Q_good:.4f}")

    # Random partition
    random_partition = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}
    Q_random = compute_modularity(graph, random_partition)
    print(f"\nRandom partition {random_partition}")
    print(f"  Modularity Q = {Q_random:.4f}")

    # All in one community
    single_partition = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    Q_single = compute_modularity(graph, single_partition)
    print(f"\nSingle community {single_partition}")
    print(f"  Modularity Q = {Q_single:.4f}")

    # NMI comparison
    print("\n--- NMI Demo ---")
    ground_truth = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    detected = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    nmi = normalized_mutual_information(ground_truth, detected)
    print(f"Ground truth: {ground_truth}")
    print(f"Detected:     {detected}")
    print(f"NMI = {nmi:.4f} (perfect match)")

    detected_off = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1}  # Node 2 wrong
    nmi_off = normalized_mutual_information(ground_truth, detected_off)
    print(f"\nDetected off: {detected_off}")
    print(f"NMI = {nmi_off:.4f} (one node wrong)")
