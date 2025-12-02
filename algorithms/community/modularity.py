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

Reference:
    Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community
    structure in networks. Physical Review E, 69(2), 026113.
"""

from typing import Dict, List, Set, Tuple
import math


def compute_modularity(
    graph: Dict[int, List[int]],
    partition: Dict[int, int],
    resolution: float = 1.0
) -> float:
    """
    Compute the modularity score of a graph partition.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    partition : dict[int, int]
        Mapping from node -> community_id.

    resolution : float, optional (default=1.0)
        Resolution parameter. Higher values favor smaller communities.

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
        Q = (1/2m) * Σ_ij [A_ij - γ * k_i*k_j/(2m)] * δ(c_i, c_j)

    This can be rewritten for efficiency as:
        Q = Σ_c [ e_c - γ * a_c² ]

    where:
        e_c = fraction of edges within community c
        a_c = fraction of edge endpoints in community c
        γ = resolution parameter
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
            for neighbor in graph.get(node, []):
                if neighbor in members:
                    edges_within += 1
        # Each edge counted twice in undirected graph
        edges_within /= 2
        e_c = edges_within / m

        # a_c: sum of degrees in community (as fraction of 2m)
        degree_sum = sum(degree.get(node, 0) for node in members)
        a_c = degree_sum / (2 * m)

        Q += e_c - resolution * a_c * a_c

    return Q


def compute_modularity_gain(
    node: int,
    target_community: int,
    graph: Dict[int, List[int]],
    partition: Dict[int, int],
    degree: Dict[int, int],
    community_internal_edges: Dict[int, int],
    community_total_degree: Dict[int, int],
    m: float,
    resolution: float = 1.0
) -> float:
    """
    Compute the modularity gain from moving a node to a target community.

    Uses the efficient formula that avoids recomputing full modularity.

    Parameters
    ----------
    node : int
        The node to move.

    target_community : int
        The community to move the node to.

    graph : dict[int, list[int]]
        Adjacency list.

    partition : dict[int, int]
        Current partition.

    degree : dict[int, int]
        Node degrees.

    community_internal_edges : dict[int, int]
        Sum of internal edges * 2 for each community.

    community_total_degree : dict[int, int]
        Sum of degrees for each community.

    m : float
        Total number of edges.

    resolution : float, optional (default=1.0)
        Resolution parameter.

    Returns
    -------
    float
        The change in modularity if node is moved to target_community.

    Notes
    -----
    The modularity gain formula for moving node i to community C is:
        ΔQ = [k_{i,C} / m - resolution * Σ_C * k_i / (2m²)]

    where:
        k_{i,C} = number of edges from i to nodes in C
        Σ_C = sum of degrees in C
        k_i = degree of node i
    """
    if m == 0:
        return 0.0

    current_community = partition[node]
    
    if current_community == target_community:
        return 0.0

    k_i = degree[node]
    
    # Count edges from node to target community
    k_i_target = 0
    for neighbor in graph.get(node, []):
        if partition.get(neighbor) == target_community:
            k_i_target += 1

    # Count edges from node to current community (excluding self)
    k_i_current = 0
    for neighbor in graph.get(node, []):
        if partition.get(neighbor) == current_community and neighbor != node:
            k_i_current += 1

    # Get community totals
    sigma_target = community_total_degree.get(target_community, 0)
    sigma_current = community_total_degree.get(current_community, 0) - k_i

    # Compute gain
    # Moving out of current community: lose k_i_current, gain back resolution * sigma_current * k_i / (2m²)
    # Moving into target community: gain k_i_target, lose resolution * sigma_target * k_i / (2m²)
    
    gain = (k_i_target - k_i_current) / m
    gain -= resolution * k_i * (sigma_target - sigma_current) / (2 * m * m)
    
    return gain


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


def adjusted_rand_index(
    partition1: Dict[int, int],
    partition2: Dict[int, int]
) -> float:
    """
    Compute Adjusted Rand Index (ARI) between two partitions.

    ARI is a measure of the similarity between two clusterings, adjusted
    for chance. It ranges from -1 to 1, with 1 meaning perfect agreement.

    Parameters
    ----------
    partition1 : dict[int, int]
        First partition.

    partition2 : dict[int, int]
        Second partition.

    Returns
    -------
    float
        ARI score in [-1, 1].
        1 = identical partitions
        0 = random agreement
        < 0 = less than random agreement
    """
    nodes = set(partition1.keys()) & set(partition2.keys())
    n = len(nodes)
    
    if n < 2:
        return 1.0 if n == 1 else 0.0

    # Build contingency table
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

    # Compute the contingency table n_ij
    contingency = {}
    for c1_id, c1_nodes in comms1.items():
        for c2_id, c2_nodes in comms2.items():
            n_ij = len(c1_nodes & c2_nodes)
            if n_ij > 0:
                contingency[(c1_id, c2_id)] = n_ij

    # Compute a_i (row sums) and b_j (column sums)
    a = {c1: len(nodes) for c1, nodes in comms1.items()}
    b = {c2: len(nodes) for c2, nodes in comms2.items()}

    # Compute the index
    def comb2(x):
        return x * (x - 1) // 2

    sum_nij_comb = sum(comb2(n_ij) for n_ij in contingency.values())
    sum_a_comb = sum(comb2(a_i) for a_i in a.values())
    sum_b_comb = sum(comb2(b_j) for b_j in b.values())
    n_comb = comb2(n)

    if n_comb == 0:
        return 1.0

    expected = sum_a_comb * sum_b_comb / n_comb
    max_index = (sum_a_comb + sum_b_comb) / 2
    
    if max_index == expected:
        return 1.0

    return (sum_nij_comb - expected) / (max_index - expected)


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

    # ARI comparison
    print("\n--- ARI Demo ---")
    ari = adjusted_rand_index(ground_truth, detected)
    print(f"ARI (perfect match) = {ari:.4f}")
    
    ari_off = adjusted_rand_index(ground_truth, detected_off)
    print(f"ARI (one node wrong) = {ari_off:.4f}")
