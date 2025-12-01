"""
Jaccard Similarity and related link prediction metrics.

Jaccard Similarity measures the overlap between two sets, commonly used
for link prediction in social networks to predict potential friendships.

For two users u and v:
    Jaccard(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

where N(x) is the set of neighbors (friends) of x.

Interpretation:
- High Jaccard score = many common friends relative to total unique friends
- Jaccard = 1 means identical friend sets
- Jaccard = 0 means no common friends

Applications:
- Friend recommendation ("People you may know")
- Link prediction
- Similarity-based clustering
"""

from typing import Dict, List, Set, Tuple
import math


def jaccard_similarity(
    graph: Dict[int, List[int]], 
    u: int, 
    v: int
) -> float:
    """
    Compute Jaccard similarity between two nodes.

    Jaccard similarity is defined as:
        J(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    u : int
        First node.

    v : int
        Second node.

    Returns
    -------
    float
        Jaccard similarity score in [0, 1].
        Returns 0 if both nodes have no neighbors.

    Time Complexity
    ---------------
    O(deg(u) + deg(v))

    Examples
    --------
    >>> graph = {0: [1, 2, 3], 1: [0, 2], 2: [0, 1, 3], 3: [0, 2]}
    >>> jaccard_similarity(graph, 1, 3)  # Common friends: {0, 2}
    0.5

    Notes
    -----
    High Jaccard similarity between non-adjacent nodes suggests
    they might become friends (triadic closure principle).
    """
    neighbors_u = set(graph.get(u, []))
    neighbors_v = set(graph.get(v, []))

    intersection = neighbors_u & neighbors_v
    union = neighbors_u | neighbors_v

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def jaccard_similarity_all_pairs(
    graph: Dict[int, List[int]],
    exclude_existing: bool = True
) -> List[Tuple[int, int, float]]:
    """
    Compute Jaccard similarity for all non-adjacent node pairs.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    exclude_existing : bool, optional (default=True)
        If True, only compute for pairs that are NOT currently connected.
        These are the potential new links.

    Returns
    -------
    list[tuple]
        List of (u, v, similarity) tuples, sorted by similarity descending.

    Time Complexity
    ---------------
    O(n² * average_degree)

    Examples
    --------
    >>> graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    >>> pairs = jaccard_similarity_all_pairs(graph)
    >>> # Returns non-adjacent pairs like (0,2), (0,3), (1,3) with scores
    """
    nodes = list(graph.keys())
    n = len(nodes)

    # Precompute neighbor sets
    neighbor_sets = {node: set(graph[node]) for node in nodes}

    results = []

    for i in range(n):
        u = nodes[i]
        for j in range(i + 1, n):
            v = nodes[j]

            # Skip if already connected (and exclude_existing is True)
            if exclude_existing and v in neighbor_sets[u]:
                continue

            # Compute Jaccard
            intersection = neighbor_sets[u] & neighbor_sets[v]
            union = neighbor_sets[u] | neighbor_sets[v]

            if len(union) > 0:
                similarity = len(intersection) / len(union)
                if similarity > 0:  # Only include non-zero similarities
                    results.append((u, v, similarity))

    # Sort by similarity descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def common_neighbors(
    graph: Dict[int, List[int]], 
    u: int, 
    v: int
) -> List[int]:
    """
    Find common neighbors between two nodes.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    u : int
        First node.

    v : int
        Second node.

    Returns
    -------
    list[int]
        List of nodes that are neighbors of both u and v.

    Examples
    --------
    >>> graph = {0: [1, 2, 3], 1: [0, 2], 2: [0, 1], 3: [0]}
    >>> common_neighbors(graph, 1, 3)
    [0]
    """
    neighbors_u = set(graph.get(u, []))
    neighbors_v = set(graph.get(v, []))
    return list(neighbors_u & neighbors_v)


def common_neighbors_count(
    graph: Dict[int, List[int]], 
    u: int, 
    v: int
) -> int:
    """
    Count the number of common neighbors between two nodes.

    This is a simple link prediction score: more common neighbors
    suggests higher likelihood of forming a connection.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    u : int
        First node.

    v : int
        Second node.

    Returns
    -------
    int
        Number of common neighbors.
    """
    return len(common_neighbors(graph, u, v))


def adamic_adar_index(
    graph: Dict[int, List[int]], 
    u: int, 
    v: int
) -> float:
    """
    Compute Adamic-Adar index between two nodes.

    The Adamic-Adar index weights common neighbors by the inverse
    log of their degree. It gives more weight to common neighbors
    with few connections (they're more "meaningful").

    AA(u, v) = Σ 1 / log(|N(w)|)
               w ∈ N(u) ∩ N(v)

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    u : int
        First node.

    v : int
        Second node.

    Returns
    -------
    float
        Adamic-Adar score (higher = more likely to connect).

    Examples
    --------
    >>> graph = {0: [1, 2, 3], 1: [0, 2], 2: [0, 1, 3, 4], 3: [0, 2], 4: [2]}
    >>> adamic_adar_index(graph, 1, 3)  # Common neighbor 0 has degree 3, 2 has degree 4
    1.19...  # 1/log(3) + 1/log(4)

    Notes
    -----
    The intuition: if two people share a common friend who has only
    a few friends, that's a stronger signal than sharing a common
    friend who has thousands of connections.
    """
    common = common_neighbors(graph, u, v)

    score = 0.0
    for w in common:
        degree_w = len(graph[w])
        if degree_w > 1:  # Avoid log(1) = 0 division
            score += 1.0 / math.log(degree_w)

    return score


def preferential_attachment(
    graph: Dict[int, List[int]], 
    u: int, 
    v: int
) -> int:
    """
    Compute preferential attachment score between two nodes.

    PA(u, v) = |N(u)| * |N(v)|

    Based on the idea that nodes with many connections are more
    likely to form new connections (rich-get-richer).

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    u : int
        First node.

    v : int
        Second node.

    Returns
    -------
    int
        Preferential attachment score.
    """
    return len(graph.get(u, [])) * len(graph.get(v, []))


def resource_allocation_index(
    graph: Dict[int, List[int]], 
    u: int, 
    v: int
) -> float:
    """
    Compute Resource Allocation index between two nodes.

    RA(u, v) = Σ 1 / |N(w)|
               w ∈ N(u) ∩ N(v)

    Similar to Adamic-Adar but without the log.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    u : int
        First node.

    v : int
        Second node.

    Returns
    -------
    float
        Resource Allocation score.
    """
    common = common_neighbors(graph, u, v)

    score = 0.0
    for w in common:
        degree_w = len(graph[w])
        if degree_w > 0:
            score += 1.0 / degree_w

    return score


if __name__ == "__main__":
    # Demo
    print("=== Jaccard Similarity Demo ===\n")

    # Sample social network
    #     0 -- 1
    #    /|    |
    #   3 |    |
    #    \|    |
    #     2 -- +
    #     |
    #     4
    graph = {
        0: [1, 2, 3],
        1: [0, 2],
        2: [0, 1, 3, 4],
        3: [0, 2],
        4: [2]
    }

    print("Graph (adjacency list):")
    for node, neighbors in graph.items():
        print(f"  {node}: {neighbors}")

    # Jaccard between specific pairs
    print("\n--- Pairwise Jaccard Similarity ---")
    pairs = [(0, 4), (1, 3), (1, 4), (3, 4)]
    for u, v in pairs:
        jac = jaccard_similarity(graph, u, v)
        common = common_neighbors(graph, u, v)
        print(f"  J({u}, {v}) = {jac:.3f}  (common neighbors: {common})")

    # All non-adjacent pairs
    print("\n--- All Non-Adjacent Pairs (potential links) ---")
    all_pairs = jaccard_similarity_all_pairs(graph)
    for u, v, sim in all_pairs[:5]:
        print(f"  ({u}, {v}): Jaccard = {sim:.3f}")

    # Compare different metrics
    print("\n--- Comparing Link Prediction Metrics ---")
    test_pairs = [(1, 3), (1, 4), (3, 4)]
    print(f"{'Pair':<10} {'Jaccard':<10} {'Common':<10} {'Adamic-Adar':<12} {'Pref Attach':<12}")
    print("-" * 54)
    for u, v in test_pairs:
        jac = jaccard_similarity(graph, u, v)
        cn = common_neighbors_count(graph, u, v)
        aa = adamic_adar_index(graph, u, v)
        pa = preferential_attachment(graph, u, v)
        print(f"({u}, {v}){'':<5} {jac:<10.3f} {cn:<10} {aa:<12.3f} {pa:<12}")
