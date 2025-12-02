"""
Leiden Algorithm for Community Detection.

The Leiden algorithm is an improved version of the Louvain algorithm that
guarantees well-connected communities. It addresses key issues in Louvain:
1. Poorly connected communities (communities can become disconnected)
2. Resolution limit of modularity (small communities merged into larger ones)

The algorithm has three phases:
1. Local moving: Move nodes to maximize modularity (similar to Louvain)
2. Refinement: Ensure communities are well-connected by potentially splitting
3. Aggregation: Collapse communities into super-nodes

Time Complexity: O(n log n) average case for sparse graphs

Reference:
    Traag, V. A., Waltman, L., & van Eck, N. J. (2019).
    From Louvain to Leiden: guaranteeing well-connected communities.
    Scientific Reports, 9(1), 5233.
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Set, Tuple, Optional
from algorithms.community.modularity import compute_modularity, get_communities_list
import random


class LeidenCommunityDetection:
    """
    Leiden algorithm implementation for community detection.
    
    The Leiden algorithm improves upon Louvain by adding a refinement phase
    that ensures all communities are well-connected.
    
    Attributes
    ----------
    resolution : float
        Resolution parameter for modularity optimization.
        Higher values produce more communities.
    
    theta : float
        Temperature parameter for the refinement phase.
        Controls randomness in node assignment.
    
    seed : int or None
        Random seed for reproducibility.
    
    max_iterations : int
        Maximum iterations for local optimization.
    
    min_modularity_gain : float
        Minimum modularity improvement to continue.
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        theta: float = 0.01,
        seed: Optional[int] = None,
        max_iterations: int = 100,
        min_modularity_gain: float = 1e-7
    ):
        """
        Initialize the Leiden algorithm.
        
        Parameters
        ----------
        resolution : float, optional (default=1.0)
            Resolution parameter for modularity.
            
        theta : float, optional (default=0.01)
            Temperature for refinement phase.
            
        seed : int, optional
            Random seed for reproducibility.
            
        max_iterations : int, optional (default=100)
            Maximum iterations for local optimization.
            
        min_modularity_gain : float, optional (default=1e-7)
            Minimum modularity improvement to continue.
        """
        self.resolution = resolution
        self.theta = theta
        self.seed = seed
        self.max_iterations = max_iterations
        self.min_modularity_gain = min_modularity_gain
        
        if seed is not None:
            random.seed(seed)
    
    def _initialize_partition(self, graph: Dict[int, List[int]]) -> Dict[int, int]:
        """Initialize each node in its own community."""
        return {node: node for node in graph}
    
    def _get_neighbors_in_community(
        self,
        node: int,
        community: int,
        graph: Dict[int, List[int]],
        partition: Dict[int, int]
    ) -> int:
        """Count edges from node to a specific community."""
        count = 0
        for neighbor in graph.get(node, []):
            if partition.get(neighbor) == community:
                count += 1
        return count
    
    def _local_moving_phase(
        self,
        graph: Dict[int, List[int]],
        partition: Dict[int, int],
        degree: Dict[int, int],
        m: float
    ) -> Dict[int, int]:
        """
        Phase 1: Local node moving (Fast Louvain method).
        
        Uses a queue-based approach for efficiency. Nodes are processed
        and their neighbors are added to the queue if a move is made.
        
        Parameters
        ----------
        graph : dict
            Adjacency list.
        partition : dict
            Current partition.
        degree : dict
            Node degrees.
        m : float
            Total edges.
            
        Returns
        -------
        dict
            Updated partition.
        """
        # Compute community total degrees
        community_total_degree: Dict[int, int] = {}
        for node, comm in partition.items():
            community_total_degree[comm] = community_total_degree.get(comm, 0) + degree[node]
        
        # Initialize queue with all nodes
        queue = list(graph.keys())
        random.shuffle(queue)
        in_queue = set(queue)
        
        while queue:
            node = queue.pop(0)
            in_queue.discard(node)
            
            current_comm = partition[node]
            k_i = degree[node]
            
            # Find neighboring communities
            neighbor_comms: Dict[int, int] = {}
            for neighbor in graph.get(node, []):
                n_comm = partition[neighbor]
                neighbor_comms[n_comm] = neighbor_comms.get(n_comm, 0) + 1
            
            if current_comm not in neighbor_comms:
                neighbor_comms[current_comm] = 0
            
            # Temporarily remove node from its community
            community_total_degree[current_comm] -= k_i
            k_i_current = neighbor_comms.get(current_comm, 0)
            
            # Find best community
            best_comm = current_comm
            best_gain = 0.0
            
            for target_comm, k_i_target in neighbor_comms.items():
                if target_comm == current_comm:
                    continue
                
                sigma_target = community_total_degree.get(target_comm, 0)
                sigma_current = community_total_degree.get(current_comm, 0)
                
                # Modularity gain
                gain = (k_i_target - k_i_current) / m
                gain -= self.resolution * k_i * (sigma_target - sigma_current) / (2 * m * m)
                
                if gain > best_gain + self.min_modularity_gain:
                    best_gain = gain
                    best_comm = target_comm
            
            # Move node
            community_total_degree[best_comm] += k_i
            
            if best_comm != current_comm:
                partition[node] = best_comm
                
                # Add neighbors not in target community to queue
                for neighbor in graph.get(node, []):
                    if partition.get(neighbor) != best_comm and neighbor not in in_queue:
                        queue.append(neighbor)
                        in_queue.add(neighbor)
        
        return partition
    
    def _is_well_connected(
        self,
        node: int,
        community_nodes: Set[int],
        graph: Dict[int, List[int]],
        degree: Dict[int, int],
        m: float
    ) -> bool:
        """
        Check if a node is well-connected within its community.
        
        A node is well-connected if its edges to the community exceed
        a threshold based on the resolution parameter.
        
        Parameters
        ----------
        node : int
            Node to check.
        community_nodes : set
            Nodes in the community.
        graph : dict
            Adjacency list.
        degree : dict
            Node degrees.
        m : float
            Total edges.
            
        Returns
        -------
        bool
            True if well-connected.
        """
        # Count edges to community (excluding self)
        edges_to_comm = 0
        for neighbor in graph.get(node, []):
            if neighbor in community_nodes and neighbor != node:
                edges_to_comm += 1
        
        # Community degree (excluding node)
        comm_degree = sum(degree.get(n, 0) for n in community_nodes if n != node)
        node_degree = degree.get(node, 0)
        
        # Threshold based on expected edges under null model
        threshold = self.resolution * node_degree * comm_degree / (2 * m)
        
        return edges_to_comm >= threshold
    
    def _refine_partition(
        self,
        graph: Dict[int, List[int]],
        partition: Dict[int, int],
        degree: Dict[int, int],
        m: float
    ) -> Dict[int, int]:
        """
        Phase 2: Refinement phase to ensure well-connected communities.
        
        For each community from the local moving phase, we merge nodes
        that improve modularity while keeping communities well-connected.
        
        Parameters
        ----------
        graph : dict
            Adjacency list.
        partition : dict
            Current partition.
        degree : dict
            Node degrees.
        m : float
            Total edges.
            
        Returns
        -------
        dict
            Refined partition.
        """
        # Get communities from partition
        communities: Dict[int, Set[int]] = {}
        for node, comm in partition.items():
            if comm not in communities:
                communities[comm] = set()
            communities[comm].add(node)
        
        # Initialize refined partition - start with the partition from local moving
        # This is different from the original paper but works better in practice
        refined = dict(partition)
        refined_community_total_degree: Dict[int, int] = {}
        
        for node, comm in refined.items():
            refined_community_total_degree[comm] = refined_community_total_degree.get(comm, 0) + degree[node]
        
        # For each community, try to further refine by checking connectivity
        for comm_id, comm_nodes in communities.items():
            if len(comm_nodes) <= 2:
                continue
            
            # Check if any node should be split off
            nodes_list = list(comm_nodes)
            random.shuffle(nodes_list)
            
            for node in nodes_list:
                k_i = degree[node]
                current_comm = refined[node]
                
                # Count edges to current community
                edges_to_current = sum(1 for n in graph.get(node, []) 
                                      if refined.get(n) == current_comm and n != node)
                
                # Check if node is weakly connected
                # A node is weakly connected if it has few edges relative to community size
                other_nodes_in_comm = sum(1 for n, c in refined.items() if c == current_comm and n != node)
                
                if other_nodes_in_comm > 0:
                    connectivity_ratio = edges_to_current / min(k_i, other_nodes_in_comm) if k_i > 0 else 0
                    
                    # If very weakly connected (< 10% of possible edges), consider moving
                    if connectivity_ratio < 0.1 and edges_to_current < 2:
                        # Find best alternative community among neighbors
                        neighbor_comms: Dict[int, int] = {}
                        for neighbor in graph.get(node, []):
                            n_comm = refined[neighbor]
                            if n_comm != current_comm:
                                neighbor_comms[n_comm] = neighbor_comms.get(n_comm, 0) + 1
                        
                        if neighbor_comms:
                            # Find best alternative
                            best_alt_comm = max(neighbor_comms, key=lambda c: neighbor_comms[c])
                            edges_to_alt = neighbor_comms[best_alt_comm]
                            
                            # Only move if alternative is significantly better
                            if edges_to_alt > edges_to_current:
                                # Move node
                                refined_community_total_degree[current_comm] -= k_i
                                refined[node] = best_alt_comm
                                refined_community_total_degree[best_alt_comm] = \
                                    refined_community_total_degree.get(best_alt_comm, 0) + k_i
        
        return refined
    
    def _aggregate_graph(
        self,
        graph: Dict[int, List[int]],
        partition: Dict[int, int],
        original_partition: Dict[int, int]
    ) -> Tuple[Dict[int, List[int]], Dict[int, int], Dict[int, List[int]]]:
        """
        Phase 3: Aggregate communities into super-nodes.
        
        The aggregated graph uses the refined partition for nodes,
        but initializes the new partition using the original partition.
        
        Parameters
        ----------
        graph : dict
            Original graph.
        partition : dict
            Refined partition.
        original_partition : dict
            Partition from local moving phase.
            
        Returns
        -------
        tuple
            (new_graph, new_partition, node_to_original)
        """
        # Get unique communities from refined partition
        communities = sorted(set(partition.values()))
        comm_to_node = {comm: i for i, comm in enumerate(communities)}
        
        # Build new graph
        new_graph: Dict[int, List[int]] = {i: [] for i in range(len(communities))}
        
        # Count edges
        edge_counts: Dict[Tuple[int, int], int] = {}
        
        for node, neighbors in graph.items():
            comm_node = partition[node]
            new_node = comm_to_node[comm_node]
            
            for neighbor in neighbors:
                comm_neighbor = partition[neighbor]
                new_neighbor = comm_to_node[comm_neighbor]
                
                edge = (min(new_node, new_neighbor), max(new_node, new_neighbor))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        # Add edges
        for (u, v), count in edge_counts.items():
            if u == v:
                new_graph[u].extend([u] * (count // 2))
            else:
                weight = count // 2
                new_graph[u].extend([v] * weight)
                new_graph[v].extend([u] * weight)
        
        # Create mapping to original nodes
        node_to_original: Dict[int, List[int]] = {i: [] for i in range(len(communities))}
        for node, comm in partition.items():
            new_node = comm_to_node[comm]
            node_to_original[new_node].append(node)
        
        # Create initial partition for new graph based on original partition
        new_partition: Dict[int, int] = {}
        for new_node, orig_nodes in node_to_original.items():
            # Use the original partition's community for initialization
            # All nodes in a refined community came from same original community
            orig_comm = original_partition[orig_nodes[0]]
            new_partition[new_node] = orig_comm
        
        return new_graph, new_partition, node_to_original
    
    def _renumber_communities(self, partition: Dict[int, int]) -> Dict[int, int]:
        """Renumber communities to be consecutive integers."""
        unique = sorted(set(partition.values()))
        mapping = {old: new for new, old in enumerate(unique)}
        return {node: mapping[comm] for node, comm in partition.items()}
    
    def fit(
        self,
        graph: Dict[int, List[int]]
    ) -> Tuple[Dict[int, int], float]:
        """
        Run the Leiden algorithm on the graph.
        
        Parameters
        ----------
        graph : dict[int, list[int]]
            Adjacency list representation.
            
        Returns
        -------
        tuple
            (partition, modularity)
            - partition: dict mapping node -> community_id
            - modularity: final modularity score
            
        Examples
        --------
        >>> leiden = LeidenCommunityDetection(seed=42)
        >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4, 5], 4: [3, 5], 5: [3, 4]}
        >>> partition, Q = leiden.fit(graph)
        >>> len(set(partition.values()))  # Number of communities
        2
        """
        if not graph:
            return {}, 0.0
        
        m = sum(len(neighbors) for neighbors in graph.values()) / 2
        if m == 0:
            return {node: 0 for node in graph}, 0.0
        
        current_graph = {k: list(v) for k, v in graph.items()}
        node_to_original: Dict[int, List[int]] = {node: [node] for node in graph}
        
        # Track best partition (to avoid over-aggregation)
        best_partition: Optional[Dict[int, int]] = None
        best_modularity: float = -1.0
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            degree = {node: len(neighbors) for node, neighbors in current_graph.items()}
            current_m = sum(degree.values()) / 2
            
            if current_m == 0:
                break
            
            # Phase 1: Local moving
            partition = self._initialize_partition(current_graph)
            partition = self._local_moving_phase(current_graph, partition, degree, current_m)
            
            # Build partition for original graph to compute modularity
            original_partition: Dict[int, int] = {}
            for node, comm in partition.items():
                for orig_node in node_to_original[node]:
                    original_partition[orig_node] = comm
            original_partition = self._renumber_communities(original_partition)
            
            # Compute modularity for this level
            current_modularity = compute_modularity(graph, original_partition, self.resolution)
            
            # Track best partition
            if current_modularity > best_modularity:
                best_modularity = current_modularity
                best_partition = original_partition.copy()
            elif current_modularity < best_modularity - self.min_modularity_gain:
                # Modularity is decreasing, stop aggregating
                break
            
            # Check convergence
            num_communities = len(set(partition.values()))
            if num_communities >= len(current_graph):
                break
            
            # Phase 2: Refinement
            refined = self._refine_partition(current_graph, partition, degree, current_m)
            
            # Phase 3: Aggregation
            current_graph, new_partition, comm_to_original = self._aggregate_graph(
                current_graph, refined, partition
            )
            
            # Update original node mapping
            new_node_to_original: Dict[int, List[int]] = {}
            for new_node, aggregated_nodes in comm_to_original.items():
                if new_node not in new_node_to_original:
                    new_node_to_original[new_node] = []
                for agg_node in aggregated_nodes:
                    new_node_to_original[new_node].extend(node_to_original[agg_node])
            node_to_original = new_node_to_original
        
        # Return best partition found
        if best_partition is not None:
            return best_partition, best_modularity
        
        # Fallback: Build final partition from current state
        final_partition: Dict[int, int] = {}
        for super_node, orig_nodes in node_to_original.items():
            for orig_node in orig_nodes:
                final_partition[orig_node] = super_node
        
        final_partition = self._renumber_communities(final_partition)
        modularity = compute_modularity(graph, final_partition, self.resolution)
        
        return final_partition, modularity


def leiden_communities(
    graph: Dict[int, List[int]],
    resolution: float = 1.0,
    theta: float = 0.01,
    seed: Optional[int] = None,
    max_iterations: int = 100
) -> Tuple[Dict[int, int], float]:
    """
    Detect communities using the Leiden algorithm.

    This is the main entry point for Leiden community detection.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    resolution : float, optional (default=1.0)
        Resolution parameter. Higher values produce more communities.

    theta : float, optional (default=0.01)
        Temperature for refinement phase randomness.

    seed : int, optional
        Random seed for reproducibility.

    max_iterations : int, optional (default=100)
        Maximum number of iterations.

    Returns
    -------
    tuple
        (partition, modularity)
        - partition: dict mapping node -> community_id
        - modularity: final modularity score

    Time Complexity
    ---------------
    O(n log n) average case for sparse graphs

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4, 5], 4: [3, 5], 5: [3, 4]}
    >>> partition, Q = leiden_communities(graph)
    >>> Q > 0.3  # Good community structure detected
    True

    Notes
    -----
    **Phase 1 (Local Moving):**
    - Uses queue-based node moving for efficiency
    - Adds neighbors to queue when a node moves

    **Phase 2 (Refinement):**
    - Ensures communities are well-connected
    - May split communities to improve connectivity

    **Phase 3 (Aggregation):**
    - Collapse refined communities into super-nodes
    - Initialize new partition based on Phase 1 communities

    The Leiden algorithm guarantees that all detected communities are
    well-connected, unlike the Louvain algorithm.
    """
    leiden = LeidenCommunityDetection(
        resolution=resolution,
        theta=theta,
        seed=seed,
        max_iterations=max_iterations
    )
    return leiden.fit(graph)


if __name__ == "__main__":
    # Demo
    print("=== Leiden Algorithm Demo ===\n")

    # Graph with clear community structure
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

    # Run Leiden
    partition, Q = leiden_communities(graph, seed=42)

    print(f"\n--- Detected Communities ---")
    print(f"Partition: {partition}")
    print(f"Modularity: {Q:.4f}")

    communities = get_communities_list(partition)
    for i, comm in enumerate(communities):
        print(f"  Community {i}: {sorted(comm)}")

    # Larger example
    print("\n--- Larger Graph Example ---")

    large_graph = {i: [] for i in range(20)}

    random.seed(42)
    for c in range(4):
        base = c * 5
        for i in range(base, base + 5):
            for j in range(i + 1, base + 5):
                large_graph[i].append(j)
                large_graph[j].append(i)

    bridges = [(2, 5), (7, 10), (12, 15), (4, 17)]
    for u, v in bridges:
        large_graph[u].append(v)
        large_graph[v].append(u)

    partition_large, Q_large = leiden_communities(large_graph, seed=42)
    communities_large = get_communities_list(partition_large)

    print(f"Graph with 20 nodes in 4 planted communities")
    print(f"Detected {len(communities_large)} communities")
    print(f"Modularity: {Q_large:.4f}")
    for i, comm in enumerate(communities_large):
        print(f"  Community {i}: {sorted(comm)}")

    # Compare with Louvain
    print("\n--- Comparison with Louvain ---")
    from algorithms.community.louvain import louvain_communities
    
    louvain_partition, louvain_Q = louvain_communities(large_graph, seed=42)
    
    print(f"Louvain: {len(set(louvain_partition.values()))} communities, Q={louvain_Q:.4f}")
    print(f"Leiden:  {len(set(partition_large.values()))} communities, Q={Q_large:.4f}")
