"""
Louvain Algorithm for Community Detection.

The Louvain algorithm is a greedy optimization method for detecting
communities by maximizing modularity. It's known for being fast and
producing high-quality partitions.

The algorithm has two phases that repeat until convergence:
1. Local optimization: Move nodes to neighboring communities if it improves modularity
2. Community aggregation: Collapse communities into super-nodes and repeat

Time Complexity: O(n log n) average case for sparse graphs

Reference:
    Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
    Fast unfolding of communities in large networks.
    Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Set, Tuple, Optional
from algorithms.community.modularity import compute_modularity, get_communities_list
import random


class LouvainCommunityDetection:
    """
    Louvain algorithm implementation for community detection.
    
    This class provides a complete implementation of the Louvain method,
    including hierarchical community detection.
    
    Attributes
    ----------
    resolution : float
        Resolution parameter for modularity optimization.
    
    seed : int or None
        Random seed for reproducibility.
    
    max_iterations : int
        Maximum iterations for local optimization phase.
    
    min_modularity_gain : float
        Minimum modularity improvement to continue optimization.
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        seed: Optional[int] = None,
        max_iterations: int = 100,
        min_modularity_gain: float = 1e-7
    ):
        """
        Initialize the Louvain algorithm.
        
        Parameters
        ----------
        resolution : float, optional (default=1.0)
            Resolution parameter. Higher values produce more communities.
            
        seed : int, optional
            Random seed for reproducibility.
            
        max_iterations : int, optional (default=100)
            Maximum iterations for local optimization.
            
        min_modularity_gain : float, optional (default=1e-7)
            Minimum modularity improvement to continue.
        """
        self.resolution = resolution
        self.seed = seed
        self.max_iterations = max_iterations
        self.min_modularity_gain = min_modularity_gain
        
        if seed is not None:
            random.seed(seed)
    
    def _initialize_partition(self, graph: Dict[int, List[int]]) -> Dict[int, int]:
        """Initialize each node in its own community."""
        return {node: node for node in graph}
    
    def _compute_community_data(
        self,
        graph: Dict[int, List[int]],
        partition: Dict[int, int],
        degree: Dict[int, int]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Compute community statistics for efficient modularity gain calculation.
        
        Returns
        -------
        tuple
            (community_total_degree, community_internal_edges)
        """
        community_total_degree: Dict[int, int] = {}
        community_internal_edges: Dict[int, int] = {}
        
        for node, comm in partition.items():
            # Add node degree to community total
            community_total_degree[comm] = community_total_degree.get(comm, 0) + degree[node]
            
            # Count internal edges (edges to same community)
            internal = 0
            for neighbor in graph.get(node, []):
                if partition.get(neighbor) == comm:
                    internal += 1
            community_internal_edges[comm] = community_internal_edges.get(comm, 0) + internal
        
        return community_total_degree, community_internal_edges
    
    def _local_moving_phase(
        self,
        graph: Dict[int, List[int]],
        partition: Dict[int, int],
        degree: Dict[int, int],
        m: float
    ) -> Tuple[Dict[int, int], bool]:
        """
        Phase 1: Local node moving to maximize modularity.
        
        For each node, try moving it to neighboring communities and
        keep the move that maximizes modularity gain.
        
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
        tuple
            (updated_partition, improved)
        """
        # Compute initial community data
        community_total_degree, _ = self._compute_community_data(graph, partition, degree)
        
        improved = True
        iteration = 0
        
        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1
            
            # Randomize node order
            nodes = list(graph.keys())
            random.shuffle(nodes)
            
            for node in nodes:
                current_comm = partition[node]
                k_i = degree[node]
                
                # Find neighboring communities and edge counts
                neighbor_comms: Dict[int, int] = {}
                for neighbor in graph.get(node, []):
                    n_comm = partition[neighbor]
                    neighbor_comms[n_comm] = neighbor_comms.get(n_comm, 0) + 1
                
                # Include current community
                if current_comm not in neighbor_comms:
                    neighbor_comms[current_comm] = 0
                
                # Remove node from current community for calculation
                community_total_degree[current_comm] -= k_i
                
                # Find edges to current community (after removal)
                k_i_current = neighbor_comms.get(current_comm, 0)
                
                # Find best community
                best_comm = current_comm
                best_gain = 0.0
                
                for target_comm, k_i_target in neighbor_comms.items():
                    if target_comm == current_comm:
                        continue
                    
                    sigma_target = community_total_degree.get(target_comm, 0)
                    sigma_current = community_total_degree.get(current_comm, 0)
                    
                    # Modularity gain formula
                    # ΔQ = [k_{i,target} - k_{i,current}] / m 
                    #    - resolution * k_i * (sigma_target - sigma_current) / (2m²)
                    gain = (k_i_target - k_i_current) / m
                    gain -= self.resolution * k_i * (sigma_target - sigma_current) / (2 * m * m)
                    
                    if gain > best_gain + self.min_modularity_gain:
                        best_gain = gain
                        best_comm = target_comm
                
                # Move node to best community
                community_total_degree[best_comm] += k_i
                
                if best_comm != current_comm:
                    partition[node] = best_comm
                    improved = True
        
        return partition, iteration > 1
    
    def _aggregate_graph(
        self,
        graph: Dict[int, List[int]],
        partition: Dict[int, int]
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """
        Phase 2: Aggregate communities into super-nodes.
        
        Create a new graph where each community becomes a node,
        with weighted edges representing inter-community connections.
        
        Parameters
        ----------
        graph : dict
            Original graph.
        partition : dict
            Current partition.
            
        Returns
        -------
        tuple
            (aggregated_graph, node_to_original_mapping)
        """
        # Get unique communities
        communities = sorted(set(partition.values()))
        
        # Create community -> new node ID mapping
        comm_to_node = {comm: i for i, comm in enumerate(communities)}
        
        # Build new graph
        new_graph: Dict[int, List[int]] = {i: [] for i in range(len(communities))}
        
        # Count edges between communities
        edge_counts: Dict[Tuple[int, int], int] = {}
        
        for node, neighbors in graph.items():
            comm_node = partition[node]
            new_node = comm_to_node[comm_node]
            
            for neighbor in neighbors:
                comm_neighbor = partition[neighbor]
                new_neighbor = comm_to_node[comm_neighbor]
                
                # Count this edge
                edge = (min(new_node, new_neighbor), max(new_node, new_neighbor))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        # Add edges to new graph
        for (u, v), count in edge_counts.items():
            if u == v:
                # Self-loops: add count//2 entries (each edge was counted twice)
                new_graph[u].extend([u] * (count // 2))
            else:
                # Inter-community edges: add count//2 entries to each side
                weight = count // 2
                new_graph[u].extend([v] * weight)
                new_graph[v].extend([u] * weight)
        
        # Create mapping from new nodes to original nodes
        original_mapping: Dict[int, List[int]] = {i: [] for i in range(len(communities))}
        for node, comm in partition.items():
            new_node = comm_to_node[comm]
            original_mapping[new_node].append(node)
        
        return new_graph, comm_to_node
    
    def _renumber_communities(self, partition: Dict[int, int]) -> Dict[int, int]:
        """Renumber communities to be consecutive integers starting from 0."""
        unique_comms = sorted(set(partition.values()))
        mapping = {old: new for new, old in enumerate(unique_comms)}
        return {node: mapping[comm] for node, comm in partition.items()}
    
    def fit(
        self,
        graph: Dict[int, List[int]]
    ) -> Tuple[Dict[int, int], float]:
        """
        Run the Louvain algorithm on the graph.
        
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
        >>> louvain = LouvainCommunityDetection(seed=42)
        >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4, 5], 4: [3, 5], 5: [3, 4]}
        >>> partition, Q = louvain.fit(graph)
        >>> len(set(partition.values()))  # Number of communities
        2
        """
        if not graph:
            return {}, 0.0
        
        # Calculate total edges
        m = sum(len(neighbors) for neighbors in graph.values()) / 2
        if m == 0:
            return {node: 0 for node in graph}, 0.0
        
        # Initialize
        current_graph = {k: list(v) for k, v in graph.items()}
        original_nodes = list(graph.keys())
        
        # Track mapping from current nodes to original nodes
        node_to_original: Dict[int, List[int]] = {node: [node] for node in graph}
        
        # Track best partition (to avoid over-aggregation)
        best_partition: Optional[Dict[int, int]] = None
        best_modularity: float = -1.0
        
        while True:
            # Compute degrees
            degree = {node: len(neighbors) for node, neighbors in current_graph.items()}
            current_m = sum(degree.values()) / 2
            
            if current_m == 0:
                break
            
            # Initialize partition (each node in own community)
            partition = self._initialize_partition(current_graph)
            
            # Phase 1: Local moving
            partition, improved = self._local_moving_phase(
                current_graph, partition, degree, current_m
            )
            
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
            
            # Check if we made any progress
            num_communities = len(set(partition.values()))
            if num_communities >= len(current_graph):
                # No improvement, we're done
                break
            
            # Phase 2: Aggregate
            current_graph, comm_to_node = self._aggregate_graph(current_graph, partition)
            
            # Update original node mapping
            new_node_to_original: Dict[int, List[int]] = {}
            for node, comm in partition.items():
                new_node = comm_to_node[comm]
                if new_node not in new_node_to_original:
                    new_node_to_original[new_node] = []
                new_node_to_original[new_node].extend(node_to_original[node])
            node_to_original = new_node_to_original
        
        # Return best partition found
        if best_partition is not None:
            return best_partition, best_modularity
        
        # Fallback: Build final partition from current state
        final_partition: Dict[int, int] = {}
        for super_node, original_nodes_list in node_to_original.items():
            for orig_node in original_nodes_list:
                final_partition[orig_node] = super_node
        
        final_partition = self._renumber_communities(final_partition)
        modularity = compute_modularity(graph, final_partition, self.resolution)
        
        return final_partition, modularity
    
    def fit_hierarchical(
        self,
        graph: Dict[int, List[int]]
    ) -> List[Dict[int, int]]:
        """
        Run Louvain and return hierarchical community structure.
        
        Returns partitions at each level of the hierarchy.
        
        Parameters
        ----------
        graph : dict[int, list[int]]
            Adjacency list.
            
        Returns
        -------
        list[dict[int, int]]
            List of partitions at each hierarchical level.
            Level 0 is finest (most communities), higher levels are coarser.
        """
        if not graph:
            return []
        
        hierarchy = []
        current_graph = {k: list(v) for k, v in graph.items()}
        
        # Track mapping from current nodes to original nodes
        node_to_original: Dict[int, List[int]] = {node: [node] for node in graph}
        
        while True:
            degree = {node: len(neighbors) for node, neighbors in current_graph.items()}
            current_m = sum(degree.values()) / 2
            
            if current_m == 0:
                break
            
            partition = self._initialize_partition(current_graph)
            partition, _ = self._local_moving_phase(current_graph, partition, degree, current_m)
            
            # Build partition for original nodes
            original_partition: Dict[int, int] = {}
            for node, comm in partition.items():
                for orig_node in node_to_original[node]:
                    original_partition[orig_node] = comm
            
            hierarchy.append(self._renumber_communities(original_partition))
            
            # Check if we can aggregate further
            num_communities = len(set(partition.values()))
            if num_communities >= len(current_graph):
                break
            
            # Aggregate
            current_graph, comm_to_node = self._aggregate_graph(current_graph, partition)
            
            # Update mapping
            new_node_to_original: Dict[int, List[int]] = {}
            for node, comm in partition.items():
                new_node = comm_to_node[comm]
                if new_node not in new_node_to_original:
                    new_node_to_original[new_node] = []
                new_node_to_original[new_node].extend(node_to_original[node])
            node_to_original = new_node_to_original
        
        return hierarchy


def louvain_communities(
    graph: Dict[int, List[int]],
    resolution: float = 1.0,
    seed: Optional[int] = None,
    max_iterations: int = 100
) -> Tuple[Dict[int, int], float]:
    """
    Detect communities using the Louvain algorithm.

    This is the main entry point for Louvain community detection.

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
    O(n log n) average case for sparse graphs

    Examples
    --------
    >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4, 5], 4: [3, 5], 5: [3, 4]}
    >>> partition, Q = louvain_communities(graph)
    >>> Q > 0.3  # Good community structure detected
    True
    >>> len(set(partition.values()))  # Should detect 2 communities
    2

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
    louvain = LouvainCommunityDetection(
        resolution=resolution,
        seed=seed,
        max_iterations=max_iterations
    )
    return louvain.fit(graph)


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

    # Test hierarchical detection
    print("\n--- Hierarchical Community Detection ---")
    louvain_detector = LouvainCommunityDetection(seed=42)
    hierarchy = louvain_detector.fit_hierarchical(large_graph)
    
    print(f"Found {len(hierarchy)} levels in hierarchy")
    for level, part in enumerate(hierarchy):
        num_comm = len(set(part.values()))
        Q_level = compute_modularity(large_graph, part)
        print(f"  Level {level}: {num_comm} communities, Q={Q_level:.4f}")
