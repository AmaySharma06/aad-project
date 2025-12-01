"""
Graph generation and utilities module.

This module provides:
- Random graph generation (Erdős-Rényi model)
- Social network generation with personality tags
- Graph utility functions (conversions, BFS, validation)
- Noise injection for robustness testing
"""

from graph.graph_generator import generate_random_graph
from graph.social_network import generate_social_network, SocialNetwork
from graph.graph_utils import (
    adjacency_matrix_to_list,
    adjacency_list_to_matrix,
    bfs_shortest_paths,
    get_connected_components,
    validate_graph
)
from graph.noise import add_random_edges, remove_random_edges, apply_noise

__all__ = [
    "generate_random_graph",
    "generate_social_network",
    "SocialNetwork",
    "adjacency_matrix_to_list",
    "adjacency_list_to_matrix",
    "bfs_shortest_paths",
    "get_connected_components",
    "validate_graph",
    "add_random_edges",
    "remove_random_edges",
    "apply_noise",
]
