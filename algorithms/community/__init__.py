"""
Community Detection Algorithms.

This module provides implementations of community detection algorithms:
- Louvain: Fast greedy modularity optimization
- Leiden: Improved Louvain with guaranteed well-connected communities

Also includes modularity metrics and evaluation functions.
"""

from algorithms.community.modularity import (
    compute_modularity,
    compute_modularity_gain,
    get_communities_list,
    normalized_mutual_information,
    adjusted_rand_index
)
from algorithms.community.louvain import (
    louvain_communities,
    LouvainCommunityDetection
)
from algorithms.community.leiden import (
    leiden_communities,
    LeidenCommunityDetection
)

__all__ = [
    # Modularity functions
    "compute_modularity",
    "compute_modularity_gain",
    "get_communities_list",
    "normalized_mutual_information",
    "adjusted_rand_index",
    # Louvain algorithm
    "louvain_communities",
    "LouvainCommunityDetection",
    # Leiden algorithm
    "leiden_communities",
    "LeidenCommunityDetection",
]
