"""
Community Detection Experiments.

This module provides experiments comparing Louvain and Leiden algorithms:
- Size scaling experiments
- Quality evaluation with ground truth
- Density effects
- Resolution parameter effects
- Algorithm comparison across multiple trials
"""

from experiments.community.run_experiments import (
    run_community_size_experiment,
    run_community_quality_experiment,
    run_community_density_experiment,
    run_resolution_experiment,
    run_algorithm_comparison
)

__all__ = [
    "run_community_size_experiment",
    "run_community_quality_experiment",
    "run_community_density_experiment",
    "run_resolution_experiment",
    "run_algorithm_comparison"
]
