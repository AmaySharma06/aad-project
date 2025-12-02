"""
Community Detection Plotting Module.

Provides visualization functions for community detection experiments:
- Size scaling plots (runtime vs nodes)
- Modularity comparison plots
- Quality metrics (NMI, ARI) plots
- Resolution parameter effect plots
- Algorithm comparison bar charts
"""

from plots.community.plot_experiments import (
    plot_size_scaling,
    plot_size_modularity,
    plot_density_scaling,
    plot_quality_nmi,
    plot_quality_ari,
    plot_resolution_effect,
    plot_algorithm_comparison_bars,
    plot_communities_vs_size,
    generate_all_plots
)

__all__ = [
    "plot_size_scaling",
    "plot_size_modularity",
    "plot_density_scaling",
    "plot_quality_nmi",
    "plot_quality_ari",
    "plot_resolution_effect",
    "plot_algorithm_comparison_bars",
    "plot_communities_vs_size",
    "generate_all_plots"
]
