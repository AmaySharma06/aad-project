# Social Network Analysis - AAD Project

Graph theory algorithms for synthetic friendship networks.

## Repository Link
https://github.com/AmaySharma06/aad-project

## Structure

```
├── algorithms/          # Core implementations
│   ├── traversal/      # BFS, DFS, Union-Find
│   ├── centrality/     # Degree, Harmonic, Betweenness, PageRank
│   ├── community/      # Louvain, Leiden (BONUS)
│   └── recommender/    # Jaccard, Adamic-Adar, hybrid system
├── graph/              # Graph generation and utilities
├── experiments/        # Experiment runners
├── plots/              # Plotting scripts
├── report/             # LaTeX report
└── presentation/       # LaTeX slides
```

## Setup

```bash
# Clone repository
git clone https://github.com/AmaySharma06/aad-project.git
cd aad-project

# Install Python dependencies
pip install matplotlib numpy pandas scikit-learn networkx

# Verify installation
python -c "import matplotlib, numpy, pandas, sklearn, networkx; print('Setup complete!')"
```

## Quick Start

```bash
# Run traversal experiments
python experiments/traversal/run_experiments.py

# Run centrality experiments (all 4 algorithms: degree, harmonic closeness, betweenness, pagerank)
python experiments/centrality/run_size_experiment.py
python experiments/centrality/run_density_experiment.py

# Run recommender experiments
python experiments/recommender/run_experiments.py

# Run community detection (BONUS)
python experiments/community/run_experiments.py --experiment size
python experiments/community/run_experiments.py --experiment density
python experiments/community/run_experiments.py --experiment quality
python experiments/community/run_experiments.py --experiment resolution
python experiments/community/run_experiments.py --experiment comparison
```

## Generate Plots

```bash
# Generate plots for each module
python plots/traversal/plot_experiments.py
python plots/centrality/plot_size_vs_time.py         # Size vs time for all 4 algorithms
python plots/centrality/plot_density_vs_time.py     # Density vs time for all 4 algorithms
python plots/centrality/plot_centrality_heatmaps.py  # Centrality heatmap visualizations
python plots/recommender/plot_experiments.py
python plots/community/plot_experiments.py
```

## Build Report

```bash
cd report
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

## Build Presentation

```bash
cd presentation
pdflatex main.tex
```

## Requirements

- Python 3.12+
- Matplotlib, NumPy, Pandas, NetworkX
- LaTeX distribution (MiKTeX/TeX Live)

## Team

**DROP TABLE Teams;**
- Sarthak Mishra (2024117007)
- Amay Sharma (2024101095)
- Yashav Bhatnagar (2024101030)
- Lasya Katari (2024115004)
- Kartik Thapa (2024115009)

IIIT Hyderabad • CS1.301 • December 2025
