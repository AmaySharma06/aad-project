## Centrality

This module implements four classical centrality measures:

- Degree Centrality
- Harmonic Closeness Centrality
- Betweenness Centrality (Brandes’ Algorithm)
- PageRank

We also provide scripts to run experiments (size and density) and to generate visual heatmap plots on a custom showcase graph.  
All random experiments are seeded for reproducibility.

---

### How to Run Centrality Experiments and Plots

All commands must be run from the project root directory (aad-project/).

Required external libraries:
pip install matplotlib networkx

---

### 1. Size Experiment (Scaling with graph size)

This experiment varies the number of vertices n while keeping p = 0.1.

Run:
python -m experiments.centrality.run_size_experiment

This generates:
experiments/centrality/results/size_experiment.csv

Then generate the plot:
python -m plots.centrality.plotSizeVsTime

This produces:
plots/centrality/images/size_vs_time.png

---

### 2. Density Experiment (Scaling with edge probability p)

This experiment fixes n = 100 and varies density p across chosen values.

Run:
python -m experiments.centrality.run_density_experiment

This generates:
experiments/centrality/results/density_experiment.csv

Then generate the plot:
python -m plots.centrality.plotDensityVsTime

This produces:
plots/centrality/images/density_vs_time.png

---

### 3. Centrality Heatmaps (Showcase Graph)

This generates a structured graph designed to highlight differences between centrality algorithms.

Run:
python -m plots.centrality.plot_centrality_heatmaps

This produces four heatmaps:
plots/centrality/images/degree_heatmap.png
plots/centrality/images/harmonic_heatmap.png
plots/centrality/images/betweenness_heatmap.png
plots/centrality/images/pagerank_heatmap.png

And a 2×2 comparison grid:
plots/centrality/images/centrality_heatmap_grid.png
