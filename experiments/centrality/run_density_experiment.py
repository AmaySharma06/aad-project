import os
import random

from centrality.run_centrality_experiment import run_centrality_experiment
from common_utils import get_random_graph, mat_to_list, write_csv

# EXPERIMENT: Effect of Graph density (p) on Centrality Runtimes

def run_density_experiment(
    n=100,
    densities=[0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
    seed=42,
    out_csv="experiments/centrality/results/density_results.csv"
):
    """
    Runs centrality algorithms on graphs of fixed size (n) but varying density p.
    Stores results in a CSV file.

    CSV Columns:
        n, p, degree_time, closeness_time, betweenness_time, pagerank_time
    """
    random.seed(seed)

    rows = []
    header = [
        "n", 
        "p",
        "degree_time",
        "harmonic_closeness_time",
        "betweenness_time",
        "pagerank_time"
    ]

    # Ensure folder exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    for p in densities:
        # Generate graph
        matrix = get_random_graph(n, p)
        graph = mat_to_list(matrix)

        # Run centrality experiment
        result = run_centrality_experiment(graph)

        rows.append([
            n, 
            p,
            result["degree_time"],
            result["harmonic_closeness_time"],
            result["betweenness_time"],
            result["pagerank_time"],
        ])

    # Save CSV
    write_csv(out_csv, header, rows)
    print(f"Density experiment results saved to {out_csv}")


if __name__ == "__main__":
    run_density_experiment()
