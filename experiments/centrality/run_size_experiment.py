import os
import random

from centrality.run_centrality_experiment import run_centrality_experiment
from common_utils import get_random_graph, mat_to_list, write_csv

# EXPERIMENT: Effect of Graph size (n) on Centrality Runtimes

def run_size_experiment(
    sizes=[50, 100, 150, 200, 300, 500, 1000],
    p=0.1,
    seed=42,
    out_csv="experiments/centrality/results/size_results.csv"
):
    """
    Runs centrality algorithms on graphs of different sizes (fixed density p).
    Stores results in a CSV file.

    CSV Columns:
        n, p, degree_time, closeness_time, betweenness_time, pagerank_time
    """
    # Ensure reproducibility
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

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    for n in sizes:
        # Generate graph
        matrix = get_random_graph(n, p)
        graph = mat_to_list(matrix)

        # Run experiment
        result = run_centrality_experiment(graph)

        rows.append([
            n, 
            p,
            result["degree_time"],
            result["harmonic_closeness_time"],
            result["betweenness_time"],
            result["pagerank_time"],
        ])

    # Save the CSV
    write_csv(out_csv, header, rows)
    print(f"Size experiment results saved to {out_csv}")

if __name__ == "__main__":
    run_size_experiment()
