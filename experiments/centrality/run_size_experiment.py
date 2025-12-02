import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from centrality.run_centrality_experiment import run_centrality_experiment
from graph.graph_generator import generate_random_graph
from utils import write_csv, print_progress

# EXPERIMENT: Effect of Graph size (n) on Centrality Runtimes

def run_size_experiment(
    sizes=[50, 100, 150, 200, 300, 400, 500, 600, 700, 800],
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
    rows = []
    header = [
        "n", 
        "p",
        "degree_time",
        "harmonic_closeness_time",
        "betweenness_time",
        "pagerank_time"
    ]

    print("=== Centrality Size Experiment ===")
    print(f"Testing sizes: {sizes}")
    print(f"Edge probability: {p}")
    print()

    for i, n in enumerate(sizes):
        print_progress(i, len(sizes), prefix=f"Size {n:>5}")

        # Generate graph (returns adjacency list directly)
        graph = generate_random_graph(n, p, seed=seed + i)

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

    print_progress(len(sizes), len(sizes), prefix="Complete")
    print()

    # Save the CSV
    write_csv(out_csv, header, rows)
    print(f"Size experiment results saved to {out_csv}")


if __name__ == "__main__":
    run_size_experiment()
