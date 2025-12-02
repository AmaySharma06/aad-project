import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from centrality.run_centrality_experiment import run_centrality_experiment
from graph.graph_generator import generate_random_graph
from utils import write_csv, print_progress

# EXPERIMENT: Effect of Graph density (p) on Centrality Runtimes

def run_density_experiment(
    n=100,
    densities=[0.02, 0.04, 0.06, 0.08, 0.1, 0.3, 0.5, 0.7],
    seed=42,
    out_csv="experiments/centrality/results/density_results.csv"
):
    """
    Runs centrality algorithms on graphs of fixed size (n) but varying density p.
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

    print("=== Centrality Density Experiment ===")
    print(f"Graph size: {n} nodes")
    print(f"Testing densities: {densities}")
    print()

    for i, p in enumerate(densities):
        print_progress(i, len(densities), prefix=f"Density {p:.2f}")

        # Generate graph (returns adjacency list directly)
        graph = generate_random_graph(n, p, seed=seed + i)

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

    print_progress(len(densities), len(densities), prefix="Complete")
    print()

    # Save CSV
    write_csv(out_csv, header, rows)
    print(f"Density experiment results saved to {out_csv}")


if __name__ == "__main__":
    run_density_experiment()
