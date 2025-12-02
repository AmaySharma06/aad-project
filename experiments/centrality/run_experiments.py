"""
Experiment: PageRank Runtime and Convergence.

Tests PageRank performance across different graph sizes and densities.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph.graph_generator import generate_random_graph
from algorithms.centrality.pagerank import pagerank
from utils import write_csv, time_function, print_progress


def run_pagerank_size_experiment(
    sizes: list = None,
    p: float = 0.1,
    seed: int = 42,
    out_csv: str = "experiments/centrality/results/pagerank_size.csv"
) -> None:
    """
    Benchmark PageRank on graphs of varying sizes.

    Parameters
    ----------
    sizes : list[int], optional
        Graph sizes to test.

    p : float, optional
        Edge probability.

    seed : int, optional
        Random seed.

    out_csv : str, optional
        Output CSV path.
    """
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000]

    header = ["n", "p", "edges", "pagerank_time"]
    rows = []

    print("=== PageRank Size Experiment ===")
    print(f"Testing sizes: {sizes}")
    print(f"Edge probability: {p}")
    print()

    for i, n in enumerate(sizes):
        print_progress(i, len(sizes), prefix=f"Size {n:>5}")

        graph = generate_random_graph(n, p, seed=seed + i)
        edges = sum(len(neighbors) for neighbors in graph.values()) // 2

        _, pr_time = time_function(pagerank, graph)

        rows.append([n, p, edges, pr_time])

    print_progress(len(sizes), len(sizes), prefix="Complete")
    print()

    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    # Summary
    print("\n--- Summary ---")
    print(f"{'n':>6} {'edges':>8} {'PageRank time':>15}")
    print("-" * 35)
    for row in rows:
        print(f"{row[0]:>6} {row[2]:>8} {row[3]:>15.6f}")


def run_pagerank_density_experiment(
    n: int = 500,
    densities: list = None,
    seed: int = 42,
    out_csv: str = "experiments/centrality/results/pagerank_density.csv"
) -> None:
    """
    Benchmark PageRank on graphs of varying densities.
    """
    if densities is None:
        densities = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    header = ["n", "p", "edges", "pagerank_time"]
    rows = []

    print("=== PageRank Density Experiment ===")
    print(f"Graph size: {n} nodes")
    print(f"Testing densities: {densities}")
    print()

    for i, p in enumerate(densities):
        print_progress(i, len(densities), prefix=f"Density {p:.2f}")

        graph = generate_random_graph(n, p, seed=seed + i)
        edges = sum(len(neighbors) for neighbors in graph.values()) // 2

        _, pr_time = time_function(pagerank, graph)

        rows.append([n, p, edges, pr_time])

    print_progress(len(densities), len(densities), prefix="Complete")
    print()

    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    # Summary
    print("\n--- Summary ---")
    print(f"{'p':>6} {'edges':>8} {'PageRank time':>15}")
    print("-" * 35)
    for row in rows:
        print(f"{row[1]:>6.2f} {row[2]:>8} {row[3]:>15.6f}")


def run_pagerank_convergence_experiment(
    n: int = 500,
    p: float = 0.1,
    damping_values: list = None,
    seed: int = 42,
    out_csv: str = "experiments/centrality/results/pagerank_convergence.csv"
) -> None:
    """
    Study how damping factor affects PageRank convergence.
    """
    if damping_values is None:
        damping_values = [0.5, 0.7, 0.85, 0.9, 0.95, 0.99]

    header = ["n", "p", "damping", "pagerank_time", "top_node", "top_score"]
    rows = []

    print("=== PageRank Convergence Experiment ===")
    print(f"Graph: {n} nodes, p={p}")
    print(f"Testing damping factors: {damping_values}")
    print()

    graph = generate_random_graph(n, p, seed=seed)

    for d in damping_values:
        result, pr_time = time_function(pagerank, graph, damping=d)

        # Find top node
        top_node = max(result.items(), key=lambda x: x[1])

        rows.append([n, p, d, pr_time, top_node[0], top_node[1]])
        print(f"  d={d:.2f}: time={pr_time:.4f}s, top_node={top_node[0]}, score={top_node[1]:.6f}")

    write_csv(out_csv, header, rows)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    run_pagerank_size_experiment()
    print("\n" + "=" * 70 + "\n")
    run_pagerank_density_experiment()
    print("\n" + "=" * 70 + "\n")
    run_pagerank_convergence_experiment()
