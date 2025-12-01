"""
Experiment: BFS and DFS Runtime vs Graph Size.

Measures how BFS and DFS scale with the number of nodes.
Expected: O(V + E) for both algorithms.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph.graph_generator import generate_random_graph
from algorithms.traversal.bfs import bfs_traversal, bfs_all_paths
from algorithms.traversal.dfs import dfs_traversal, dfs_iterative
from algorithms.traversal.union_find import count_components_union_find
from utils import write_csv, time_function, print_progress


def run_traversal_size_experiment(
    sizes: list = None,
    p: float = 0.1,
    seed: int = 42,
    out_csv: str = "experiments/traversal/results/size_results.csv"
) -> None:
    """
    Benchmark traversal algorithms on graphs of varying sizes.

    Parameters
    ----------
    sizes : list[int], optional
        Graph sizes to test.

    p : float, optional (default=0.1)
        Edge probability (controls density).

    seed : int, optional
        Random seed.

    out_csv : str, optional
        Output CSV file path.
    """
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000, 5000]

    header = [
        "n",
        "p",
        "edges",
        "bfs_time",
        "dfs_recursive_time",
        "dfs_iterative_time",
        "bfs_all_paths_time",
        "union_find_time"
    ]
    rows = []

    print("=== Traversal Size Experiment ===")
    print(f"Testing sizes: {sizes}")
    print(f"Edge probability: {p}")
    print()

    for i, n in enumerate(sizes):
        print_progress(i, len(sizes), prefix=f"Size {n:>5}")

        # Generate graph
        graph = generate_random_graph(n, p, seed=seed + i)
        edges = sum(len(neighbors) for neighbors in graph.values()) // 2

        # Pick starting node (0)
        start = 0

        # BFS traversal
        _, bfs_time = time_function(bfs_traversal, graph, start)

        # DFS recursive
        _, dfs_rec_time = time_function(dfs_traversal, graph, start)

        # DFS iterative
        _, dfs_iter_time = time_function(dfs_iterative, graph, start)

        # BFS all paths (shortest paths from source)
        _, bfs_paths_time = time_function(bfs_all_paths, graph, start)

        # Union-Find component count
        _, uf_time = time_function(count_components_union_find, graph)

        rows.append([
            n,
            p,
            edges,
            bfs_time,
            dfs_rec_time,
            dfs_iter_time,
            bfs_paths_time,
            uf_time
        ])

    print_progress(len(sizes), len(sizes), prefix="Complete")
    print()

    # Save results
    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    # Print summary
    print("\n--- Summary ---")
    print(f"{'n':>6} {'edges':>8} {'BFS':>10} {'DFS(rec)':>10} {'DFS(iter)':>10} {'Union-Find':>12}")
    print("-" * 70)
    for row in rows:
        print(f"{row[0]:>6} {row[2]:>8} {row[3]:>10.6f} {row[4]:>10.6f} {row[5]:>10.6f} {row[7]:>12.6f}")


def run_traversal_density_experiment(
    n: int = 500,
    densities: list = None,
    seed: int = 42,
    out_csv: str = "experiments/traversal/results/density_results.csv"
) -> None:
    """
    Benchmark traversal algorithms on graphs of varying densities.

    Parameters
    ----------
    n : int, optional (default=500)
        Number of nodes.

    densities : list[float], optional
        Edge probabilities to test.

    seed : int, optional
        Random seed.

    out_csv : str, optional
        Output CSV file path.
    """
    if densities is None:
        densities = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    header = [
        "n",
        "p",
        "edges",
        "bfs_time",
        "dfs_recursive_time",
        "dfs_iterative_time",
        "union_find_time"
    ]
    rows = []

    print("=== Traversal Density Experiment ===")
    print(f"Graph size: {n} nodes")
    print(f"Testing densities: {densities}")
    print()

    for i, p in enumerate(densities):
        print_progress(i, len(densities), prefix=f"Density {p:.2f}")

        # Generate graph
        graph = generate_random_graph(n, p, seed=seed + i)
        edges = sum(len(neighbors) for neighbors in graph.values()) // 2

        start = 0

        # Time each algorithm
        _, bfs_time = time_function(bfs_traversal, graph, start)
        _, dfs_rec_time = time_function(dfs_traversal, graph, start)
        _, dfs_iter_time = time_function(dfs_iterative, graph, start)
        _, uf_time = time_function(count_components_union_find, graph)

        rows.append([
            n,
            p,
            edges,
            bfs_time,
            dfs_rec_time,
            dfs_iter_time,
            uf_time
        ])

    print_progress(len(densities), len(densities), prefix="Complete")
    print()

    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    # Print summary
    print("\n--- Summary ---")
    print(f"{'p':>6} {'edges':>8} {'BFS':>10} {'DFS(rec)':>10} {'DFS(iter)':>10} {'Union-Find':>12}")
    print("-" * 70)
    for row in rows:
        print(f"{row[1]:>6.2f} {row[2]:>8} {row[3]:>10.6f} {row[4]:>10.6f} {row[5]:>10.6f} {row[6]:>12.6f}")


if __name__ == "__main__":
    # Run both experiments
    run_traversal_size_experiment()
    print("\n" + "=" * 70 + "\n")
    run_traversal_density_experiment()
