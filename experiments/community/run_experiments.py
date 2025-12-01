"""
Experiment: Community Detection Performance.

Tests the Louvain algorithm:
1. Runtime scaling with graph size and density
2. Community quality (modularity, NMI with ground truth)
3. Impact of community structure strength
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph.graph_generator import generate_random_graph
from graph.social_network import generate_community_network
from algorithms.community.louvain import louvain_communities
from algorithms.community.modularity import (
    compute_modularity,
    normalized_mutual_information,
    get_communities_list
)
from utils import write_csv, time_function, print_progress


def run_community_size_experiment(
    sizes: list = None,
    p: float = 0.1,
    seed: int = 42,
    out_csv: str = "experiments/community/results/size_results.csv"
) -> None:
    """
    Benchmark Louvain algorithm on graphs of varying sizes.

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

    header = ["n", "p", "edges", "louvain_time", "num_communities", "modularity"]
    rows = []

    print("=== Community Detection Size Experiment ===")
    print(f"Testing sizes: {sizes}")
    print(f"Edge probability: {p}")
    print()

    for i, n in enumerate(sizes):
        print_progress(i, len(sizes), prefix=f"Size {n:>5}")

        graph = generate_random_graph(n, p, seed=seed + i)
        edges = sum(len(neighbors) for neighbors in graph.values()) // 2

        # Run Louvain
        (partition, modularity), louvain_time = time_function(
            louvain_communities, graph, seed=seed
        )

        num_communities = len(set(partition.values()))

        rows.append([n, p, edges, louvain_time, num_communities, modularity])

    print_progress(len(sizes), len(sizes), prefix="Complete")
    print()

    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    # Summary
    print("\n--- Summary ---")
    print(f"{'n':>6} {'edges':>8} {'Time':>10} {'#Comm':>8} {'Modularity':>12}")
    print("-" * 55)
    for row in rows:
        print(f"{row[0]:>6} {row[2]:>8} {row[3]:>10.6f} {row[4]:>8} {row[5]:>12.4f}")


def run_community_quality_experiment(
    n: int = 200,
    num_communities: int = 4,
    p_intra_values: list = None,
    p_inter: float = 0.01,
    seed: int = 42,
    out_csv: str = "experiments/community/results/quality_results.csv"
) -> None:
    """
    Evaluate community detection quality using planted partition model.

    Tests how well Louvain recovers ground truth communities as
    intra-community density varies.

    Parameters
    ----------
    n : int, optional
        Total number of nodes.

    num_communities : int, optional
        Number of planted communities.

    p_intra_values : list[float], optional
        Intra-community edge probabilities to test.

    p_inter : float, optional
        Inter-community edge probability.

    seed : int, optional
        Random seed.

    out_csv : str, optional
        Output CSV path.
    """
    if p_intra_values is None:
        p_intra_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    header = [
        "n",
        "num_planted",
        "p_intra",
        "p_inter",
        "louvain_time",
        "num_detected",
        "modularity",
        "nmi"
    ]
    rows = []

    print("=== Community Quality Experiment ===")
    print(f"Graph: {n} nodes, {num_communities} planted communities")
    print(f"p_inter: {p_inter}")
    print(f"Testing p_intra: {p_intra_values}")
    print()

    for p_intra in p_intra_values:
        # Generate network with planted communities
        network, ground_truth = generate_community_network(
            n, num_communities, p_intra, p_inter, seed=seed
        )
        graph = network.adjacency

        # Run Louvain
        (partition, modularity), louvain_time = time_function(
            louvain_communities, graph, seed=seed
        )

        num_detected = len(set(partition.values()))

        # Compare to ground truth
        nmi = normalized_mutual_information(ground_truth, partition)

        rows.append([
            n, num_communities, p_intra, p_inter,
            louvain_time, num_detected, modularity, nmi
        ])

        print(f"  p_intra={p_intra:.2f}: detected={num_detected}, "
              f"Q={modularity:.4f}, NMI={nmi:.4f}")

    write_csv(out_csv, header, rows)
    print(f"\nResults saved to {out_csv}")


def run_community_density_experiment(
    n: int = 500,
    densities: list = None,
    seed: int = 42,
    out_csv: str = "experiments/community/results/density_results.csv"
) -> None:
    """
    Study how graph density affects community detection.
    """
    if densities is None:
        densities = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    header = ["n", "p", "edges", "louvain_time", "num_communities", "modularity"]
    rows = []

    print("=== Community Density Experiment ===")
    print(f"Graph size: {n} nodes")
    print(f"Testing densities: {densities}")
    print()

    for i, p in enumerate(densities):
        print_progress(i, len(densities), prefix=f"Density {p:.2f}")

        graph = generate_random_graph(n, p, seed=seed + i)
        edges = sum(len(neighbors) for neighbors in graph.values()) // 2

        (partition, modularity), louvain_time = time_function(
            louvain_communities, graph, seed=seed
        )

        num_communities = len(set(partition.values()))

        rows.append([n, p, edges, louvain_time, num_communities, modularity])

    print_progress(len(densities), len(densities), prefix="Complete")
    print()

    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    # Summary
    print("\n--- Summary ---")
    print(f"{'p':>6} {'edges':>8} {'Time':>10} {'#Comm':>8} {'Modularity':>12}")
    print("-" * 55)
    for row in rows:
        print(f"{row[1]:>6.2f} {row[2]:>8} {row[3]:>10.6f} {row[4]:>8} {row[5]:>12.4f}")


if __name__ == "__main__":
    run_community_size_experiment()
    print("\n" + "=" * 70 + "\n")
    run_community_quality_experiment()
    print("\n" + "=" * 70 + "\n")
    run_community_density_experiment()
