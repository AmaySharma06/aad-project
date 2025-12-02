"""
Experiment: Community Detection Performance.

Tests the Louvain and Leiden algorithms:
1. Runtime scaling with graph size and density
2. Community quality (modularity, NMI with ground truth)
3. Comparison between algorithms
4. Resolution parameter effects
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph.graph_generator import generate_random_graph
from graph.social_network import generate_community_network
from algorithms.community.louvain import louvain_communities
from algorithms.community.leiden import leiden_communities
from algorithms.community.modularity import (
    compute_modularity,
    normalized_mutual_information,
    adjusted_rand_index,
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
    Benchmark Louvain and Leiden algorithms on graphs of varying sizes.

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

    header = [
        "n", "p", "edges",
        "louvain_time", "louvain_communities", "louvain_modularity",
        "leiden_time", "leiden_communities", "leiden_modularity"
    ]
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
        (louvain_partition, louvain_mod), louvain_time = time_function(
            louvain_communities, graph, seed=seed
        )
        louvain_num = len(set(louvain_partition.values()))

        # Run Leiden
        (leiden_partition, leiden_mod), leiden_time = time_function(
            leiden_communities, graph, seed=seed
        )
        leiden_num = len(set(leiden_partition.values()))

        rows.append([
            n, p, edges,
            louvain_time, louvain_num, louvain_mod,
            leiden_time, leiden_num, leiden_mod
        ])

    print_progress(len(sizes), len(sizes), prefix="Complete")
    print()

    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    # Summary - matching NetworkX output format
    print("\n--- Summary ---")
    print(f"{'n':>6} {'Louvain(s)':>11} {'Leiden(s)':>11} {'Louvain Q':>11} {'Leiden Q':>11} {'Louvain #':>11} {'Leiden #':>11}")
    print("-" * 80)
    for row in rows:
        # row: [n, p, edges, louv_time, louv_num, louv_mod, leid_time, leid_num, leid_mod]
        print(f"{row[0]:>6} {row[3]:>11.4f} {row[6]:>11.4f} {row[5]:>11.4f} {row[8]:>11.4f} {row[4]:>11} {row[7]:>11}")


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

    Tests how well Louvain and Leiden recover ground truth communities
    as intra-community density varies.

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
        "n", "num_planted", "p_intra", "p_inter",
        "louvain_time", "louvain_detected", "louvain_modularity", "louvain_nmi", "louvain_ari",
        "leiden_time", "leiden_detected", "leiden_modularity", "leiden_nmi", "leiden_ari"
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
        (louvain_partition, louvain_mod), louvain_time = time_function(
            louvain_communities, graph, seed=seed
        )
        louvain_num = len(set(louvain_partition.values()))
        louvain_nmi = normalized_mutual_information(ground_truth, louvain_partition)
        louvain_ari = adjusted_rand_index(ground_truth, louvain_partition)

        # Run Leiden
        (leiden_partition, leiden_mod), leiden_time = time_function(
            leiden_communities, graph, seed=seed
        )
        leiden_num = len(set(leiden_partition.values()))
        leiden_nmi = normalized_mutual_information(ground_truth, leiden_partition)
        leiden_ari = adjusted_rand_index(ground_truth, leiden_partition)

        rows.append([
            n, num_communities, p_intra, p_inter,
            louvain_time, louvain_num, louvain_mod, louvain_nmi, louvain_ari,
            leiden_time, leiden_num, leiden_mod, leiden_nmi, leiden_ari
        ])

        print(f"p_intra={p_intra:.2f}:")
        print(f"  Louvain: {louvain_num} comms, Q={louvain_mod:.4f}, NMI={louvain_nmi:.4f}, ARI={louvain_ari:.4f}")
        print(f"  Leiden:  {leiden_num} comms, Q={leiden_mod:.4f}, NMI={leiden_nmi:.4f}, ARI={leiden_ari:.4f}")

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

    Parameters
    ----------
    n : int, optional
        Graph size.

    densities : list[float], optional
        Edge probabilities to test.

    seed : int, optional
        Random seed.

    out_csv : str, optional
        Output CSV path.
    """
    if densities is None:
        densities = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    header = [
        "n", "p", "edges",
        "louvain_time", "louvain_communities", "louvain_modularity",
        "leiden_time", "leiden_communities", "leiden_modularity"
    ]
    rows = []

    print("=== Community Density Experiment ===")
    print(f"Graph size: {n} nodes")
    print(f"Testing densities: {densities}")
    print()

    for i, p in enumerate(densities):
        print_progress(i, len(densities), prefix=f"Density {p:.2f}")

        graph = generate_random_graph(n, p, seed=seed + i)
        edges = sum(len(neighbors) for neighbors in graph.values()) // 2

        # Louvain
        (louvain_partition, louvain_mod), louvain_time = time_function(
            louvain_communities, graph, seed=seed
        )
        louvain_num = len(set(louvain_partition.values()))

        # Leiden
        (leiden_partition, leiden_mod), leiden_time = time_function(
            leiden_communities, graph, seed=seed
        )
        leiden_num = len(set(leiden_partition.values()))

        rows.append([
            n, p, edges,
            louvain_time, louvain_num, louvain_mod,
            leiden_time, leiden_num, leiden_mod
        ])

    print_progress(len(densities), len(densities), prefix="Complete")
    print()

    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    # Summary - matching NetworkX output format
    print("\n--- Summary ---")
    print(f"{'p':>6} {'Louvain(s)':>11} {'Leiden(s)':>11} {'Louvain Q':>11} {'Leiden Q':>11} {'Louvain #':>11} {'Leiden #':>11}")
    print("-" * 80)
    for row in rows:
        # row: [n, p, edges, louv_time, louv_num, louv_mod, leid_time, leid_num, leid_mod]
        print(f"{row[1]:>6.2f} {row[3]:>11.4f} {row[6]:>11.4f} {row[5]:>11.4f} {row[8]:>11.4f} {row[4]:>11} {row[7]:>11}")


def run_resolution_experiment(
    n: int = 500,
    num_communities: int = 10,
    p_intra: float = 0.3,
    p_inter: float = 0.01,
    resolutions: list = None,
    seed: int = 42,
    out_csv: str = "experiments/community/results/resolution_results.csv"
) -> None:
    """
    Study the effect of resolution parameter on community detection.

    Parameters
    ----------
    n : int, optional
        Graph size.

    num_communities : int, optional
        Number of planted communities.

    p_intra : float, optional
        Intra-community edge probability.

    p_inter : float, optional
        Inter-community edge probability.

    resolutions : list[float], optional
        Resolution values to test.

    seed : int, optional
        Random seed.

    out_csv : str, optional
        Output CSV path.
    """
    if resolutions is None:
        resolutions = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

    # Generate network once
    network, ground_truth = generate_community_network(
        n, num_communities, p_intra, p_inter, seed=seed
    )
    graph = network.adjacency

    header = [
        "resolution", "algorithm",
        "num_communities", "modularity", "nmi", "ari"
    ]
    rows = []

    print("=== Resolution Parameter Experiment ===")
    print(f"Graph: {n} nodes, {num_communities} planted communities")
    print(f"p_intra={p_intra}, p_inter={p_inter}")
    print(f"Testing resolutions: {resolutions}")
    print()

    for res in resolutions:
        # Louvain
        louvain_partition, louvain_mod = louvain_communities(graph, resolution=res, seed=seed)
        louvain_num = len(set(louvain_partition.values()))
        louvain_nmi = normalized_mutual_information(ground_truth, louvain_partition)
        louvain_ari = adjusted_rand_index(ground_truth, louvain_partition)
        
        rows.append([res, "louvain", louvain_num, louvain_mod, louvain_nmi, louvain_ari])

        # Leiden
        leiden_partition, leiden_mod = leiden_communities(graph, resolution=res, seed=seed)
        leiden_num = len(set(leiden_partition.values()))
        leiden_nmi = normalized_mutual_information(ground_truth, leiden_partition)
        leiden_ari = adjusted_rand_index(ground_truth, leiden_partition)
        
        rows.append([res, "leiden", leiden_num, leiden_mod, leiden_nmi, leiden_ari])

        print(f"Resolution {res:.2f}:")
        print(f"  Louvain: {louvain_num} comms, NMI={louvain_nmi:.4f}")
        print(f"  Leiden:  {leiden_num} comms, NMI={leiden_nmi:.4f}")

    write_csv(out_csv, header, rows)
    print(f"\nResults saved to {out_csv}")


def run_algorithm_comparison(
    n: int = 500,
    num_communities: int = 5,
    p_intra: float = 0.4,
    p_inter: float = 0.02,
    num_trials: int = 10,
    seed: int = 42,
    out_csv: str = "experiments/community/results/comparison_results.csv"
) -> None:
    """
    Compare Louvain and Leiden across multiple random trials.

    Parameters
    ----------
    n : int
        Graph size.

    num_communities : int
        Number of planted communities.

    p_intra : float
        Intra-community edge probability.

    p_inter : float
        Inter-community edge probability.

    num_trials : int
        Number of random trials.

    seed : int
        Base random seed.

    out_csv : str
        Output CSV path.
    """
    header = [
        "trial", "algorithm",
        "time", "num_communities", "modularity", "nmi", "ari"
    ]
    rows = []

    print("=== Algorithm Comparison Experiment ===")
    print(f"Graph: {n} nodes, {num_communities} planted communities")
    print(f"p_intra={p_intra}, p_inter={p_inter}")
    print(f"Number of trials: {num_trials}")
    print()

    louvain_times = []
    leiden_times = []
    louvain_nmis = []
    leiden_nmis = []

    for trial in range(num_trials):
        print_progress(trial, num_trials, prefix=f"Trial {trial+1}")
        
        # Generate new network for each trial
        network, ground_truth = generate_community_network(
            n, num_communities, p_intra, p_inter, seed=seed + trial
        )
        graph = network.adjacency

        # Louvain
        (louvain_partition, louvain_mod), louvain_time = time_function(
            louvain_communities, graph, seed=seed + trial
        )
        louvain_num = len(set(louvain_partition.values()))
        louvain_nmi = normalized_mutual_information(ground_truth, louvain_partition)
        louvain_ari = adjusted_rand_index(ground_truth, louvain_partition)
        
        rows.append([trial, "louvain", louvain_time, louvain_num, louvain_mod, louvain_nmi, louvain_ari])
        louvain_times.append(louvain_time)
        louvain_nmis.append(louvain_nmi)

        # Leiden
        (leiden_partition, leiden_mod), leiden_time = time_function(
            leiden_communities, graph, seed=seed + trial
        )
        leiden_num = len(set(leiden_partition.values()))
        leiden_nmi = normalized_mutual_information(ground_truth, leiden_partition)
        leiden_ari = adjusted_rand_index(ground_truth, leiden_partition)
        
        rows.append([trial, "leiden", leiden_time, leiden_num, leiden_mod, leiden_nmi, leiden_ari])
        leiden_times.append(leiden_time)
        leiden_nmis.append(leiden_nmi)

    print_progress(num_trials, num_trials, prefix="Complete")
    print()

    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Louvain avg time: {sum(louvain_times)/len(louvain_times):.4f}s")
    print(f"Leiden avg time:  {sum(leiden_times)/len(leiden_times):.4f}s")
    print(f"Louvain avg NMI:  {sum(louvain_nmis)/len(louvain_nmis):.4f}")
    print(f"Leiden avg NMI:   {sum(leiden_nmis)/len(leiden_nmis):.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run community detection experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["size", "quality", "density", "resolution", "comparison", "all"],
                        help="Which experiment to run")
    args = parser.parse_args()
    
    if args.experiment == "size" or args.experiment == "all":
        run_community_size_experiment()
        print("\n" + "=" * 70 + "\n")
    
    if args.experiment == "quality" or args.experiment == "all":
        run_community_quality_experiment()
        print("\n" + "=" * 70 + "\n")
    
    if args.experiment == "density" or args.experiment == "all":
        run_community_density_experiment()
        print("\n" + "=" * 70 + "\n")
    
    if args.experiment == "resolution" or args.experiment == "all":
        run_resolution_experiment()
        print("\n" + "=" * 70 + "\n")
    
    if args.experiment == "comparison" or args.experiment == "all":
        run_algorithm_comparison()
