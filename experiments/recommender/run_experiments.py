"""
Experiment: Friend Recommender Performance.

Tests Jaccard similarity and friend recommendation algorithms:
1. Runtime scaling with graph size and density
2. Recommendation quality (precision, recall, hit rate)
3. Impact of noise on recommendation accuracy
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph.social_network import generate_social_network
from graph.noise import split_edges_for_testing, apply_noise
from algorithms.recommender.jaccard import jaccard_similarity_all_pairs
from algorithms.recommender.friend_recommender import (
    FriendRecommender,
    recommend_friends,
    evaluate_recommendations
)
from utils import write_csv, time_function, print_progress


def run_recommender_size_experiment(
    sizes: list = None,
    p: float = 0.1,
    seed: int = 42,
    out_csv: str = "experiments/recommender/results/size_results.csv"
) -> None:
    """
    Benchmark recommender algorithms on social networks of varying sizes.

    Parameters
    ----------
    sizes : list[int], optional
        Network sizes to test.

    p : float, optional
        Friendship probability.

    seed : int, optional
        Random seed.

    out_csv : str, optional
        Output CSV path.
    """
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000]

    header = [
        "n",
        "p",
        "edges",
        "jaccard_all_pairs_time",
        "recommend_single_time",
        "recommend_all_time"
    ]
    rows = []

    print("=== Recommender Size Experiment ===")
    print(f"Testing sizes: {sizes}")
    print(f"Edge probability: {p}")
    print()

    for i, n in enumerate(sizes):
        print_progress(i, len(sizes), prefix=f"Size {n:>5}")

        # Generate social network
        network = generate_social_network(n, p, seed=seed + i)
        graph = network.adjacency
        tags = network.tags
        edges = network.num_friendships()

        # Time Jaccard all pairs
        _, jac_time = time_function(jaccard_similarity_all_pairs, graph)

        # Time single user recommendation
        recommender = FriendRecommender(graph, tags)
        _, rec_single_time = time_function(recommender.recommend, 0, 10)

        # Time recommend all
        _, rec_all_time = time_function(recommender.recommend_all, 10)

        rows.append([n, p, edges, jac_time, rec_single_time, rec_all_time])

    print_progress(len(sizes), len(sizes), prefix="Complete")
    print()

    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    # Summary
    print("\n--- Summary ---")
    print(f"{'n':>6} {'edges':>8} {'Jaccard':>12} {'Single rec':>12} {'All recs':>12}")
    print("-" * 60)
    for row in rows:
        print(f"{row[0]:>6} {row[2]:>8} {row[3]:>12.6f} {row[4]:>12.6f} {row[5]:>12.6f}")


def run_recommender_quality_experiment(
    n: int = 500,
    p: float = 0.1,
    test_fractions: list = None,
    k_values: list = None,
    seed: int = 42,
    out_csv: str = "experiments/recommender/results/quality_results.csv"
) -> None:
    """
    Evaluate recommendation quality using train/test split.

    Parameters
    ----------
    n : int, optional
        Network size.

    p : float, optional
        Friendship probability.

    test_fractions : list[float], optional
        Fractions of edges to hold out for testing.

    k_values : list[int], optional
        Number of recommendations to evaluate.

    seed : int, optional
        Random seed.

    out_csv : str, optional
        Output CSV path.
    """
    if test_fractions is None:
        test_fractions = [0.1, 0.2, 0.3]

    if k_values is None:
        k_values = [5, 10, 20]

    header = ["n", "p", "test_fraction", "k", "precision", "recall", "hit_rate"]
    rows = []

    print("=== Recommender Quality Experiment ===")
    print(f"Network: {n} nodes, p={p}")
    print(f"Test fractions: {test_fractions}")
    print(f"K values: {k_values}")
    print()

    # Generate base network
    network = generate_social_network(n, p, seed=seed)
    graph = network.adjacency
    tags = network.tags

    for test_frac in test_fractions:
        # Split into train/test
        train_graph, test_edges = split_edges_for_testing(
            graph, test_fraction=test_frac, seed=seed
        )

        for k in k_values:
            # Evaluate
            metrics = evaluate_recommendations(train_graph, test_edges, tags, k=k)

            rows.append([
                n, p, test_frac, k,
                metrics["precision"],
                metrics["recall"],
                metrics["hit_rate"]
            ])

            print(f"  test_frac={test_frac:.1f}, k={k:>2}: "
                  f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                  f"Hit={metrics['hit_rate']:.3f}")

    write_csv(out_csv, header, rows)
    print(f"\nResults saved to {out_csv}")


def run_recommender_noise_experiment(
    n: int = 500,
    p: float = 0.1,
    noise_levels: list = None,
    k: int = 10,
    seed: int = 42,
    out_csv: str = "experiments/recommender/results/noise_results.csv"
) -> None:
    """
    Study how noise affects recommendation quality.

    Parameters
    ----------
    n : int, optional
        Network size.

    p : float, optional
        Friendship probability.

    noise_levels : list[float], optional
        Fractions of edges to perturb.

    k : int, optional
        Number of recommendations.

    seed : int, optional
        Random seed.

    out_csv : str, optional
        Output CSV path.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

    header = ["n", "p", "noise_level", "k", "precision", "recall", "hit_rate"]
    rows = []

    print("=== Recommender Noise Experiment ===")
    print(f"Network: {n} nodes, p={p}")
    print(f"Noise levels: {noise_levels}")
    print()

    # Generate base network
    network = generate_social_network(n, p, seed=seed)
    original_graph = network.adjacency
    tags = network.tags

    # Create train/test split from original
    train_original, test_edges = split_edges_for_testing(
        original_graph, test_fraction=0.2, seed=seed
    )

    for noise in noise_levels:
        # Apply noise to training graph
        if noise > 0:
            noisy_train = apply_noise(
                train_original,
                add_fraction=noise / 2,
                remove_fraction=noise / 2,
                seed=seed
            )
        else:
            noisy_train = train_original

        # Evaluate on noisy training data
        metrics = evaluate_recommendations(noisy_train, test_edges, tags, k=k)

        rows.append([
            n, p, noise, k,
            metrics["precision"],
            metrics["recall"],
            metrics["hit_rate"]
        ])

        print(f"  noise={noise:.2f}: P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, Hit={metrics['hit_rate']:.3f}")

    write_csv(out_csv, header, rows)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    run_recommender_size_experiment()
    print("\n" + "=" * 70 + "\n")
    run_recommender_quality_experiment()
    print("\n" + "=" * 70 + "\n")
    run_recommender_noise_experiment()
