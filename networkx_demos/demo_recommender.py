import sys
import os
import networkx as nx
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.social_network import generate_social_network
from graph.noise import split_edges_for_testing, apply_noise
from utils import write_csv, time_function, print_progress

def to_nx_graph(adj_list):
    G = nx.Graph()
    for u, neighbors in adj_list.items():
        G.add_node(u)
        for v in neighbors:
            if u < v:
                G.add_edge(u, v)
    return G

def run_nx_recommender_size_experiment(
    sizes: list = None,
    p: float = 0.1,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/recommender_size_nx.csv"
) -> None:
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000]

    header = ["n", "p", "edges", "jaccard_all_pairs_time"]
    rows = []

    print("=== NetworkX Recommender Size Experiment ===")
    print(f"Testing sizes: {sizes}")
    print(f"Edge probability: {p}")
    print()

    for i, n in enumerate(sizes):
        print_progress(i, len(sizes), prefix=f"Size {n:>5}")

        network = generate_social_network(n, p, seed=seed + i)
        G = to_nx_graph(network.adjacency)
        edges = G.number_of_edges()

        start = time.time()
        # nx.jaccard_coefficient returns an iterator, so we must consume it
        _ = list(nx.jaccard_coefficient(G))
        jac_time = time.time() - start

        rows.append([n, p, edges, jac_time])

    print_progress(len(sizes), len(sizes), prefix="Complete")
    print()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    print("\n--- Summary ---")
    print(f"{'n':>6} {'edges':>8} {'Jaccard':>12}")
    print("-" * 30)
    for row in rows:
        print(f"{row[0]:>6} {row[2]:>8} {row[3]:>12.6f}")


def evaluate_nx_recommendations(
    train_graph_nx,
    test_edges,
    k=10
):
    # For NetworkX, we can use Jaccard Coefficient or Adamic-Adar
    # Let's use Jaccard as the primary metric to compare with our Jaccard
    
    # Precompute all pairs jaccard
    preds = list(nx.jaccard_coefficient(train_graph_nx))
    # preds is list of (u, v, p)
    
    # Organize by user
    # We need to recommend for all users involved in test_edges
    test_users = set()
    for u, v in test_edges:
        test_users.add(u)
        test_users.add(v)
        
    user_recs = {u: [] for u in test_users}
    
    for u, v, p in preds:
        if u in user_recs:
            user_recs[u].append((v, p))
        if v in user_recs:
            user_recs[v].append((u, p))
            
    # Sort and pick top k
    for u in user_recs:
        user_recs[u].sort(key=lambda x: x[1], reverse=True)
        user_recs[u] = [x[0] for x in user_recs[u][:k]]
        
    # Calculate metrics
    total_precision = 0.0
    total_recall = 0.0
    num_hits = 0
    num_users = 0
    
    for user in test_users:
        if user not in user_recs:
            continue
            
        recommended_set = set(user_recs[user])
        
        # Find test partners
        user_test_partners = set()
        for u, v in test_edges:
            if u == user:
                user_test_partners.add(v)
            elif v == user:
                user_test_partners.add(u)
                
        if not user_test_partners:
            continue
            
        hits = recommended_set & user_test_partners
        precision = len(hits) / len(recommended_set) if recommended_set else 0
        recall = len(hits) / len(user_test_partners) if user_test_partners else 0
        
        total_precision += precision
        total_recall += recall
        if hits:
            num_hits += 1
        num_users += 1
        
    if num_users == 0:
        return {"precision": 0, "recall": 0, "hit_rate": 0}
        
    return {
        "precision": total_precision / num_users,
        "recall": total_recall / num_users,
        "hit_rate": num_hits / num_users
    }


def run_nx_recommender_quality_experiment(
    n: int = 500,
    p: float = 0.1,
    test_fractions: list = None,
    k_values: list = None,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/recommender_quality_nx.csv"
) -> None:
    if test_fractions is None:
        test_fractions = [0.1, 0.2, 0.3]
    if k_values is None:
        k_values = [5, 10, 20]

    header = ["n", "p", "test_fraction", "k", "precision", "recall", "hit_rate"]
    rows = []

    print("=== NetworkX Recommender Quality Experiment ===")
    print(f"Network: {n} nodes, p={p}")
    print(f"Test fractions: {test_fractions}")
    print(f"K values: {k_values}")
    print()

    network = generate_social_network(n, p, seed=seed)
    graph = network.adjacency

    for test_frac in test_fractions:
        train_graph, test_edges = split_edges_for_testing(
            graph, test_fraction=test_frac, seed=seed
        )
        train_graph_nx = to_nx_graph(train_graph)

        for k in k_values:
            metrics = evaluate_nx_recommendations(train_graph_nx, test_edges, k=k)

            rows.append([
                n, p, test_frac, k,
                metrics["precision"],
                metrics["recall"],
                metrics["hit_rate"]
            ])

            print(f"  test_frac={test_frac:.1f}, k={k:>2}: "
                  f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                  f"Hit={metrics['hit_rate']:.3f}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"\nResults saved to {out_csv}")


def run_nx_recommender_noise_experiment(
    n: int = 500,
    p: float = 0.1,
    noise_levels: list = None,
    k: int = 10,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/recommender_noise_nx.csv"
) -> None:
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

    header = ["n", "p", "noise_level", "k", "precision", "recall", "hit_rate"]
    rows = []

    print("=== NetworkX Recommender Noise Experiment ===")
    print(f"Network: {n} nodes, p={p}")
    print(f"Noise levels: {noise_levels}")
    print()

    network = generate_social_network(n, p, seed=seed)
    original_graph = network.adjacency

    train_original, test_edges = split_edges_for_testing(
        original_graph, test_fraction=0.2, seed=seed
    )

    for noise in noise_levels:
        if noise > 0:
            noisy_train = apply_noise(
                train_original,
                add_fraction=noise / 2,
                remove_fraction=noise / 2,
                seed=seed
            )
        else:
            noisy_train = train_original
            
        noisy_train_nx = to_nx_graph(noisy_train)

        metrics = evaluate_nx_recommendations(noisy_train_nx, test_edges, k=k)

        rows.append([
            n, p, noise, k,
            metrics["precision"],
            metrics["recall"],
            metrics["hit_rate"]
        ])

        print(f"  noise={noise:.2f}: P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, Hit={metrics['hit_rate']:.3f}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    run_nx_recommender_size_experiment()
    print("\n" + "=" * 70 + "\n")
    run_nx_recommender_quality_experiment()
    print("\n" + "=" * 70 + "\n")
    run_nx_recommender_noise_experiment()
