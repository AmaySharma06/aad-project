import sys
import os
import networkx as nx
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.graph_generator import generate_random_graph
from utils import write_csv, time_function, print_progress

def to_nx_graph(adj_list):
    G = nx.Graph()
    for u, neighbors in adj_list.items():
        G.add_node(u)
        for v in neighbors:
            if u < v:
                G.add_edge(u, v)
    return G

def run_nx_pagerank_size_experiment(
    sizes: list = None,
    p: float = 0.1,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/pagerank_size_nx.csv"
) -> None:
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000]

    header = ["n", "p", "edges", "pagerank_time"]
    rows = []

    print("=== NetworkX PageRank Size Experiment ===")
    print(f"Testing sizes: {sizes}")
    print(f"Edge probability: {p}")
    print()

    for i, n in enumerate(sizes):
        print_progress(i, len(sizes), prefix=f"Size {n:>5}")

        adj_list = generate_random_graph(n, p, seed=seed + i)
        G = to_nx_graph(adj_list)
        edges = G.number_of_edges()

        start = time.time()
        nx.pagerank(G)
        pr_time = time.time() - start

        rows.append([n, p, edges, pr_time])

    print_progress(len(sizes), len(sizes), prefix="Complete")
    print()

    # Ensure results directory exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    print("\n--- Summary ---")
    print(f"{'n':>6} {'edges':>8} {'PageRank time':>15}")
    print("-" * 35)
    for row in rows:
        print(f"{row[0]:>6} {row[2]:>8} {row[3]:>15.6f}")


def run_nx_pagerank_density_experiment(
    n: int = 500,
    densities: list = None,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/pagerank_density_nx.csv"
) -> None:
    if densities is None:
        densities = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    header = ["n", "p", "edges", "pagerank_time"]
    rows = []

    print("=== NetworkX PageRank Density Experiment ===")
    print(f"Graph size: {n} nodes")
    print(f"Testing densities: {densities}")
    print()

    for i, p in enumerate(densities):
        print_progress(i, len(densities), prefix=f"Density {p:.2f}")

        adj_list = generate_random_graph(n, p, seed=seed + i)
        G = to_nx_graph(adj_list)
        edges = G.number_of_edges()

        start = time.time()
        nx.pagerank(G)
        pr_time = time.time() - start

        rows.append([n, p, edges, pr_time])

    print_progress(len(densities), len(densities), prefix="Complete")
    print()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    print("\n--- Summary ---")
    print(f"{'p':>6} {'edges':>8} {'PageRank time':>15}")
    print("-" * 35)
    for row in rows:
        print(f"{row[1]:>6.2f} {row[2]:>8} {row[3]:>15.6f}")


def run_nx_pagerank_convergence_experiment(
    n: int = 500,
    p: float = 0.1,
    damping_values: list = None,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/pagerank_convergence_nx.csv"
) -> None:
    if damping_values is None:
        damping_values = [0.5, 0.7, 0.85, 0.9, 0.95, 0.99]

    header = ["n", "p", "damping", "pagerank_time", "top_node", "top_score"]
    rows = []

    print("=== NetworkX PageRank Convergence Experiment ===")
    print(f"Graph: {n} nodes, p={p}")
    print(f"Testing damping factors: {damping_values}")
    print()

    adj_list = generate_random_graph(n, p, seed=seed)
    G = to_nx_graph(adj_list)

    for d in damping_values:
        start = time.time()
        result = nx.pagerank(G, alpha=d)
        pr_time = time.time() - start

        top_node = max(result.items(), key=lambda x: x[1])

        rows.append([n, p, d, pr_time, top_node[0], top_node[1]])
        print(f"  d={d:.2f}: time={pr_time:.4f}s, top_node={top_node[0]}, score={top_node[1]:.6f}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    run_nx_pagerank_size_experiment()
    print("\n" + "=" * 70 + "\n")
    run_nx_pagerank_density_experiment()
    print("\n" + "=" * 70 + "\n")
    run_nx_pagerank_convergence_experiment()
