import sys
import os
import networkx as nx
import time

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

def run_nx_traversal_size_experiment(
    sizes: list = None,
    p: float = 0.1,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/traversal_size_nx.csv"
) -> None:
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000, 5000]

    header = [
        "n", "p", "edges",
        "bfs_time", "dfs_time", "connected_components_time"
    ]
    rows = []

    print("=== NetworkX Traversal Size Experiment ===")
    print(f"Testing sizes: {sizes}")
    print(f"Edge probability: {p}")
    print()

    for i, n in enumerate(sizes):
        print_progress(i, len(sizes), prefix=f"Size {n:>5}")

        adj_list = generate_random_graph(n, p, seed=seed + i)
        G = to_nx_graph(adj_list)
        edges = G.number_of_edges()
        start_node = 0

        # BFS
        start = time.time()
        # nx.bfs_tree computes the BFS tree
        _ = nx.bfs_tree(G, source=start_node)
        bfs_time = time.time() - start

        # DFS
        start = time.time()
        _ = nx.dfs_tree(G, source=start_node)
        dfs_time = time.time() - start

        # Connected Components (similar to Union-Find)
        start = time.time()
        _ = list(nx.connected_components(G))
        cc_time = time.time() - start

        rows.append([n, p, edges, bfs_time, dfs_time, cc_time])

    print_progress(len(sizes), len(sizes), prefix="Complete")
    print()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    print("\n--- Summary ---")
    print(f"{'n':>6} {'edges':>8} {'BFS':>10} {'DFS':>10} {'CC':>10}")
    print("-" * 50)
    for row in rows:
        print(f"{row[0]:>6} {row[2]:>8} {row[3]:>10.6f} {row[4]:>10.6f} {row[5]:>10.6f}")


def run_nx_traversal_density_experiment(
    n: int = 500,
    densities: list = None,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/traversal_density_nx.csv"
) -> None:
    if densities is None:
        densities = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    header = [
        "n", "p", "edges",
        "bfs_time", "dfs_time", "connected_components_time"
    ]
    rows = []

    print("=== NetworkX Traversal Density Experiment ===")
    print(f"Graph size: {n} nodes")
    print(f"Testing densities: {densities}")
    print()

    for i, p in enumerate(densities):
        print_progress(i, len(densities), prefix=f"Density {p:.2f}")

        adj_list = generate_random_graph(n, p, seed=seed + i)
        G = to_nx_graph(adj_list)
        edges = G.number_of_edges()
        start_node = 0

        # BFS
        start = time.time()
        _ = nx.bfs_tree(G, source=start_node)
        bfs_time = time.time() - start

        # DFS
        start = time.time()
        _ = nx.dfs_tree(G, source=start_node)
        dfs_time = time.time() - start

        # Connected Components
        start = time.time()
        _ = list(nx.connected_components(G))
        cc_time = time.time() - start

        rows.append([n, p, edges, bfs_time, dfs_time, cc_time])

    print_progress(len(densities), len(densities), prefix="Complete")
    print()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    print("\n--- Summary ---")
    print(f"{'p':>6} {'edges':>8} {'BFS':>10} {'DFS':>10} {'CC':>10}")
    print("-" * 50)
    for row in rows:
        print(f"{row[1]:>6.2f} {row[2]:>8} {row[3]:>10.6f} {row[4]:>10.6f} {row[5]:>10.6f}")


if __name__ == "__main__":
    run_nx_traversal_size_experiment()
    print("\n" + "=" * 70 + "\n")
    run_nx_traversal_density_experiment()
