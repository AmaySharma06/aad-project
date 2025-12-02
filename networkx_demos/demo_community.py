"""
NetworkX and external library community detection demos.

Compares our implementations with NetworkX Louvain and leidenalg Leiden.
"""

import sys
import os
import networkx as nx
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.graph_generator import generate_random_graph
from graph.social_network import generate_community_network
from algorithms.community.modularity import normalized_mutual_information, adjusted_rand_index
from utils import write_csv, time_function, print_progress

# Try to import optional dependencies
try:
    import igraph as ig
    import leidenalg
    LEIDENALG_AVAILABLE = True
except ImportError:
    LEIDENALG_AVAILABLE = False
    print("Warning: leidenalg not available. Install with: pip install leidenalg igraph")

def to_nx_graph(adj_list):
    G = nx.Graph()
    for u, neighbors in adj_list.items():
        G.add_node(u)
        for v in neighbors:
            if u < v:
                G.add_edge(u, v)
    return G

def nx_to_igraph(G):
    # Efficient conversion from NetworkX to igraph
    # We can use the native conversion if available, or build manually
    # igraph.Graph.from_networkx(G) is standard
    return ig.Graph.from_networkx(G)

def run_nx_community_size_experiment(
    sizes: list = None,
    p: float = 0.1,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/community_size_nx.csv"
) -> None:
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000]

    header = [
        "n", "p", "edges", 
        "louvain_time", "louvain_communities", "louvain_modularity",
        "leiden_time", "leiden_communities", "leiden_modularity"
    ]
    rows = []

    print("=== NetworkX + Leidenalg Community Detection Size Experiment ===")
    print(f"Testing sizes: {sizes}")
    print(f"Edge probability: {p}")
    print()

    for i, n in enumerate(sizes):
        print_progress(i, len(sizes), prefix=f"Size {n:>5}")

        adj_list = generate_random_graph(n, p, seed=seed + i)
        G = to_nx_graph(adj_list)
        edges = G.number_of_edges()

        # Louvain (NetworkX)
        try:
            start = time.time()
            communities_louvain = nx.community.louvain_communities(G, seed=seed)
            louvain_time = time.time() - start
            
            louvain_num = len(communities_louvain)
            louvain_mod = nx.community.quality.modularity(G, communities_louvain)
        except AttributeError:
            louvain_time, louvain_num, louvain_mod = 0, 0, 0

        # Leiden (leidenalg + igraph)
        try:
            # Conversion time is arguably part of the "using external lib" cost, 
            # but for pure algo comparison we might want to exclude it.
            # However, since we start with NX graph, we must convert.
            # We'll include conversion in the time to be realistic about "using leidenalg from python"
            start = time.time()
            H = nx_to_igraph(G)
            # leidenalg.find_partition returns a partition object
            part = leidenalg.find_partition(H, leidenalg.ModularityVertexPartition, seed=seed)
            leiden_time = time.time() - start
            
            leiden_num = len(part)
            leiden_mod = part.quality() # Modularity
        except Exception as e:
            print(f"Leiden failed: {e}")
            leiden_time, leiden_num, leiden_mod = 0, 0, 0

        rows.append([
            n, p, edges, 
            louvain_time, louvain_num, louvain_mod,
            leiden_time, leiden_num, leiden_mod
        ])

    print_progress(len(sizes), len(sizes), prefix="Complete")
    print()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    print("\n--- Summary ---")
    print(f"{'n':>6} {'Louvain(s)':>10} {'Leiden(s)':>10} {'Louvain Q':>10} {'Leiden Q':>10} {'Louvain #':>10} {'Leiden #':>10}")
    print("-" * 80)
    for row in rows:
        print(f"{row[0]:>6} {row[3]:>10.4f} {row[6]:>10.4f} {row[5]:>10.4f} {row[8]:>10.4f} {row[4]:>10} {row[7]:>10}")


def run_nx_community_quality_experiment(
    n: int = 200,
    num_communities: int = 4,
    p_intra_values: list = None,
    p_inter: float = 0.01,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/community_quality_nx.csv"
) -> None:
    if p_intra_values is None:
        p_intra_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    header = [
        "n", "num_planted", "p_intra", "p_inter",
        "louvain_time", "louvain_detected", "louvain_modularity", "louvain_nmi",
        "leiden_time", "leiden_detected", "leiden_modularity", "leiden_nmi"
    ]
    rows = []

    print("=== NetworkX + Leidenalg Community Quality Experiment ===")
    print(f"Graph: {n} nodes, {num_communities} planted communities")
    print(f"p_inter: {p_inter}")
    print(f"Testing p_intra: {p_intra_values}")
    print()

    for p_intra in p_intra_values:
        network, ground_truth = generate_community_network(
            n, num_communities, p_intra, p_inter, seed=seed
        )
        G = to_nx_graph(network.adjacency)

        # Louvain
        try:
            start = time.time()
            communities_louvain = nx.community.louvain_communities(G, seed=seed)
            louvain_time = time.time() - start
            
            louvain_num = len(communities_louvain)
            louvain_mod = nx.community.quality.modularity(G, communities_louvain)
            
            louvain_partition = {}
            for idx, comm in enumerate(communities_louvain):
                for node in comm:
                    louvain_partition[node] = idx
            
            louvain_nmi = normalized_mutual_information(ground_truth, louvain_partition)
        except AttributeError:
             louvain_time, louvain_num, louvain_mod, louvain_nmi = 0, 0, 0, 0

        # Leiden
        try:
            start = time.time()
            H = nx_to_igraph(G)
            part = leidenalg.find_partition(H, leidenalg.ModularityVertexPartition, seed=seed)
            leiden_time = time.time() - start
            
            leiden_num = len(part)
            leiden_mod = part.quality()
            
            # Convert igraph partition to dict for NMI
            # part is a list of lists of node indices. 
            # Note: igraph nodes are 0-indexed and map 1-to-1 to NX nodes if we added them in order (0..n-1)
            # Our generator makes nodes 0..n-1.
            leiden_partition = {}
            for idx, comm in enumerate(part):
                for node in comm:
                    # igraph node index matches our node ID
                    leiden_partition[node] = idx
            
            leiden_nmi = normalized_mutual_information(ground_truth, leiden_partition)
        except Exception as e:
            print(f"Leiden failed: {e}")
            leiden_time, leiden_num, leiden_mod, leiden_nmi = 0, 0, 0, 0

        rows.append([
            n, num_communities, p_intra, p_inter,
            louvain_time, louvain_num, louvain_mod, louvain_nmi,
            leiden_time, leiden_num, leiden_mod, leiden_nmi
        ])

        print(f"  p_intra={p_intra:.2f}:")
        print(f"    Louvain: Q={louvain_mod:.4f}, NMI={louvain_nmi:.4f}")
        print(f"    Leiden:  Q={leiden_mod:.4f}, NMI={leiden_nmi:.4f}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"\nResults saved to {out_csv}")


def run_nx_community_density_experiment(
    n: int = 500,
    densities: list = None,
    seed: int = 42,
    out_csv: str = "networkx_demos/results/community_density_nx.csv"
) -> None:
    if densities is None:
        densities = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    header = [
        "n", "p", "edges", 
        "louvain_time", "louvain_communities", "louvain_modularity",
        "leiden_time", "leiden_communities", "leiden_modularity"
    ]
    rows = []

    print("=== NetworkX + Leidenalg Community Density Experiment ===")
    print(f"Graph size: {n} nodes")
    print(f"Testing densities: {densities}")
    print()

    for i, p in enumerate(densities):
        print_progress(i, len(densities), prefix=f"Density {p:.2f}")

        adj_list = generate_random_graph(n, p, seed=seed + i)
        G = to_nx_graph(adj_list)
        edges = G.number_of_edges()

        # Louvain
        try:
            start = time.time()
            communities_louvain = nx.community.louvain_communities(G, seed=seed)
            louvain_time = time.time() - start
            
            louvain_num = len(communities_louvain)
            louvain_mod = nx.community.quality.modularity(G, communities_louvain)
        except AttributeError:
            louvain_time, louvain_num, louvain_mod = 0, 0, 0

        # Leiden
        try:
            start = time.time()
            H = nx_to_igraph(G)
            part = leidenalg.find_partition(H, leidenalg.ModularityVertexPartition, seed=seed)
            leiden_time = time.time() - start
            
            leiden_num = len(part)
            leiden_mod = part.quality()
        except Exception as e:
            leiden_time, leiden_num, leiden_mod = 0, 0, 0

        rows.append([
            n, p, edges, 
            louvain_time, louvain_num, louvain_mod,
            leiden_time, leiden_num, leiden_mod
        ])

    print_progress(len(densities), len(densities), prefix="Complete")
    print()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv(out_csv, header, rows)
    print(f"Results saved to {out_csv}")

    print("\n--- Summary ---")
    print(f"{'p':>6} {'Louvain(s)':>10} {'Leiden(s)':>10} {'Louvain Q':>10} {'Leiden Q':>10} {'Louvain #':>10} {'Leiden #':>10}")
    print("-" * 80)
    for row in rows:
        print(f"{row[1]:>6.2f} {row[3]:>10.4f} {row[6]:>10.4f} {row[5]:>10.4f} {row[8]:>10.4f} {row[4]:>10} {row[7]:>10}")


if __name__ == "__main__":
    run_nx_community_size_experiment()
    print("\n" + "=" * 70 + "\n")
    run_nx_community_quality_experiment()
    print("\n" + "=" * 70 + "\n")
    run_nx_community_density_experiment()
