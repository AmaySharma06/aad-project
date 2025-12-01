"""
Plotting functions for centrality algorithm experiments.
Visualizes PageRank performance and convergence.
"""

import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import read_csv


def plot_size_scaling(csv_path: str, output_path: str = None):
    """
    Plot PageRank runtime vs graph size.
    
    Args:
        csv_path: Path to CSV file with columns [size, pagerank_time, iterations]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    sizes = [row['size'] for row in data]
    times = [row['pagerank_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-', color='purple', linewidth=2, markersize=8)
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('PageRank: Runtime vs Graph Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_density_scaling(csv_path: str, output_path: str = None):
    """
    Plot PageRank runtime vs graph density.
    
    Args:
        csv_path: Path to CSV file with columns [density, pagerank_time, iterations]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    densities = [row['density'] for row in data]
    times = [row['pagerank_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(densities, times, 's-', color='purple', linewidth=2, markersize=8)
    
    plt.xlabel('Edge Probability (p)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('PageRank: Runtime vs Graph Density', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_iterations_vs_size(csv_path: str, output_path: str = None):
    """
    Plot number of iterations to convergence vs graph size.
    
    Args:
        csv_path: Path to CSV file with columns [size, pagerank_time, iterations]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    sizes = [row['size'] for row in data]
    iterations = [row.get('iterations', 0) for row in data]
    
    if all(i == 0 for i in iterations):
        print("No iteration data available")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, iterations, '^-', color='orange', linewidth=2, markersize=8)
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Iterations to Convergence', fontsize=12)
    plt.title('PageRank: Convergence vs Graph Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_damping_factor_effect(csv_path: str, output_path: str = None):
    """
    Plot effect of damping factor on PageRank performance.
    
    Args:
        csv_path: Path to CSV file with columns [damping, pagerank_time, iterations]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    dampings = [row['damping'] for row in data]
    times = [row['pagerank_time'] for row in data]
    iterations = [row.get('iterations', 0) for row in data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Runtime plot
    ax1.plot(dampings, times, 'o-', color='purple', linewidth=2, markersize=8)
    ax1.set_xlabel('Damping Factor', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('PageRank: Runtime vs Damping Factor', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Iterations plot
    if not all(i == 0 for i in iterations):
        ax2.plot(dampings, iterations, '^-', color='orange', linewidth=2, markersize=8)
        ax2.set_xlabel('Damping Factor', fontsize=12)
        ax2.set_ylabel('Iterations to Convergence', fontsize=12)
        ax2.set_title('PageRank: Iterations vs Damping Factor', fontsize=14)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No iteration data', ha='center', va='center', fontsize=14)
        ax2.set_axis_off()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pagerank_distribution(ranks: dict, top_k: int = 20, output_path: str = None):
    """
    Plot PageRank score distribution for top-k nodes.
    
    Args:
        ranks: Dictionary mapping node -> PageRank score
        top_k: Number of top nodes to show
        output_path: Path to save the plot (optional)
    """
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:top_k]
    nodes = [str(node) for node, _ in sorted_ranks]
    scores = [score for _, score in sorted_ranks]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(nodes)), scores, color='purple', alpha=0.7, edgecolor='black')
    plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
    
    plt.xlabel('Node', fontsize=12)
    plt.ylabel('PageRank Score', fontsize=12)
    plt.title(f'Top {top_k} Nodes by PageRank Score', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_complexity_analysis(csv_path: str, output_path: str = None):
    """
    Plot PageRank runtime with theoretical complexity overlay.
    
    Args:
        csv_path: Path to CSV file with size scaling data
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    sizes = [row['size'] for row in data]
    times = [row['pagerank_time'] for row in data]
    
    # Theoretical O(k * E) complexity where k is iterations
    # For Erdős-Rényi with p=0.1: E ≈ p * n^2 / 2
    p = 0.1
    k = 20  # Average iterations
    theoretical = [k * p * s * s / 2 for s in sizes]
    
    # Normalize theoretical to match scale
    scale = times[-1] / theoretical[-1] if theoretical[-1] > 0 else 1
    theoretical = [t * scale for t in theoretical]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-', label='PageRank (measured)', color='purple', linewidth=2, markersize=6)
    plt.plot(sizes, theoretical, '--', label='O(k·E) theoretical', color='gray', linewidth=2)
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('PageRank: Measured vs Theoretical Complexity', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage - generate plots from experiment results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'centrality')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    size_csv = os.path.join(results_dir, 'size_scaling.csv')
    density_csv = os.path.join(results_dir, 'density_scaling.csv')
    damping_csv = os.path.join(results_dir, 'damping_effect.csv')
    
    if os.path.exists(size_csv):
        plot_size_scaling(size_csv, os.path.join(output_dir, 'pagerank_size_scaling.png'))
        plot_iterations_vs_size(size_csv, os.path.join(output_dir, 'pagerank_iterations.png'))
        plot_complexity_analysis(size_csv, os.path.join(output_dir, 'pagerank_complexity.png'))
    
    if os.path.exists(density_csv):
        plot_density_scaling(density_csv, os.path.join(output_dir, 'pagerank_density_scaling.png'))
    
    if os.path.exists(damping_csv):
        plot_damping_factor_effect(damping_csv, os.path.join(output_dir, 'pagerank_damping.png'))
    
    print("Centrality plots generated successfully!")
