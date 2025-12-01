"""
Plotting functions for traversal algorithm experiments.
Visualizes BFS, DFS, and Union-Find performance.
"""

import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import read_csv


def plot_size_scaling(csv_path: str, output_path: str = None):
    """
    Plot runtime vs graph size for traversal algorithms.
    
    Args:
        csv_path: Path to CSV file with columns [size, bfs_time, dfs_time, union_find_time]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    sizes = [row['size'] for row in data]
    bfs_times = [row['bfs_time'] for row in data]
    dfs_times = [row['dfs_time'] for row in data]
    uf_times = [row['union_find_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, bfs_times, 'o-', label='BFS', color='blue', linewidth=2, markersize=6)
    plt.plot(sizes, dfs_times, 's-', label='DFS', color='green', linewidth=2, markersize=6)
    plt.plot(sizes, uf_times, '^-', label='Union-Find', color='red', linewidth=2, markersize=6)
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Traversal Algorithms: Runtime vs Graph Size', fontsize=14)
    plt.legend(fontsize=10)
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
    Plot runtime vs graph density for traversal algorithms.
    
    Args:
        csv_path: Path to CSV file with columns [density, bfs_time, dfs_time, union_find_time]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    densities = [row['density'] for row in data]
    bfs_times = [row['bfs_time'] for row in data]
    dfs_times = [row['dfs_time'] for row in data]
    uf_times = [row['union_find_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(densities, bfs_times, 'o-', label='BFS', color='blue', linewidth=2, markersize=6)
    plt.plot(densities, dfs_times, 's-', label='DFS', color='green', linewidth=2, markersize=6)
    plt.plot(densities, uf_times, '^-', label='Union-Find', color='red', linewidth=2, markersize=6)
    
    plt.xlabel('Edge Probability (p)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Traversal Algorithms: Runtime vs Graph Density', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison_bar(csv_path: str, output_path: str = None):
    """
    Create a bar chart comparing average runtimes of traversal algorithms.
    
    Args:
        csv_path: Path to CSV file with timing data
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    # Calculate averages
    avg_bfs = sum(row['bfs_time'] for row in data) / len(data)
    avg_dfs = sum(row['dfs_time'] for row in data) / len(data)
    avg_uf = sum(row['union_find_time'] for row in data) / len(data)
    
    algorithms = ['BFS', 'DFS', 'Union-Find']
    times = [avg_bfs, avg_dfs, avg_uf]
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(algorithms, times, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{time:.4f}s', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Average Time (seconds)', fontsize=12)
    plt.title('Traversal Algorithms: Average Runtime Comparison', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_complexity_analysis(csv_path: str, output_path: str = None):
    """
    Plot runtime with theoretical complexity overlay.
    
    Args:
        csv_path: Path to CSV file with size scaling data
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    sizes = [row['size'] for row in data]
    bfs_times = [row['bfs_time'] for row in data]
    
    # Theoretical O(V + E) complexity
    # For Erdős-Rényi with p=0.1: E ≈ p * n^2 / 2
    p = 0.1
    theoretical = [s + p * s * s / 2 for s in sizes]
    
    # Normalize theoretical to match scale
    scale = bfs_times[-1] / theoretical[-1] if theoretical[-1] > 0 else 1
    theoretical = [t * scale for t in theoretical]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, bfs_times, 'o-', label='BFS (measured)', color='blue', linewidth=2, markersize=6)
    plt.plot(sizes, theoretical, '--', label='O(V + E) theoretical', color='gray', linewidth=2)
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('BFS: Measured vs Theoretical Complexity', fontsize=14)
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
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'traversal')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    size_csv = os.path.join(results_dir, 'size_scaling.csv')
    density_csv = os.path.join(results_dir, 'density_scaling.csv')
    
    if os.path.exists(size_csv):
        plot_size_scaling(size_csv, os.path.join(output_dir, 'traversal_size_scaling.png'))
        plot_comparison_bar(size_csv, os.path.join(output_dir, 'traversal_comparison.png'))
        plot_complexity_analysis(size_csv, os.path.join(output_dir, 'traversal_complexity.png'))
    
    if os.path.exists(density_csv):
        plot_density_scaling(density_csv, os.path.join(output_dir, 'traversal_density_scaling.png'))
    
    print("Traversal plots generated successfully!")
