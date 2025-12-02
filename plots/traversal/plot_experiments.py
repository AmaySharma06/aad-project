"""
Plotting functions for traversal algorithm experiments.
Visualizes BFS, DFS, and Union-Find performance.
"""

import matplotlib.pyplot as plt
import sys
import os
import csv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def read_csv_data(csv_path: str):
    """Read CSV file and return list of dictionaries."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row_dict = {}
            for key, value in row.items():
                try:
                    row_dict[key] = float(value)
                except ValueError:
                    row_dict[key] = value
            data.append(row_dict)
    return data


def plot_size_scaling(csv_path: str, output_path: str = None):
    """
    Plot runtime vs graph size for traversal algorithms.
    
    Args:
        csv_path: Path to CSV file with columns [n, bfs_time, dfs_time, union_find_time]
        output_path: Path to save the plot (optional)
    """
    data = read_csv_data(csv_path)
    
    sizes = [row['n'] for row in data]
    bfs_times = [row['bfs_time'] for row in data]
    dfs_rec_times = [row['dfs_recursive_time'] for row in data]
    dfs_iter_times = [row['dfs_iterative_time'] for row in data]
    uf_times = [row['union_find_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, bfs_times, 'o-', label='BFS', color='#1f77b4', linewidth=2, markersize=6)
    plt.plot(sizes, dfs_rec_times, 's-', label='DFS (Recursive)', color='#2ca02c', linewidth=2, markersize=6)
    plt.plot(sizes, dfs_iter_times, 'd-', label='DFS (Iterative)', color='#98df8a', linewidth=2, markersize=6)
    plt.plot(sizes, uf_times, '^-', label='Union-Find', color='#d62728', linewidth=2, markersize=6)
    
    plt.xlabel('Number of Nodes (n)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Traversal Algorithms: Runtime vs Graph Size (p=0.1)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Size scaling plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_density_scaling(csv_path: str, output_path: str = None):
    """
    Plot runtime vs graph density for traversal algorithms.
    
    Args:
        csv_path: Path to CSV file with columns [p, bfs_time, dfs_time, union_find_time]
        output_path: Path to save the plot (optional)
    """
    data = read_csv_data(csv_path)
    
    densities = [row['p'] for row in data]
    bfs_times = [row['bfs_time'] for row in data]
    dfs_rec_times = [row['dfs_recursive_time'] for row in data]
    dfs_iter_times = [row['dfs_iterative_time'] for row in data]
    uf_times = [row['union_find_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(densities, bfs_times, 'o-', label='BFS', color='#1f77b4', linewidth=2, markersize=6)
    plt.plot(densities, dfs_rec_times, 's-', label='DFS (Recursive)', color='#2ca02c', linewidth=2, markersize=6)
    plt.plot(densities, dfs_iter_times, 'd-', label='DFS (Iterative)', color='#98df8a', linewidth=2, markersize=6)
    plt.plot(densities, uf_times, '^-', label='Union-Find', color='#d62728', linewidth=2, markersize=6)
    
    plt.xlabel('Edge Probability (p)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Traversal Algorithms: Runtime vs Graph Density (n=500)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Density scaling plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Generate plots from experiment results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Input CSV files
    size_csv = os.path.join(project_root, 'experiments', 'traversal', 'results', 'size_results.csv')
    density_csv = os.path.join(project_root, 'experiments', 'traversal', 'results', 'density_results.csv')
    
    # Output directory for plots
    output_dir = os.path.join(project_root, 'report', 'plots', 'traversal')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating traversal plots...")
    print(f"Output directory: {output_dir}")
    
    if os.path.exists(size_csv):
        print(f"\nProcessing {size_csv}")
        plot_size_scaling(size_csv, os.path.join(output_dir, 'size_vs_time.png'))
    else:
        print(f"Warning: {size_csv} not found")
    
    if os.path.exists(density_csv):
        print(f"\nProcessing {density_csv}")
        plot_density_scaling(density_csv, os.path.join(output_dir, 'density_vs_time.png'))
    else:
        print(f"Warning: {density_csv} not found")
    
    print("\nTraversal plots generated successfully!")
