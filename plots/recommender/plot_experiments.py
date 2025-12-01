"""
Plotting functions for recommender system experiments.
Visualizes Jaccard similarity and friend recommendation performance.
"""

import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import read_csv


def plot_size_scaling(csv_path: str, output_path: str = None):
    """
    Plot runtime vs graph size for similarity algorithms.
    
    Args:
        csv_path: Path to CSV file with columns [size, jaccard_time, recommender_time]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    sizes = [row['size'] for row in data]
    jaccard_times = [row['jaccard_time'] for row in data]
    recommender_times = [row['recommender_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, jaccard_times, 'o-', label='Jaccard (all pairs)', color='teal', linewidth=2, markersize=6)
    plt.plot(sizes, recommender_times, 's-', label='Friend Recommender', color='coral', linewidth=2, markersize=6)
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Recommender System: Runtime vs Graph Size', fontsize=14)
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
    Plot runtime vs graph density for similarity algorithms.
    
    Args:
        csv_path: Path to CSV file with columns [density, jaccard_time, recommender_time]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    densities = [row['density'] for row in data]
    jaccard_times = [row['jaccard_time'] for row in data]
    recommender_times = [row['recommender_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(densities, jaccard_times, 'o-', label='Jaccard (all pairs)', color='teal', linewidth=2, markersize=6)
    plt.plot(densities, recommender_times, 's-', label='Friend Recommender', color='coral', linewidth=2, markersize=6)
    
    plt.xlabel('Edge Probability (p)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Recommender System: Runtime vs Graph Density', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_quality_metrics(csv_path: str, output_path: str = None):
    """
    Plot recommendation quality metrics (precision, recall, MAP).
    
    Args:
        csv_path: Path to CSV file with columns [k, precision, recall, map]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    k_values = [row['k'] for row in data]
    precision = [row['precision'] for row in data]
    recall = [row['recall'] for row in data]
    map_scores = [row.get('map', 0) for row in data]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision plot
    axes[0].plot(k_values, precision, 'o-', color='blue', linewidth=2, markersize=8)
    axes[0].set_xlabel('k (top-k recommendations)', fontsize=12)
    axes[0].set_ylabel('Precision@k', fontsize=12)
    axes[0].set_title('Precision vs k', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    
    # Recall plot
    axes[1].plot(k_values, recall, 's-', color='green', linewidth=2, markersize=8)
    axes[1].set_xlabel('k (top-k recommendations)', fontsize=12)
    axes[1].set_ylabel('Recall@k', fontsize=12)
    axes[1].set_title('Recall vs k', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    
    # MAP plot
    if not all(m == 0 for m in map_scores):
        axes[2].plot(k_values, map_scores, '^-', color='red', linewidth=2, markersize=8)
        axes[2].set_xlabel('k (top-k recommendations)', fontsize=12)
        axes[2].set_ylabel('MAP@k', fontsize=12)
        axes[2].set_title('Mean Average Precision vs k', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1.05)
    else:
        axes[2].text(0.5, 0.5, 'No MAP data', ha='center', va='center', fontsize=14)
        axes[2].set_axis_off()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(csv_path: str, output_path: str = None):
    """
    Plot precision-recall curve.
    
    Args:
        csv_path: Path to CSV file with columns [k, precision, recall]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    precision = [row['precision'] for row in data]
    recall = [row['recall'] for row in data]
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, 'o-', color='purple', linewidth=2, markersize=8)
    
    # Add k labels
    for i, row in enumerate(data):
        plt.annotate(f"k={int(row['k'])}", (recall[i], precision[i]), 
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_similarity_heatmap(similarity_matrix: dict, nodes: list = None, output_path: str = None):
    """
    Plot a heatmap of pairwise Jaccard similarities.
    
    Args:
        similarity_matrix: Dictionary of {(u, v): similarity}
        nodes: List of nodes to include (optional, uses all if None)
        output_path: Path to save the plot (optional)
    """
    import numpy as np
    
    if nodes is None:
        all_nodes = set()
        for u, v in similarity_matrix.keys():
            all_nodes.add(u)
            all_nodes.add(v)
        nodes = sorted(all_nodes)[:20]  # Limit to 20 for readability
    
    n = len(nodes)
    matrix = np.zeros((n, n))
    
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = similarity_matrix.get((u, v), similarity_matrix.get((v, u), 0))
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Jaccard Similarity')
    
    plt.xticks(range(n), nodes, rotation=45, ha='right')
    plt.yticks(range(n), nodes)
    
    plt.xlabel('Node', fontsize=12)
    plt.ylabel('Node', fontsize=12)
    plt.title('Jaccard Similarity Heatmap', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_noise_robustness(csv_path: str, output_path: str = None):
    """
    Plot recommendation quality under different noise levels.
    
    Args:
        csv_path: Path to CSV file with columns [noise_level, precision, recall]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    noise_levels = [row['noise_level'] for row in data]
    precision = [row['precision'] for row in data]
    recall = [row['recall'] for row in data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(noise_levels, precision, 'o-', label='Precision', color='blue', linewidth=2, markersize=8)
    ax.plot(noise_levels, recall, 's-', label='Recall', color='green', linewidth=2, markersize=8)
    
    ax.set_xlabel('Noise Level (% edges modified)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Recommendation Quality vs Noise Level', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage - generate plots from experiment results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'recommender')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    size_csv = os.path.join(results_dir, 'size_scaling.csv')
    density_csv = os.path.join(results_dir, 'density_scaling.csv')
    quality_csv = os.path.join(results_dir, 'quality_metrics.csv')
    noise_csv = os.path.join(results_dir, 'noise_robustness.csv')
    
    if os.path.exists(size_csv):
        plot_size_scaling(size_csv, os.path.join(output_dir, 'recommender_size_scaling.png'))
    
    if os.path.exists(density_csv):
        plot_density_scaling(density_csv, os.path.join(output_dir, 'recommender_density_scaling.png'))
    
    if os.path.exists(quality_csv):
        plot_quality_metrics(quality_csv, os.path.join(output_dir, 'recommender_quality.png'))
        plot_precision_recall_curve(quality_csv, os.path.join(output_dir, 'precision_recall_curve.png'))
    
    if os.path.exists(noise_csv):
        plot_noise_robustness(noise_csv, os.path.join(output_dir, 'recommender_noise.png'))
    
    print("Recommender plots generated successfully!")
