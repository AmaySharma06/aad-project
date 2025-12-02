"""
Plotting functions for recommender system experiments.
Visualizes Jaccard similarity and friend recommendation performance.
"""

import matplotlib.pyplot as plt
import csv
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def read_csv_data(csv_path: str):
    """Read CSV file and return list of dictionaries."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_dict = {}
            for key, value in row.items():
                try:
                    row_dict[key] = float(value)
                except ValueError:
                    row_dict[key] = value
            data.append(row_dict)
    return data


def plot_size_scaling(csv_path: str, output_path: str = None):
    """Plot runtime vs graph size for recommender algorithms."""
    data = read_csv_data(csv_path)
    
    sizes = [row['n'] for row in data]
    jaccard_times = [row['jaccard_all_pairs_time'] for row in data]
    single_rec_times = [row['recommend_single_time'] for row in data]
    all_rec_times = [row['recommend_all_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, jaccard_times, 'o-', label='Jaccard (all pairs)', color='#1f77b4', linewidth=2, markersize=6)
    plt.plot(sizes, single_rec_times, 's-', label='Single User Recommendation', color='#ff7f0e', linewidth=2, markersize=6)
    plt.plot(sizes, all_rec_times, '^-', label='All Users Recommendation', color='#2ca02c', linewidth=2, markersize=6)
    
    plt.xlabel('Number of Nodes (n)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Recommender System: Runtime vs Graph Size (p=0.1)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Size scaling plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_quality_by_k(csv_path: str, output_path: str = None):
    """Plot precision, recall, and hit rate by k value."""
    data = read_csv_data(csv_path)
    
    # Filter for test_fraction = 0.2 (most common scenario)
    data_filtered = [row for row in data if row['test_fraction'] == 0.2]
    
    k_values = [row['k'] for row in data_filtered]
    precision = [row['precision'] for row in data_filtered]
    recall = [row['recall'] for row in data_filtered]
    hit_rate = [row['hit_rate'] for row in data_filtered]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision plot
    axes[0].plot(k_values, precision, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    axes[0].set_xlabel('k (Number of Recommendations)', fontsize=12)
    axes[0].set_ylabel('Precision@k', fontsize=12)
    axes[0].set_title('Precision vs k', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_ylim(0, max(precision) * 1.2)
    
    # Recall plot
    axes[1].plot(k_values, recall, 's-', color='#ff7f0e', linewidth=2, markersize=8)
    axes[1].set_xlabel('k (Number of Recommendations)', fontsize=12)
    axes[1].set_ylabel('Recall@k', fontsize=12)
    axes[1].set_title('Recall vs k', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_ylim(0, max(recall) * 1.2)
    
    # Hit Rate plot
    axes[2].plot(k_values, hit_rate, '^-', color='#2ca02c', linewidth=2, markersize=8)
    axes[2].set_xlabel('k (Number of Recommendations)', fontsize=12)
    axes[2].set_ylabel('Hit Rate@k', fontsize=12)
    axes[2].set_title('Hit Rate vs k', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_ylim(0, max(hit_rate) * 1.2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Quality by k plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_noise_impact(csv_path: str, output_path: str = None):
    """Plot how noise affects recommendation quality."""
    data = read_csv_data(csv_path)
    
    noise_levels = [row['noise_level'] for row in data]
    precision = [row['precision'] for row in data]
    recall = [row['recall'] for row in data]
    hit_rate = [row['hit_rate'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, precision, 'o-', label='Precision@10', color='#1f77b4', linewidth=2, markersize=6)
    plt.plot(noise_levels, recall, 's-', label='Recall@10', color='#ff7f0e', linewidth=2, markersize=6)
    plt.plot(noise_levels, hit_rate, '^-', label='Hit Rate@10', color='#2ca02c', linewidth=2, markersize=6)
    
    plt.xlabel('Noise Level (Fraction of Edges Perturbed)', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Recommendation Quality vs Noise Level (n=500, k=10)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Noise impact plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Generate plots from experiment results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Input CSV files
    size_csv = os.path.join(project_root, 'experiments', 'recommender', 'results', 'size_results.csv')
    quality_csv = os.path.join(project_root, 'experiments', 'recommender', 'results', 'quality_results.csv')
    noise_csv = os.path.join(project_root, 'experiments', 'recommender', 'results', 'noise_results.csv')
    
    # Output directory for plots
    output_dir = os.path.join(project_root, 'report', 'plots', 'recommender')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating recommender plots...")
    print(f"Output directory: {output_dir}")
    
    if os.path.exists(size_csv):
        print(f"\nProcessing {size_csv}")
        plot_size_scaling(size_csv, os.path.join(output_dir, 'size_vs_time.png'))
    else:
        print(f"Warning: {size_csv} not found")
    
    if os.path.exists(quality_csv):
        print(f"\nProcessing {quality_csv}")
        plot_quality_by_k(quality_csv, os.path.join(output_dir, 'quality_vs_k.png'))
    else:
        print(f"Warning: {quality_csv} not found")
    
    if os.path.exists(noise_csv):
        print(f"\nProcessing {noise_csv}")
        plot_noise_impact(noise_csv, os.path.join(output_dir, 'noise_vs_quality.png'))
    else:
        print(f"Warning: {noise_csv} not found")
    
    print("\nRecommender plots generated successfully!")
