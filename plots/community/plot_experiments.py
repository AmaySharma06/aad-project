"""
Plotting functions for community detection experiments.
Visualizes Louvain algorithm performance and community quality.
"""

import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import read_csv


def plot_size_scaling(csv_path: str, output_path: str = None):
    """
    Plot Louvain runtime vs graph size.
    
    Args:
        csv_path: Path to CSV file with columns [size, louvain_time, num_communities]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    sizes = [row['size'] for row in data]
    times = [row['louvain_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-', color='darkgreen', linewidth=2, markersize=8)
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Louvain Community Detection: Runtime vs Graph Size', fontsize=14)
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
    Plot Louvain runtime vs graph density.
    
    Args:
        csv_path: Path to CSV file with columns [density, louvain_time, num_communities]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    densities = [row['density'] for row in data]
    times = [row['louvain_time'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(densities, times, 's-', color='darkgreen', linewidth=2, markersize=8)
    
    plt.xlabel('Edge Probability (p)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Louvain Community Detection: Runtime vs Graph Density', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_modularity_vs_size(csv_path: str, output_path: str = None):
    """
    Plot modularity score vs graph size.
    
    Args:
        csv_path: Path to CSV file with columns [size, modularity]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    sizes = [row['size'] for row in data]
    modularity = [row['modularity'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, modularity, 'o-', color='navy', linewidth=2, markersize=8)
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Modularity Score', fontsize=12)
    plt.title('Community Quality: Modularity vs Graph Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_communities_vs_size(csv_path: str, output_path: str = None):
    """
    Plot number of detected communities vs graph size.
    
    Args:
        csv_path: Path to CSV file with columns [size, num_communities]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    sizes = [row['size'] for row in data]
    num_communities = [row['num_communities'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, num_communities, '^-', color='maroon', linewidth=2, markersize=8)
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Number of Communities', fontsize=12)
    plt.title('Community Detection: Number of Communities vs Graph Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_nmi_vs_noise(csv_path: str, output_path: str = None):
    """
    Plot NMI (Normalized Mutual Information) vs noise level.
    
    Args:
        csv_path: Path to CSV file with columns [noise_level, nmi]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    noise_levels = [row['noise_level'] for row in data]
    nmi = [row['nmi'] for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, nmi, 'o-', color='crimson', linewidth=2, markersize=8)
    
    plt.xlabel('Noise Level (% edges modified)', fontsize=12)
    plt.ylabel('Normalized Mutual Information', fontsize=12)
    plt.title('Community Detection Robustness: NMI vs Noise', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_resolution_effect(csv_path: str, output_path: str = None):
    """
    Plot effect of resolution parameter on community detection.
    
    Args:
        csv_path: Path to CSV file with columns [resolution, num_communities, modularity]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    resolutions = [row['resolution'] for row in data]
    num_communities = [row['num_communities'] for row in data]
    modularity = [row['modularity'] for row in data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Number of communities
    ax1.plot(resolutions, num_communities, 'o-', color='maroon', linewidth=2, markersize=8)
    ax1.set_xlabel('Resolution Parameter', fontsize=12)
    ax1.set_ylabel('Number of Communities', fontsize=12)
    ax1.set_title('Communities vs Resolution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Modularity
    ax2.plot(resolutions, modularity, 's-', color='navy', linewidth=2, markersize=8)
    ax2.set_xlabel('Resolution Parameter', fontsize=12)
    ax2.set_ylabel('Modularity Score', fontsize=12)
    ax2.set_title('Modularity vs Resolution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_community_size_distribution(communities: dict, output_path: str = None):
    """
    Plot distribution of community sizes.
    
    Args:
        communities: Dictionary mapping node -> community_id
        output_path: Path to save the plot (optional)
    """
    from collections import Counter
    
    community_sizes = Counter(communities.values())
    sizes = sorted(community_sizes.values(), reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    ax1.bar(range(len(sizes)), sizes, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Community (ranked by size)', fontsize=12)
    ax1.set_ylabel('Number of Nodes', fontsize=12)
    ax1.set_title('Community Size Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Pie chart (top communities)
    top_k = min(10, len(sizes))
    top_sizes = sizes[:top_k]
    other = sum(sizes[top_k:])
    
    if other > 0:
        labels = [f'C{i+1}' for i in range(top_k)] + ['Other']
        pie_sizes = top_sizes + [other]
    else:
        labels = [f'C{i+1}' for i in range(top_k)]
        pie_sizes = top_sizes
    
    colors = plt.cm.tab20(range(len(pie_sizes)))
    ax2.pie(pie_sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Top {top_k} Communities by Size', fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_quality_comparison(csv_path: str, output_path: str = None):
    """
    Create a combined plot comparing modularity and NMI.
    
    Args:
        csv_path: Path to CSV file with columns [size, modularity, nmi]
        output_path: Path to save the plot (optional)
    """
    data = read_csv(csv_path)
    
    sizes = [row['size'] for row in data]
    modularity = [row['modularity'] for row in data]
    nmi = [row.get('nmi', 0) for row in data]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'navy'
    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Modularity', color=color1, fontsize=12)
    line1 = ax1.plot(sizes, modularity, 'o-', color=color1, linewidth=2, markersize=8, label='Modularity')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    if not all(n == 0 for n in nmi):
        ax2 = ax1.twinx()
        color2 = 'crimson'
        ax2.set_ylabel('NMI', color=color2, fontsize=12)
        line2 = ax2.plot(sizes, nmi, 's-', color=color2, linewidth=2, markersize=8, label='NMI')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    plt.title('Community Detection Quality Metrics', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage - generate plots from experiment results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'community')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    size_csv = os.path.join(results_dir, 'size_scaling.csv')
    density_csv = os.path.join(results_dir, 'density_scaling.csv')
    resolution_csv = os.path.join(results_dir, 'resolution_effect.csv')
    noise_csv = os.path.join(results_dir, 'noise_robustness.csv')
    
    if os.path.exists(size_csv):
        plot_size_scaling(size_csv, os.path.join(output_dir, 'louvain_size_scaling.png'))
        plot_modularity_vs_size(size_csv, os.path.join(output_dir, 'modularity_vs_size.png'))
        plot_communities_vs_size(size_csv, os.path.join(output_dir, 'communities_vs_size.png'))
        plot_quality_comparison(size_csv, os.path.join(output_dir, 'quality_comparison.png'))
    
    if os.path.exists(density_csv):
        plot_density_scaling(density_csv, os.path.join(output_dir, 'louvain_density_scaling.png'))
    
    if os.path.exists(resolution_csv):
        plot_resolution_effect(resolution_csv, os.path.join(output_dir, 'resolution_effect.png'))
    
    if os.path.exists(noise_csv):
        plot_nmi_vs_noise(noise_csv, os.path.join(output_dir, 'nmi_vs_noise.png'))
    
    print("Community detection plots generated successfully!")
