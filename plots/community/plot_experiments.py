"""
Plotting functions for community detection experiments.
Visualizes Louvain and Leiden algorithm performance and community quality.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import read_csv


def plot_size_scaling(csv_path: str, output_path: str = None):
    """
    Plot Louvain vs Leiden runtime scaling with graph size.
    
    Parameters
    ----------
    csv_path : str
        Path to size_results.csv
    output_path : str, optional
        Path to save the plot
    """
    _, data = read_csv(csv_path)
    
    sizes = [int(row['n']) for row in data]
    louvain_times = [float(row['louvain_time']) for row in data]
    leiden_times = [float(row['leiden_time']) for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, louvain_times, 'o-', color='darkgreen', linewidth=2, 
             markersize=8, label='Louvain')
    plt.plot(sizes, leiden_times, 's-', color='darkblue', linewidth=2, 
             markersize=8, label='Leiden')
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Community Detection: Runtime vs Graph Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_size_modularity(csv_path: str, output_path: str = None):
    """
    Plot modularity achieved by both algorithms vs graph size.
    
    Parameters
    ----------
    csv_path : str
        Path to size_results.csv
    output_path : str, optional
        Path to save the plot
    """
    _, data = read_csv(csv_path)
    
    sizes = [int(row['n']) for row in data]
    louvain_mod = [float(row['louvain_modularity']) for row in data]
    leiden_mod = [float(row['leiden_modularity']) for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, louvain_mod, 'o-', color='darkgreen', linewidth=2, 
             markersize=8, label='Louvain')
    plt.plot(sizes, leiden_mod, 's-', color='darkblue', linewidth=2, 
             markersize=8, label='Leiden')
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Modularity Score', fontsize=12)
    plt.title('Community Detection: Modularity vs Graph Size', fontsize=14)
    plt.legend(fontsize=11)
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
    Plot runtime vs graph density for both algorithms.
    
    Parameters
    ----------
    csv_path : str
        Path to density_results.csv
    output_path : str, optional
        Path to save the plot
    """
    _, data = read_csv(csv_path)
    
    densities = [float(row['p']) for row in data]
    louvain_times = [float(row['louvain_time']) for row in data]
    leiden_times = [float(row['leiden_time']) for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(densities, louvain_times, 'o-', color='darkgreen', linewidth=2, 
             markersize=8, label='Louvain')
    plt.plot(densities, leiden_times, 's-', color='darkblue', linewidth=2, 
             markersize=8, label='Leiden')
    
    plt.xlabel('Edge Probability (p)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Community Detection: Runtime vs Graph Density', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_quality_nmi(csv_path: str, output_path: str = None):
    """
    Plot NMI (Normalized Mutual Information) vs intra-community density.
    
    Parameters
    ----------
    csv_path : str
        Path to quality_results.csv
    output_path : str, optional
        Path to save the plot
    """
    _, data = read_csv(csv_path)
    
    p_intra = [float(row['p_intra']) for row in data]
    louvain_nmi = [float(row['louvain_nmi']) for row in data]
    leiden_nmi = [float(row['leiden_nmi']) for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_intra, louvain_nmi, 'o-', color='darkgreen', linewidth=2, 
             markersize=8, label='Louvain')
    plt.plot(p_intra, leiden_nmi, 's-', color='darkblue', linewidth=2, 
             markersize=8, label='Leiden')
    
    plt.xlabel('Intra-community Edge Probability', fontsize=12)
    plt.ylabel('Normalized Mutual Information', fontsize=12)
    plt.title('Community Detection Quality: NMI vs Community Strength', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_quality_ari(csv_path: str, output_path: str = None):
    """
    Plot ARI (Adjusted Rand Index) vs intra-community density.
    
    Parameters
    ----------
    csv_path : str
        Path to quality_results.csv
    output_path : str, optional
        Path to save the plot
    """
    _, data = read_csv(csv_path)
    
    p_intra = [float(row['p_intra']) for row in data]
    louvain_ari = [float(row['louvain_ari']) for row in data]
    leiden_ari = [float(row['leiden_ari']) for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_intra, louvain_ari, 'o-', color='darkgreen', linewidth=2, 
             markersize=8, label='Louvain')
    plt.plot(p_intra, leiden_ari, 's-', color='darkblue', linewidth=2, 
             markersize=8, label='Leiden')
    
    plt.xlabel('Intra-community Edge Probability', fontsize=12)
    plt.ylabel('Adjusted Rand Index', fontsize=12)
    plt.title('Community Detection Quality: ARI vs Community Strength', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.05)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_resolution_effect(csv_path: str, output_path: str = None):
    """
    Plot the effect of resolution parameter on detected communities.
    
    Parameters
    ----------
    csv_path : str
        Path to resolution_results.csv
    output_path : str, optional
        Path to save the plot
    """
    _, data = read_csv(csv_path)
    
    # Separate Louvain and Leiden data
    louvain_data = [row for row in data if row['algorithm'] == 'louvain']
    leiden_data = [row for row in data if row['algorithm'] == 'leiden']
    
    resolutions_l = [float(row['resolution']) for row in louvain_data]
    louvain_comms = [int(row['num_communities']) for row in louvain_data]
    louvain_nmi = [float(row['nmi']) for row in louvain_data]
    
    resolutions_d = [float(row['resolution']) for row in leiden_data]
    leiden_comms = [int(row['num_communities']) for row in leiden_data]
    leiden_nmi = [float(row['nmi']) for row in leiden_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Number of communities
    ax1.plot(resolutions_l, louvain_comms, 'o-', color='darkgreen', linewidth=2, 
             markersize=8, label='Louvain')
    ax1.plot(resolutions_d, leiden_comms, 's-', color='darkblue', linewidth=2, 
             markersize=8, label='Leiden')
    ax1.set_xlabel('Resolution Parameter', fontsize=12)
    ax1.set_ylabel('Number of Communities', fontsize=12)
    ax1.set_title('Communities vs Resolution', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # NMI
    ax2.plot(resolutions_l, louvain_nmi, 'o-', color='darkgreen', linewidth=2, 
             markersize=8, label='Louvain')
    ax2.plot(resolutions_d, leiden_nmi, 's-', color='darkblue', linewidth=2, 
             markersize=8, label='Leiden')
    ax2.set_xlabel('Resolution Parameter', fontsize=12)
    ax2.set_ylabel('Normalized Mutual Information', fontsize=12)
    ax2.set_title('NMI vs Resolution', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_algorithm_comparison_bars(csv_path: str, output_path: str = None):
    """
    Create bar chart comparing Louvain and Leiden across trials.
    
    Parameters
    ----------
    csv_path : str
        Path to comparison_results.csv
    output_path : str, optional
        Path to save the plot
    """
    _, data = read_csv(csv_path)
    
    # Separate and aggregate data
    louvain_data = [row for row in data if row['algorithm'] == 'louvain']
    leiden_data = [row for row in data if row['algorithm'] == 'leiden']
    
    # Calculate averages
    metrics = ['time', 'nmi', 'ari']
    metric_names = ['Runtime (s)', 'NMI', 'ARI']
    
    louvain_avgs = [
        np.mean([float(row['time']) for row in louvain_data]),
        np.mean([float(row['nmi']) for row in louvain_data]),
        np.mean([float(row['ari']) for row in louvain_data])
    ]
    leiden_avgs = [
        np.mean([float(row['time']) for row in leiden_data]),
        np.mean([float(row['nmi']) for row in leiden_data]),
        np.mean([float(row['ari']) for row in leiden_data])
    ]
    
    louvain_stds = [
        np.std([float(row['time']) for row in louvain_data]),
        np.std([float(row['nmi']) for row in louvain_data]),
        np.std([float(row['ari']) for row in louvain_data])
    ]
    leiden_stds = [
        np.std([float(row['time']) for row in leiden_data]),
        np.std([float(row['nmi']) for row in leiden_data]),
        np.std([float(row['ari']) for row in leiden_data])
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(2)
    width = 0.5
    
    for i, (ax, name) in enumerate(zip(axes, metric_names)):
        vals = [louvain_avgs[i], leiden_avgs[i]]
        errs = [louvain_stds[i], leiden_stds[i]]
        colors = ['darkgreen', 'darkblue']
        
        bars = ax.bar(x, vals, width, yerr=errs, capsize=5, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(['Louvain', 'Leiden'], fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} Comparison', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
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
    
    Parameters
    ----------
    csv_path : str
        Path to size_results.csv
    output_path : str, optional
        Path to save the plot
    """
    _, data = read_csv(csv_path)
    
    sizes = [int(row['n']) for row in data]
    louvain_comms = [int(row['louvain_communities']) for row in data]
    leiden_comms = [int(row['leiden_communities']) for row in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, louvain_comms, 'o-', color='darkgreen', linewidth=2, 
             markersize=8, label='Louvain')
    plt.plot(sizes, leiden_comms, 's-', color='darkblue', linewidth=2, 
             markersize=8, label='Leiden')
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Number of Communities Detected', fontsize=12)
    plt.title('Communities Detected vs Graph Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_plots():
    """Generate all community detection plots from experiment results."""
    results_dir = "experiments/community/results"
    output_dir = "plots/community/output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Generating Community Detection Plots ===\n")
    
    # Size experiment plots
    size_csv = os.path.join(results_dir, "size_results.csv")
    if os.path.exists(size_csv):
        print("Generating size scaling plots...")
        plot_size_scaling(size_csv, os.path.join(output_dir, "size_scaling.png"))
        plot_size_modularity(size_csv, os.path.join(output_dir, "size_modularity.png"))
        plot_communities_vs_size(size_csv, os.path.join(output_dir, "communities_vs_size.png"))
    else:
        print(f"Warning: {size_csv} not found. Run size experiment first.")
    
    # Density experiment plots
    density_csv = os.path.join(results_dir, "density_results.csv")
    if os.path.exists(density_csv):
        print("Generating density scaling plot...")
        plot_density_scaling(density_csv, os.path.join(output_dir, "density_scaling.png"))
    else:
        print(f"Warning: {density_csv} not found. Run density experiment first.")
    
    # Quality experiment plots
    quality_csv = os.path.join(results_dir, "quality_results.csv")
    if os.path.exists(quality_csv):
        print("Generating quality plots...")
        plot_quality_nmi(quality_csv, os.path.join(output_dir, "quality_nmi.png"))
        plot_quality_ari(quality_csv, os.path.join(output_dir, "quality_ari.png"))
    else:
        print(f"Warning: {quality_csv} not found. Run quality experiment first.")
    
    # Resolution experiment plot
    resolution_csv = os.path.join(results_dir, "resolution_results.csv")
    if os.path.exists(resolution_csv):
        print("Generating resolution effect plot...")
        plot_resolution_effect(resolution_csv, os.path.join(output_dir, "resolution_effect.png"))
    else:
        print(f"Warning: {resolution_csv} not found. Run resolution experiment first.")
    
    # Comparison experiment plot
    comparison_csv = os.path.join(results_dir, "comparison_results.csv")
    if os.path.exists(comparison_csv):
        print("Generating algorithm comparison plot...")
        plot_algorithm_comparison_bars(comparison_csv, os.path.join(output_dir, "algorithm_comparison.png"))
    else:
        print(f"Warning: {comparison_csv} not found. Run comparison experiment first.")
    
    print("\n=== Plot generation complete ===")


if __name__ == "__main__":
    generate_all_plots()
