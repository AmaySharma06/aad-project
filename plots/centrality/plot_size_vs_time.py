import csv
import matplotlib.pyplot as plt
import os

RESULTS_PATH = "experiments/centrality/results/size_results.csv"
OUT_PATH = "plots/centrality/images/size_vs_time.png"


def plot_size_vs_time(csv_path=RESULTS_PATH, out_path=OUT_PATH):
    n_vals = []
    degree = []
    harmonic = []
    bet = []
    pr = []

    # Read CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_vals.append(int(row["n"]))
            degree.append(float(row["degree_time"]))
            harmonic.append(float(row["harmonic_closeness_time"]))
            bet.append(float(row["betweenness_time"]))
            pr.append(float(row["pagerank_time"]))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, degree, marker='o', label="Degree Centrality")
    plt.plot(n_vals, harmonic, marker='o', label="Harmonic Closeness")
    plt.plot(n_vals, bet, marker='o', label="Betweenness")
    plt.plot(n_vals, pr, marker='o', label="PageRank")

    plt.xlabel("Number of nodes (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Centrality Runtime vs Graph Size")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    print(f"Saved size-vs-time plot to {out_path}")


if __name__ == "__main__":
    plot_size_vs_time()
