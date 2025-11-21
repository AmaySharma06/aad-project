import random
import os
import csv

def get_random_graph(n, p=0.5):
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                matrix[i][j] = 1
                matrix[j][i] = 1
    return matrix

def mat_to_list(matrix):
    adj = {}
    for i, row in enumerate(matrix):
        adj[i] = [j for j, val in enumerate(row) if val == 1]
    return adj

def list_to_mat(adj):
    n = len(adj)
    matrix = [[0] * n for _ in range(n)]
    for node, neighbors in adj.items():
        for neighbor in neighbors:
            matrix[node][neighbor] = 1
    return matrix

def write_csv(filepath, header, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

# Usage
if __name__ == "__main__":
    n = 5
    matrix = get_random_graph(n)
    adj_list = mat_to_list(matrix)
    matrix_back = list_to_mat(adj_list)

    print(matrix)
    print(adj_list)