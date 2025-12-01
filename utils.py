"""
Utility Functions for Experiments and Benchmarking.

Provides:
- CSV file I/O for storing experiment results
- Timing utilities for measuring algorithm performance
- Metrics calculation (precision, recall, etc.)
"""

import csv
import os
import time
from typing import Dict, List, Callable, Any, Tuple
from functools import wraps


# ============================================================================
# CSV Utilities
# ============================================================================

def write_csv(
    filepath: str,
    header: List[str],
    rows: List[List[Any]]
) -> None:
    """
    Write experiment results to a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the output CSV file.

    header : list[str]
        Column names.

    rows : list[list]
        Data rows to write.

    Examples
    --------
    >>> header = ["n", "time", "result"]
    >>> rows = [[100, 0.5, "success"], [200, 1.2, "success"]]
    >>> write_csv("results.csv", header, rows)
    """
    # Ensure directory exists
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def read_csv(filepath: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Read experiment results from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    tuple
        (header, rows)
        - header: list of column names
        - rows: list of dicts, each row as {column: value}

    Examples
    --------
    >>> header, rows = read_csv("results.csv")
    >>> rows[0]["n"]
    "100"
    """
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    return header, rows


def append_csv(filepath: str, row: List[Any]) -> None:
    """
    Append a single row to an existing CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    row : list
        Data row to append.
    """
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ============================================================================
# Timing Utilities
# ============================================================================

class Timer:
    """
    Context manager for timing code blocks.

    Examples
    --------
    >>> with Timer() as t:
    ...     # code to time
    ...     pass
    >>> print(f"Elapsed: {t.elapsed:.4f}s")
    """

    def __init__(self):
        self.elapsed = 0.0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def time_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Time a function call.

    Parameters
    ----------
    func : callable
        Function to time.

    *args, **kwargs
        Arguments to pass to the function.

    Returns
    -------
    tuple
        (result, elapsed_time)

    Examples
    --------
    >>> def slow_function(n):
    ...     return sum(range(n))
    >>> result, elapsed = time_function(slow_function, 1000000)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def timed(func: Callable) -> Callable:
    """
    Decorator to automatically time a function.

    The decorated function returns (result, elapsed_time).

    Examples
    --------
    >>> @timed
    ... def my_algorithm(graph):
    ...     return compute_something(graph)
    >>> result, elapsed = my_algorithm(graph)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


def benchmark(
    func: Callable,
    args_list: List[tuple],
    num_runs: int = 3
) -> List[Dict[str, Any]]:
    """
    Benchmark a function with multiple inputs.

    Parameters
    ----------
    func : callable
        Function to benchmark.

    args_list : list[tuple]
        List of argument tuples to test.

    num_runs : int, optional (default=3)
        Number of runs per input for averaging.

    Returns
    -------
    list[dict]
        List of benchmark results with timing stats.

    Examples
    --------
    >>> def algorithm(n, p):
    ...     return n * p
    >>> results = benchmark(algorithm, [(100, 0.1), (200, 0.2)])
    """
    results = []

    for args in args_list:
        times = []
        result = None

        for _ in range(num_runs):
            start = time.perf_counter()
            result = func(*args)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        results.append({
            "args": args,
            "result": result,
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / len(times),
            "times": times
        })

    return results


# ============================================================================
# Metrics Utilities
# ============================================================================

def precision_at_k(
    predicted: List[Any],
    relevant: set,
    k: int
) -> float:
    """
    Compute Precision@k.

    Precision@k = |predicted[:k] ∩ relevant| / k

    Parameters
    ----------
    predicted : list
        Ranked list of predictions.

    relevant : set
        Set of relevant items.

    k : int
        Number of top predictions to consider.

    Returns
    -------
    float
        Precision@k score in [0, 1].
    """
    if k == 0:
        return 0.0

    top_k = set(predicted[:k])
    hits = len(top_k & relevant)
    return hits / k


def recall_at_k(
    predicted: List[Any],
    relevant: set,
    k: int
) -> float:
    """
    Compute Recall@k.

    Recall@k = |predicted[:k] ∩ relevant| / |relevant|

    Parameters
    ----------
    predicted : list
        Ranked list of predictions.

    relevant : set
        Set of relevant items.

    k : int
        Number of top predictions to consider.

    Returns
    -------
    float
        Recall@k score in [0, 1].
    """
    if len(relevant) == 0:
        return 0.0

    top_k = set(predicted[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    """
    Compute F1 score from precision and recall.

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_reciprocal_rank(
    rankings: List[List[Any]],
    relevant_sets: List[set]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR = (1/|Q|) * Σ (1 / rank_i)

    where rank_i is the rank of the first relevant item for query i.

    Parameters
    ----------
    rankings : list[list]
        List of ranked prediction lists, one per query.

    relevant_sets : list[set]
        List of relevant item sets, one per query.

    Returns
    -------
    float
        MRR score in [0, 1].
    """
    reciprocal_ranks = []

    for ranking, relevant in zip(rankings, relevant_sets):
        for rank, item in enumerate(ranking, start=1):
            if item in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# ============================================================================
# Progress Tracking
# ============================================================================

def progress_bar(
    current: int,
    total: int,
    bar_length: int = 40,
    prefix: str = "Progress"
) -> str:
    """
    Generate a text progress bar.

    Parameters
    ----------
    current : int
        Current progress count.

    total : int
        Total count.

    bar_length : int, optional (default=40)
        Length of the progress bar.

    prefix : str, optional
        Prefix text.

    Returns
    -------
    str
        Formatted progress bar string.

    Examples
    --------
    >>> print(progress_bar(50, 100))
    Progress: [====================                    ] 50.0%
    """
    percent = current / total if total > 0 else 0
    filled = int(bar_length * percent)
    bar = "=" * filled + " " * (bar_length - filled)
    return f"{prefix}: [{bar}] {percent*100:.1f}%"


def print_progress(
    current: int,
    total: int,
    prefix: str = "Progress",
    end: str = "\r"
) -> None:
    """
    Print a progress bar (in-place update).

    Parameters
    ----------
    current : int
        Current progress.

    total : int
        Total count.

    prefix : str, optional
        Prefix text.

    end : str, optional
        Line ending (use "\r" for in-place update, "\n" for new line).
    """
    print(progress_bar(current, total, prefix=prefix), end=end, flush=True)
    if current >= total:
        print()  # New line when complete


if __name__ == "__main__":
    # Demo
    print("=== Utils Demo ===\n")

    # Timer demo
    print("--- Timer Demo ---")
    with Timer() as t:
        total = sum(range(1000000))
    print(f"Sum: {total}, Time: {t.elapsed:.4f}s")

    # time_function demo
    print("\n--- time_function Demo ---")
    result, elapsed = time_function(sum, range(1000000))
    print(f"Sum: {result}, Time: {elapsed:.4f}s")

    # Benchmark demo
    print("\n--- Benchmark Demo ---")

    def test_func(n):
        return sum(range(n))

    results = benchmark(test_func, [(100000,), (500000,), (1000000,)], num_runs=3)
    for r in results:
        print(f"  n={r['args'][0]:>8}: avg={r['avg_time']:.4f}s")

    # Metrics demo
    print("\n--- Metrics Demo ---")
    predicted = [1, 3, 2, 5, 4, 7, 6]
    relevant = {1, 2, 4, 6, 8}

    p5 = precision_at_k(predicted, relevant, k=5)
    r5 = recall_at_k(predicted, relevant, k=5)
    f1 = f1_score(p5, r5)

    print(f"Predicted: {predicted}")
    print(f"Relevant: {relevant}")
    print(f"Precision@5: {p5:.3f}")
    print(f"Recall@5: {r5:.3f}")
    print(f"F1@5: {f1:.3f}")

    # Progress bar demo
    print("\n--- Progress Bar Demo ---")
    for i in range(101):
        if i % 20 == 0:
            print(progress_bar(i, 100))

    # CSV demo
    print("\n--- CSV Demo ---")
    test_file = "/tmp/test_results.csv"
    write_csv(test_file, ["n", "time", "result"], [
        [100, 0.001, "pass"],
        [200, 0.002, "pass"],
        [300, 0.005, "pass"]
    ])
    print(f"Wrote CSV to {test_file}")

    header, rows = read_csv(test_file)
    print(f"Read back: {header}")
    for row in rows:
        print(f"  {row}")
