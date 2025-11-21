import time

from centrality.degree_centrality import degree_centrality
from centrality.harmonic_closeness import harmonic_closeness
from centrality.betweenness_centrality import brandes_betweenness
from centrality.pagerank import pagerank


def run_centrality_experiment(graph):
    """
    Run all four centrality algorithms on a given graph and measure runtimes.

    Parameters
    ----------
    graph : dict
        Adjacency list representation of the graph.

    Returns
    -------
    result : dict
        A single flat dictionary containing:
            - each algorithm's runtime
            - each algorithm's centrality dictionary

        Example:
        {
            "degree_time": 0.0001,
            "degree": {node: score, ...},

            "harmonic_closeness_time": 0.0123,
            "harmonic_closeness": {node: score, ...},

            "betweenness_time": 0.0345,
            "betweenness": {node: score, ...},

            "pagerank_time": 0.0012,
            "pagerank": {node: score, ...}
        }
    """

    result = {}

    # Degree
    start = time.perf_counter()
    result["degree"] = degree_centrality(graph)
    result["degree_time"] = time.perf_counter() - start

    # Harmonic closeness
    start = time.perf_counter()
    result["harmonic_closeness"] = harmonic_closeness(graph)
    result["harmonic_closeness_time"] = time.perf_counter() - start

    # Betweenness
    start = time.perf_counter()
    result["betweenness"] = brandes_betweenness(graph)
    result["betweenness_time"] = time.perf_counter() - start

    # PageRank
    start = time.perf_counter()
    result["pagerank"] = pagerank(graph)
    result["pagerank_time"] = time.perf_counter() - start

    return result
