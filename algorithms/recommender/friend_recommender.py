"""
Friend Recommendation Engine.

Combines multiple signals to suggest potential friends:
1. Jaccard similarity (common neighbors)
2. Personality tag matching
3. Adamic-Adar index
4. Community membership (if available)

This module provides a complete friend recommendation system for
social networks with user attributes.
"""

from typing import Dict, List, Tuple, Set, Optional
from algorithms.recommender.jaccard import (
    jaccard_similarity,
    common_neighbors_count,
    adamic_adar_index
)


class FriendRecommender:
    """
    Friend recommendation engine combining multiple similarity signals.

    This recommender uses a weighted combination of:
    - Graph structure (Jaccard similarity, common neighbors)
    - User attributes (personality tag overlap)
    - Link prediction metrics (Adamic-Adar)

    Attributes
    ----------
    graph : dict[int, list[int]]
        Social network adjacency list.

    tags : dict[int, list[str]]
        Personality tags for each user.

    weights : dict[str, float]
        Weights for combining different similarity metrics.

    Examples
    --------
    >>> from graph.social_network import generate_social_network
    >>> network = generate_social_network(100, p=0.1, seed=42)
    >>> recommender = FriendRecommender(network.adjacency, network.tags)
    >>> recommendations = recommender.recommend(user=0, k=5)
    >>> print(recommendations)  # List of (user_id, score) tuples
    """

    def __init__(
        self,
        graph: Dict[int, List[int]],
        tags: Dict[int, List[str]] = None,
        weights: Dict[str, float] = None
    ):
        """
        Initialize the friend recommender.

        Parameters
        ----------
        graph : dict[int, list[int]]
            Adjacency list representation of the social network.

        tags : dict[int, list[str]], optional
            Personality tags for each user.
            If None, tag-based similarity is not used.

        weights : dict[str, float], optional
            Weights for combining similarity metrics.
            Default: {"jaccard": 0.4, "tags": 0.3, "adamic_adar": 0.3}
        """
        self.graph = graph
        self.tags = tags if tags is not None else {}
        self.weights = weights if weights is not None else {
            "jaccard": 0.4,
            "tags": 0.3,
            "adamic_adar": 0.3
        }

        # Precompute neighbor sets for efficiency
        self._neighbor_sets = {node: set(neighbors) for node, neighbors in graph.items()}

    def tag_similarity(self, u: int, v: int) -> float:
        """
        Compute tag-based similarity between two users.

        Uses Jaccard similarity on personality tags:
            TagSim(u, v) = |tags(u) ∩ tags(v)| / |tags(u) ∪ tags(v)|

        Parameters
        ----------
        u : int
            First user.

        v : int
            Second user.

        Returns
        -------
        float
            Tag similarity score in [0, 1].
        """
        tags_u = set(self.tags.get(u, []))
        tags_v = set(self.tags.get(v, []))

        if not tags_u and not tags_v:
            return 0.0

        intersection = tags_u & tags_v
        union = tags_u | tags_v

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)

    def combined_score(self, u: int, v: int) -> float:
        """
        Compute combined similarity score between two users.

        Combines multiple signals with configurable weights:
            Score(u, v) = w1 * Jaccard(u, v) + w2 * TagSim(u, v) + w3 * AA(u, v)

        Parameters
        ----------
        u : int
            First user.

        v : int
            Second user.

        Returns
        -------
        float
            Combined similarity score.
        """
        # Jaccard similarity (graph structure)
        jac = jaccard_similarity(self.graph, u, v)

        # Tag similarity (user attributes)
        tag_sim = self.tag_similarity(u, v)

        # Adamic-Adar index (weighted common neighbors)
        aa = adamic_adar_index(self.graph, u, v)

        # Normalize Adamic-Adar to [0, 1] range (approximately)
        # AA can be unbounded, so we use a sigmoid-like normalization
        aa_normalized = aa / (1 + aa) if aa > 0 else 0

        # Weighted combination
        score = (
            self.weights.get("jaccard", 0.4) * jac +
            self.weights.get("tags", 0.3) * tag_sim +
            self.weights.get("adamic_adar", 0.3) * aa_normalized
        )

        return score

    def recommend(
        self,
        user: int,
        k: int = 10,
        exclude_friends: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Recommend potential friends for a user.

        Parameters
        ----------
        user : int
            User ID to generate recommendations for.

        k : int, optional (default=10)
            Number of recommendations to return.

        exclude_friends : bool, optional (default=True)
            If True, don't recommend existing friends.

        Returns
        -------
        list[tuple[int, float]]
            List of (user_id, score) tuples, sorted by score descending.

        Examples
        --------
        >>> recommender = FriendRecommender(graph, tags)
        >>> recommendations = recommender.recommend(user=0, k=5)
        >>> for friend_id, score in recommendations:
        ...     print(f"Suggest user {friend_id} (score: {score:.3f})")
        """
        if user not in self.graph:
            return []

        current_friends = self._neighbor_sets.get(user, set())
        candidates = []

        for candidate in self.graph:
            # Skip self
            if candidate == user:
                continue

            # Skip existing friends
            if exclude_friends and candidate in current_friends:
                continue

            # Compute combined score
            score = self.combined_score(user, candidate)
            if score > 0:
                candidates.append((candidate, score))

        # Sort by score descending and return top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def recommend_all(
        self,
        k: int = 10,
        exclude_friends: bool = True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for all users.

        Parameters
        ----------
        k : int, optional (default=10)
            Number of recommendations per user.

        exclude_friends : bool, optional (default=True)
            If True, don't recommend existing friends.

        Returns
        -------
        dict[int, list[tuple[int, float]]]
            Mapping from user ID to list of recommendations.
        """
        return {
            user: self.recommend(user, k=k, exclude_friends=exclude_friends)
            for user in self.graph
        }

    def recommend_all_inverted_index(
        self,
        k: int = 10,
        exclude_friends: bool = True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations using an inverted index optimization.
        
        Instead of iterating all pairs (O(V^2)), we iterate neighbors to find
        common neighbors efficiently.
        
        Complexity: O(V * average_degree^2) which is usually << O(V^2) for sparse graphs.
        """
        recommendations = {}
        nodes = list(self.graph.keys())
        
        # Precompute Tag Inverted Index: tag -> [users]
        tag_index = {}
        for u, user_tags in self.tags.items():
            for tag in user_tags:
                if tag not in tag_index:
                    tag_index[tag] = []
                tag_index[tag].append(u)
        
        for user in nodes:
            # Set of all candidates (users who share a friend OR a tag)
            candidates = set()
            
            user_friends = self._neighbor_sets.get(user, set())
            
            # 1. Candidates from Graph (Friends of Friends)
            for friend in user_friends:
                for friend_of_friend in self.graph.get(friend, []):
                    if friend_of_friend == user: continue
                    if exclude_friends and friend_of_friend in user_friends: continue
                    candidates.add(friend_of_friend)
            
            # 2. Candidates from Tags (Share a tag)
            user_tags = self.tags.get(user, [])
            for tag in user_tags:
                for u_tag in tag_index.get(tag, []):
                    if u_tag == user: continue
                    if exclude_friends and u_tag in user_friends: continue
                    candidates.add(u_tag)
            
            # Calculate full scores for identified candidates
            user_recs = []
            
            for candidate in candidates:
                score = self.combined_score(user, candidate)
                user_recs.append((candidate, score))
            
            user_recs.sort(key=lambda x: x[1], reverse=True)
            recommendations[user] = user_recs[:k]
            
        return recommendations

    def recommend_all_sparse_matrix(
        self,
        k: int = 10,
        exclude_friends: bool = True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations using sparse matrix multiplication logic.
        
        Simulates:
        1. C_graph = A * A (Common neighbors)
        2. C_tags = T * T^T (Shared tags)
        
        Then merges the non-zero entries to find candidates.
        """
        recommendations = {}
        nodes = list(self.graph.keys())
        n = len(nodes)
        
        # Map node IDs to 0..n-1 indices
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        
        # 1. Graph Matrix Multiplication (A * A)
        # A[i] = list of neighbor indices
        A = [[] for _ in range(n)]
        for u, neighbors in self.graph.items():
            u_idx = node_to_idx[u]
            for v in neighbors:
                if v in node_to_idx:
                    A[u_idx].append(node_to_idx[v])
        
        # Candidates set for each user (using a set of indices)
        candidates_matrix = [set() for _ in range(n)]
        
        # A * A
        for i in range(n):
            for neighbor in A[i]: # neighbor is neighbor of i
                for j in A[neighbor]: # j is neighbor of neighbor
                    if i == j: continue
                    candidates_matrix[i].add(j)
                    
        # 2. Tag Matrix Multiplication (T * T^T)
        # T[i] = list of tags for user i
        # But we need T^T (Tag -> Users) to iterate efficiently
        # tag_to_users[tag] = [user_indices]
        tag_to_users = {}
        for u, user_tags in self.tags.items():
            if u not in node_to_idx: continue
            u_idx = node_to_idx[u]
            for tag in user_tags:
                if tag not in tag_to_users:
                    tag_to_users[tag] = []
                tag_to_users[tag].append(u_idx)
                
        # Multiply T * T^T
        # For each user i, iterate their tags, then find other users with those tags
        for i in range(n):
            user = idx_to_node[i]
            for tag in self.tags.get(user, []):
                for j in tag_to_users.get(tag, []):
                    if i == j: continue
                    candidates_matrix[i].add(j)
        
        # Convert back to recommendations
        for i in range(n):
            user = idx_to_node[i]
            user_recs = []
            user_friends = self._neighbor_sets.get(user, set())
            
            for j in candidates_matrix[i]:
                candidate = idx_to_node[j]
                
                if exclude_friends and candidate in user_friends:
                    continue
                
                score = self.combined_score(user, candidate)
                user_recs.append((candidate, score))
                
            user_recs.sort(key=lambda x: x[1], reverse=True)
            recommendations[user] = user_recs[:k]
            
        return recommendations

    def explain_recommendation(
        self,
        user: int,
        candidate: int
    ) -> Dict[str, float]:
        """
        Explain why a candidate was recommended for a user.

        Parameters
        ----------
        user : int
            User ID.

        candidate : int
            Recommended candidate ID.

        Returns
        -------
        dict[str, float]
            Breakdown of similarity scores.
        """
        common = self._neighbor_sets[user] & self._neighbor_sets[candidate]
        shared_tags = set(self.tags.get(user, [])) & set(self.tags.get(candidate, []))

        return {
            "jaccard_similarity": jaccard_similarity(self.graph, user, candidate),
            "tag_similarity": self.tag_similarity(user, candidate),
            "adamic_adar": adamic_adar_index(self.graph, user, candidate),
            "common_friends_count": len(common),
            "common_friends": list(common),
            "shared_tags": list(shared_tags),
            "combined_score": self.combined_score(user, candidate)
        }


def recommend_friends(
    graph: Dict[int, List[int]],
    user: int,
    k: int = 10,
    tags: Dict[int, List[str]] = None
) -> List[Tuple[int, float]]:
    """
    Simple function to recommend friends for a user.

    Convenience wrapper around FriendRecommender class.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Social network adjacency list.

    user : int
        User to generate recommendations for.

    k : int, optional (default=10)
        Number of recommendations.

    tags : dict[int, list[str]], optional
        User personality tags.

    Returns
    -------
    list[tuple[int, float]]
        Recommended users with scores.
    """
    recommender = FriendRecommender(graph, tags)
    return recommender.recommend(user, k=k)


def evaluate_recommendations(
    train_graph: Dict[int, List[int]],
    test_edges: List[Tuple[int, int]],
    tags: Dict[int, List[str]] = None,
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate recommendation quality using held-out edges.

    Metrics:
    - Precision@k: Fraction of recommendations that are in test set
    - Recall@k: Fraction of test edges that appear in recommendations
    - Hit Rate: Fraction of users with at least one correct recommendation

    Parameters
    ----------
    train_graph : dict[int, list[int]]
        Training graph (test edges removed).

    test_edges : list[tuple[int, int]]
        Held-out edges (ground truth).

    tags : dict[int, list[str]], optional
        User personality tags.

    k : int, optional (default=10)
        Number of recommendations per user.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.

    Examples
    --------
    >>> from graph.noise import split_edges_for_testing
    >>> train, test = split_edges_for_testing(graph, test_fraction=0.2)
    >>> metrics = evaluate_recommendations(train, test, k=10)
    >>> print(f"Precision@10: {metrics['precision']:.3f}")
    """
    # Build test edge lookup
    test_set = set()
    for u, v in test_edges:
        test_set.add((min(u, v), max(u, v)))

    # Initialize recommender on training graph
    recommender = FriendRecommender(train_graph, tags)

    # Track metrics
    total_precision = 0.0
    total_recall = 0.0
    num_hits = 0
    num_users = 0

    # For each user that appears in test edges
    test_users = set()
    for u, v in test_edges:
        test_users.add(u)
        test_users.add(v)

    for user in test_users:
        if user not in train_graph:
            continue

        # Get recommendations
        recommendations = recommender.recommend(user, k=k)
        recommended_set = {rec[0] for rec in recommendations}

        # Find which test edges involve this user
        user_test_partners = set()
        for u, v in test_edges:
            if u == user:
                user_test_partners.add(v)
            elif v == user:
                user_test_partners.add(u)

        if not user_test_partners:
            continue

        # Calculate precision and recall for this user
        hits = recommended_set & user_test_partners
        
        precision = len(hits) / len(recommended_set) if recommended_set else 0
        recall = len(hits) / len(user_test_partners) if user_test_partners else 0

        total_precision += precision
        total_recall += recall
        if hits:
            num_hits += 1
        num_users += 1

    # Aggregate metrics
    if num_users == 0:
        return {"precision": 0, "recall": 0, "hit_rate": 0, "num_users": 0}

    return {
        "precision": total_precision / num_users,
        "recall": total_recall / num_users,
        "hit_rate": num_hits / num_users,
        "num_users": num_users
    }


if __name__ == "__main__":
    # Demo
    print("=== Friend Recommender Demo ===\n")

    # Create a sample social network with tags
    graph = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 4],
        3: [1, 5],
        4: [2, 5],
        5: [3, 4, 6],
        6: [5]
    }

    tags = {
        0: ["sports", "music"],
        1: ["sports", "tech"],
        2: ["music", "travel"],
        3: ["tech", "travel"],
        4: ["sports", "travel"],
        5: ["tech", "music"],
        6: ["sports", "tech"]
    }

    print("Social Network:")
    for user, friends in graph.items():
        print(f"  User {user}: friends={friends}, tags={tags[user]}")

    # Create recommender
    recommender = FriendRecommender(graph, tags)

    # Generate recommendations for user 0
    print("\n--- Recommendations for User 0 ---")
    print(f"User 0: friends={graph[0]}, tags={tags[0]}")
    recs = recommender.recommend(user=0, k=5)

    for candidate, score in recs:
        explanation = recommender.explain_recommendation(0, candidate)
        print(f"\n  Recommend User {candidate} (score: {score:.3f})")
        print(f"    Tags: {tags[candidate]}")
        print(f"    Common friends: {explanation['common_friends']}")
        print(f"    Shared tags: {explanation['shared_tags']}")
        print(f"    Jaccard: {explanation['jaccard_similarity']:.3f}")
        print(f"    Tag similarity: {explanation['tag_similarity']:.3f}")

    # Simple function usage
    print("\n--- Simple Usage ---")
    simple_recs = recommend_friends(graph, user=0, k=3, tags=tags)
    print(f"Top 3 recommendations for user 0: {simple_recs}")

    # Evaluation demo
    print("\n--- Evaluation Demo ---")
    # Simulate train/test split by removing some edges
    from graph.noise import split_edges_for_testing
    from graph.graph_utils import copy_graph

    # We need a larger graph for meaningful evaluation
    large_graph = {i: [] for i in range(20)}
    import random
    random.seed(42)
    for i in range(20):
        for j in range(i+1, 20):
            if random.random() < 0.2:
                large_graph[i].append(j)
                large_graph[j].append(i)

    train, test = split_edges_for_testing(large_graph, test_fraction=0.3, seed=42)
    metrics = evaluate_recommendations(train, test, k=5)
    print(f"Test edges: {len(test)}")
    print(f"Precision@5: {metrics['precision']:.3f}")
    print(f"Recall@5: {metrics['recall']:.3f}")
    print(f"Hit Rate: {metrics['hit_rate']:.3f}")

    # Optimization Benchmark
    print("\n--- Optimization Benchmark ---")
    import time
    from graph.graph_generator import generate_random_graph
    import random
    
    # Generate graph
    print("Generating graph with 500 nodes...")
    bench_graph = generate_random_graph(500, p=0.05, seed=42)
    # Add random tags
    bench_tags = {}
    available_tags = ["sports", "music", "tech", "travel", "food", "art", "gaming"]
    for i in range(500):
        bench_tags[i] = random.sample(available_tags, random.randint(1, 3))
        
    bench_rec = FriendRecommender(bench_graph, bench_tags)
    
    print("Running Baseline (O(N^2))...")
    start = time.time()
    res_base = bench_rec.recommend_all(k=5)
    t_base = time.time() - start
    print(f"Baseline time: {t_base:.4f}s")
    
    print("Running Inverted Index...")
    start = time.time()
    res_inv = bench_rec.recommend_all_inverted_index(k=5)
    t_inv = time.time() - start
    print(f"Inverted Index time: {t_inv:.4f}s (Speedup: {t_base/t_inv:.2f}x)")
    
    print("Running Sparse Matrix...")
    start = time.time()
    res_sparse = bench_rec.recommend_all_sparse_matrix(k=5)
    t_sparse = time.time() - start
    print(f"Sparse Matrix time: {t_sparse:.4f}s (Speedup: {t_base/t_sparse:.2f}x)")
    
    # Verify results match
    match = True
    for u in bench_graph:
        base_ids = [x[0] for x in res_base[u]]
        inv_ids = [x[0] for x in res_inv[u]]
        sparse_ids = [x[0] for x in res_sparse[u]]
        
        # Check set equality of top K (order might differ for ties)
        # But scores should be identical.
        # Let's check if lists are identical for now.
        if base_ids != inv_ids or base_ids != sparse_ids:
            match = False
            print(f"Mismatch for User {u}:")
            print(f"  Base:   {base_ids}")
            print(f"  Inv:    {inv_ids}")
            print(f"  Sparse: {sparse_ids}")
            break
            
    print(f"Results match: {match}")
