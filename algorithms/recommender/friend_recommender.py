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
