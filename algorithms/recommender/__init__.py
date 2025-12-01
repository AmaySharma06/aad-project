"""
Recommender system algorithms for friend suggestions.
"""

from algorithms.recommender.jaccard import (
    jaccard_similarity,
    jaccard_similarity_all_pairs,
    common_neighbors,
    adamic_adar_index
)
from algorithms.recommender.friend_recommender import (
    FriendRecommender,
    recommend_friends,
    evaluate_recommendations
)

__all__ = [
    "jaccard_similarity",
    "jaccard_similarity_all_pairs",
    "common_neighbors",
    "adamic_adar_index",
    "FriendRecommender",
    "recommend_friends",
    "evaluate_recommendations",
]
