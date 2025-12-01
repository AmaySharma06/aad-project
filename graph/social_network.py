"""
Social Network Graph Generator.

Generates synthetic Facebook-like social networks where:
- Nodes represent users
- Edges represent friendships
- Each user has personality tags (sports, music, travel, tech)

This is used for testing friend recommendation algorithms.
"""

import random
from typing import List, Dict, Optional


# Available personality tags
PERSONALITY_TAGS = ["sports", "music", "travel", "tech"]


class SocialNetwork:
    """
    A social network graph with user attributes.

    This class bundles together:
    - An adjacency list representing friendships
    - Node attributes (personality tags for each user)

    Attributes
    ----------
    adjacency : dict[int, list[int]]
        Adjacency list representation. adjacency[u] = list of friends of user u.

    tags : dict[int, list[str]]
        Personality tags for each user. tags[u] = list of tags for user u.

    n : int
        Number of users in the network.

    Example
    -------
    >>> network = SocialNetwork(adjacency={0: [1], 1: [0]}, tags={0: ["sports"], 1: ["music"]})
    >>> network.get_friends(0)
    [1]
    >>> network.get_tags(0)
    ["sports"]
    """

    def __init__(self, adjacency: dict, tags: dict):
        """
        Initialize a social network.

        Parameters
        ----------
        adjacency : dict[int, list[int]]
            Friendship adjacency list.

        tags : dict[int, list[str]]
            Personality tags for each user.
        """
        self.adjacency = adjacency
        self.tags = tags
        self.n = len(adjacency)

    def get_friends(self, user: int) -> List[int]:
        """Return list of friends for a user."""
        return self.adjacency.get(user, [])

    def get_tags(self, user: int) -> List[str]:
        """Return personality tags for a user."""
        return self.tags.get(user, [])

    def get_all_users(self) -> List[int]:
        """Return list of all user IDs."""
        return list(self.adjacency.keys())

    def num_users(self) -> int:
        """Return the number of users."""
        return self.n

    def num_friendships(self) -> int:
        """Return the number of friendships (edges)."""
        return sum(len(friends) for friends in self.adjacency.values()) // 2

    def are_friends(self, u: int, v: int) -> bool:
        """Check if two users are friends."""
        return v in self.adjacency.get(u, [])

    def common_friends(self, u: int, v: int) -> List[int]:
        """Return list of common friends between two users."""
        friends_u = set(self.adjacency.get(u, []))
        friends_v = set(self.adjacency.get(v, []))
        return list(friends_u & friends_v)

    def shared_tags(self, u: int, v: int) -> List[str]:
        """Return list of shared personality tags between two users."""
        tags_u = set(self.tags.get(u, []))
        tags_v = set(self.tags.get(v, []))
        return list(tags_u & tags_v)

    def to_dict(self) -> dict:
        """
        Convert to dictionary format.
        
        Returns
        -------
        dict
            {"adjacency": {...}, "tags": {...}}
        """
        return {
            "adjacency": self.adjacency,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SocialNetwork":
        """
        Create SocialNetwork from dictionary.
        
        Parameters
        ----------
        data : dict
            {"adjacency": {...}, "tags": {...}}
        """
        return cls(adjacency=data["adjacency"], tags=data["tags"])

    def __repr__(self):
        return f"SocialNetwork(users={self.n}, friendships={self.num_friendships()})"


def generate_social_network(
    n: int,
    p: float = 0.1,
    tags_per_user: int = 2,
    available_tags: List[str] = None,
    seed: int = None
) -> SocialNetwork:
    """
    Generate a synthetic social network with personality tags.

    Creates a random friendship graph where each user is assigned
    random personality tags from a predefined set.

    Parameters
    ----------
    n : int
        Number of users in the network.

    p : float, optional (default=0.1)
        Probability of friendship between any two users.
        Social networks are typically sparse, so p should be small.

    tags_per_user : int, optional (default=2)
        Number of personality tags to assign to each user.

    available_tags : list[str], optional
        List of available personality tags.
        Default: ["sports", "music", "travel", "tech"]

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SocialNetwork
        A social network object with adjacency list and user tags.

    Examples
    --------
    >>> network = generate_social_network(100, p=0.1, seed=42)
    >>> print(network)
    SocialNetwork(users=100, friendships=...)
    >>> network.get_tags(0)
    ['sports', 'music']  # example output

    Notes
    -----
    The generated network uses the Erdős-Rényi G(n, p) model for friendships.
    Real social networks follow power-law degree distributions, but G(n, p)
    is simpler and sufficient for testing algorithms.
    """
    if seed is not None:
        random.seed(seed)

    if available_tags is None:
        available_tags = PERSONALITY_TAGS.copy()

    # Generate friendship graph (Erdős-Rényi model)
    adjacency = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # Assign random personality tags to each user
    tags = {}
    for user in range(n):
        # Sample tags_per_user tags without replacement
        num_tags = min(tags_per_user, len(available_tags))
        user_tags = random.sample(available_tags, num_tags)
        tags[user] = user_tags

    return SocialNetwork(adjacency=adjacency, tags=tags)


def generate_community_network(
    n: int,
    num_communities: int = 4,
    p_intra: float = 0.3,
    p_inter: float = 0.01,
    seed: int = None
) -> tuple:
    """
    Generate a social network with planted community structure.

    Uses the Stochastic Block Model (SBM) where:
    - Nodes are divided into communities
    - Intra-community edges have higher probability (p_intra)
    - Inter-community edges have lower probability (p_inter)

    Parameters
    ----------
    n : int
        Total number of nodes.

    num_communities : int, optional (default=4)
        Number of communities to create.

    p_intra : float, optional (default=0.3)
        Edge probability within the same community.

    p_inter : float, optional (default=0.01)
        Edge probability between different communities.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (SocialNetwork, ground_truth_communities)
        where ground_truth_communities is a dict mapping user -> community_id

    Notes
    -----
    The Stochastic Block Model is useful for testing community detection
    algorithms because we know the true community structure.
    """
    if seed is not None:
        random.seed(seed)

    # Assign nodes to communities (roughly equal sizes)
    community_size = n // num_communities
    ground_truth = {}
    
    for i in range(n):
        community = min(i // community_size, num_communities - 1)
        ground_truth[i] = community

    # Assign tags based on community (similar users in same community)
    community_tags = {
        0: ["sports", "travel"],
        1: ["music", "tech"],
        2: ["sports", "music"],
        3: ["travel", "tech"],
    }
    
    # Extend if more communities needed
    while len(community_tags) < num_communities:
        idx = len(community_tags)
        community_tags[idx] = random.sample(PERSONALITY_TAGS, 2)

    tags = {}
    for user, comm in ground_truth.items():
        tags[user] = community_tags[comm].copy()

    # Generate edges based on community membership
    adjacency = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if ground_truth[i] == ground_truth[j]:
                # Same community: use p_intra
                prob = p_intra
            else:
                # Different communities: use p_inter
                prob = p_inter

            if random.random() < prob:
                adjacency[i].append(j)
                adjacency[j].append(i)

    network = SocialNetwork(adjacency=adjacency, tags=tags)
    return network, ground_truth


if __name__ == "__main__":
    # Demo: Generate a social network
    print("=== Social Network Demo ===\n")
    
    network = generate_social_network(10, p=0.3, seed=42)
    print(f"Generated: {network}")
    print(f"\nUser 0 friends: {network.get_friends(0)}")
    print(f"User 0 tags: {network.get_tags(0)}")
    
    # Show some relationships
    for u in range(min(3, network.n)):
        print(f"\nUser {u}:")
        print(f"  Tags: {network.get_tags(u)}")
        print(f"  Friends: {network.get_friends(u)}")

    print("\n=== Community Network Demo ===\n")
    
    comm_network, truth = generate_community_network(20, num_communities=4, seed=42)
    print(f"Generated: {comm_network}")
    print(f"\nGround truth communities:")
    for comm_id in range(4):
        members = [u for u, c in truth.items() if c == comm_id]
        print(f"  Community {comm_id}: {members}")
