"""
Union-Find (Disjoint Set Union) Data Structure.

Union-Find is an efficient data structure for tracking disjoint sets
and performing two key operations:
- Find: Determine which set an element belongs to
- Union: Merge two sets into one

Applications:
- Connected component tracking
- Kruskal's MST algorithm
- Detecting cycles in graphs
- Network connectivity queries

This implementation uses two optimizations:
1. Path Compression: Flatten tree during Find operations
2. Union by Rank: Attach smaller tree under larger tree's root

These optimizations give nearly O(1) amortized time per operation.
"""

from typing import Dict, List, Optional, Set


class UnionFind:
    """
    Union-Find (Disjoint Set Union) with path compression and union by rank.

    Attributes
    ----------
    parent : dict[int, int]
        parent[x] = parent of x in the forest

    rank : dict[int, int]
        rank[x] = upper bound on height of subtree rooted at x

    num_components : int
        Current number of disjoint components

    Examples
    --------
    >>> uf = UnionFind(5)  # 5 elements: 0, 1, 2, 3, 4
    >>> uf.find(0)
    0
    >>> uf.union(0, 1)
    True
    >>> uf.connected(0, 1)
    True
    >>> uf.num_components
    4

    Notes
    -----
    Time Complexity:
    - Find: O(α(n)) amortized, where α is inverse Ackermann function (≈ O(1))
    - Union: O(α(n)) amortized
    - Connected: O(α(n)) amortized

    Space Complexity: O(n)
    """

    def __init__(self, n: int = 0, elements: List[int] = None):
        """
        Initialize Union-Find structure.

        Parameters
        ----------
        n : int, optional
            If provided, initialize with elements 0 to n-1.

        elements : list[int], optional
            If provided, initialize with these specific elements.
            If both n and elements are given, elements takes precedence.
        """
        if elements is not None:
            nodes = elements
        elif n > 0:
            nodes = list(range(n))
        else:
            nodes = []

        # Each element is its own parent initially (self-loop)
        self.parent: Dict[int, int] = {x: x for x in nodes}

        # Rank starts at 0 for all elements
        self.rank: Dict[int, int] = {x: 0 for x in nodes}

        # Number of disjoint components
        self.num_components = len(nodes)

    def add(self, x: int) -> None:
        """
        Add a new element as its own component.

        Parameters
        ----------
        x : int
            Element to add.

        Notes
        -----
        If x already exists, this is a no-op.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.num_components += 1

    def find(self, x: int) -> int:
        """
        Find the root/representative of the set containing x.

        Uses path compression: makes every node on the path point directly
        to the root, flattening the tree structure.

        Parameters
        ----------
        x : int
            Element to find the root of.

        Returns
        -------
        int
            The root element of the set containing x.

        Raises
        ------
        KeyError
            If x is not in the Union-Find structure.

        Examples
        --------
        >>> uf = UnionFind(5)
        >>> uf.union(0, 1)
        >>> uf.union(1, 2)
        >>> root = uf.find(2)
        >>> uf.find(0) == root
        True

        Notes
        -----
        Path compression makes subsequent finds faster by directly
        linking nodes to the root:

        Before compression:     After compression:
              0                      0
             /                      /|\\
            1           ->         1 2 3
           /
          2
         /
        3
        """
        if x not in self.parent:
            raise KeyError(f"Element {x} not found in Union-Find")

        # Path compression: recursively find root and update parent
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Merge the sets containing x and y.

        Uses union by rank: attaches the tree with lower rank
        under the root of the tree with higher rank.

        Parameters
        ----------
        x : int
            First element.

        y : int
            Second element.

        Returns
        -------
        bool
            True if x and y were in different sets (merge happened).
            False if x and y were already in the same set.

        Examples
        --------
        >>> uf = UnionFind(5)
        >>> uf.union(0, 1)  # Merge sets {0} and {1}
        True
        >>> uf.union(0, 1)  # Already in same set
        False
        >>> uf.num_components
        4

        Notes
        -----
        Union by rank keeps trees balanced:
        - If ranks differ, attach smaller rank tree under larger
        - If ranks are equal, attach either way and increment rank

        This ensures tree height stays O(log n) even without path compression.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        # Already in the same set
        if root_x == root_y:
            return False

        # Union by rank: attach smaller tree under larger tree's root
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y

        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x

        else:
            # Equal ranks: choose one as root and increment its rank
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.num_components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """
        Check if two elements are in the same set.

        Parameters
        ----------
        x : int
            First element.

        y : int
            Second element.

        Returns
        -------
        bool
            True if x and y are in the same set.

        Examples
        --------
        >>> uf = UnionFind(5)
        >>> uf.connected(0, 1)
        False
        >>> uf.union(0, 1)
        >>> uf.connected(0, 1)
        True
        """
        return self.find(x) == self.find(y)

    def get_components(self) -> List[Set[int]]:
        """
        Get all connected components as a list of sets.

        Returns
        -------
        list[set[int]]
            Each set contains elements in one component.

        Examples
        --------
        >>> uf = UnionFind(5)
        >>> uf.union(0, 1)
        >>> uf.union(2, 3)
        >>> components = uf.get_components()
        >>> len(components)
        3  # {0,1}, {2,3}, {4}
        """
        component_map: Dict[int, Set[int]] = {}

        for element in self.parent:
            root = self.find(element)
            if root not in component_map:
                component_map[root] = set()
            component_map[root].add(element)

        return list(component_map.values())

    def component_size(self, x: int) -> int:
        """
        Get the size of the component containing x.

        Parameters
        ----------
        x : int
            Element in the component.

        Returns
        -------
        int
            Number of elements in the component.
        """
        root = self.find(x)
        return sum(1 for elem in self.parent if self.find(elem) == root)

    def __len__(self) -> int:
        """Return total number of elements."""
        return len(self.parent)

    def __contains__(self, x: int) -> bool:
        """Check if element is in the structure."""
        return x in self.parent

    def __repr__(self) -> str:
        return f"UnionFind(elements={len(self)}, components={self.num_components})"


def build_union_find_from_graph(graph: Dict[int, List[int]]) -> UnionFind:
    """
    Build a Union-Find structure from a graph's edges.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    Returns
    -------
    UnionFind
        Union-Find with nodes connected according to graph edges.

    Examples
    --------
    >>> graph = {0: [1], 1: [0, 2], 2: [1], 3: []}
    >>> uf = build_union_find_from_graph(graph)
    >>> uf.connected(0, 2)
    True
    >>> uf.connected(0, 3)
    False
    """
    uf = UnionFind(elements=list(graph.keys()))

    for u in graph:
        for v in graph[u]:
            uf.union(u, v)

    return uf


def count_components_union_find(graph: Dict[int, List[int]]) -> int:
    """
    Count connected components using Union-Find.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    Returns
    -------
    int
        Number of connected components.

    Notes
    -----
    This is an alternative to BFS/DFS component counting.
    Union-Find is especially efficient for dynamic graphs where
    edges are added incrementally.
    """
    uf = build_union_find_from_graph(graph)
    return uf.num_components


if __name__ == "__main__":
    # Demo
    print("=== Union-Find Demo ===\n")

    # Basic usage
    print("--- Basic Operations ---")
    uf = UnionFind(6)  # Elements 0-5
    print(f"Initial: {uf}")
    print(f"Components: {uf.get_components()}")

    print("\nUnion operations:")
    operations = [(0, 1), (2, 3), (0, 2), (4, 5)]
    for x, y in operations:
        result = uf.union(x, y)
        print(f"  union({x}, {y}) -> merged={result}, components={uf.num_components}")

    print(f"\nFinal components: {uf.get_components()}")
    print(f"connected(0, 3) = {uf.connected(0, 3)}")
    print(f"connected(0, 4) = {uf.connected(0, 4)}")

    # Build from graph
    print("\n--- Building from Graph ---")
    graph = {
        0: [1, 2],
        1: [0],
        2: [0],
        3: [4],
        4: [3],
        5: []
    }
    print(f"Graph: {graph}")

    uf_graph = build_union_find_from_graph(graph)
    print(f"Union-Find: {uf_graph}")
    print(f"Components: {uf_graph.get_components()}")

    # Count components
    print(f"\nComponent count: {count_components_union_find(graph)}")

    # Demonstrate path compression
    print("\n--- Path Compression Demo ---")
    uf2 = UnionFind(8)
    # Create a chain: 0-1-2-3-4-5-6-7
    for i in range(7):
        uf2.union(i, i + 1)

    print("After creating chain 0-1-2-3-4-5-6-7:")
    print(f"  parent before find(7): {dict(uf2.parent)}")

    root = uf2.find(7)
    print(f"  find(7) = {root}")
    print(f"  parent after find(7):  {dict(uf2.parent)}")
    print("  (Notice: path compression flattened the tree)")
