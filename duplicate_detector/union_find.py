"""
Disjoint Set Union (Union-Find).

Used to merge duplicate image pairs into duplicate groups.
"""

from __future__ import annotations

from collections import defaultdict


class UnionFind:
    """
    High-performance Union-Find implementation.

    Optimizations:
        - Path compression
        - Union by rank
        - Iterative find()
        - Constant-time amortized operations
    """

    __slots__ = (
        "_parent",
        "_rank",
        "_set_count",
    )

    def __init__(self, size: int):
        self._parent = list(range(size))
        self._rank = [0] * size
        self._set_count = size

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def find(self, node: int) -> int:
        """
        Returns the representative element of a set.
        """

        parent = self._parent

        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]

        return node

    def union(self, a: int, b: int) -> bool:
        """
        Merge two sets.

        Returns
        -------
        bool
            True if a merge occurred.
            False if both nodes already belonged to the same set.
        """

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        parent = self._parent
        rank = self._rank

        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b

        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a

        else:
            parent[root_b] = root_a
            rank[root_a] += 1

        self._set_count -= 1

        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def connected(self, a: int, b: int) -> bool:
        """
        Return True if both nodes belong to the same set.
        """
        return self.find(a) == self.find(b)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def components(self) -> list[list[int]]:
        """
        Returns all connected components.

        Components are sorted by descending size.

        Example
        -------
        [
            [1, 5, 8, 9],
            [2, 4],
            [7],
        ]
        """

        groups = defaultdict(list)

        for node in range(len(self._parent)):
            groups[self.find(node)].append(node)

        components = list(groups.values())

        components.sort(
            key=len,
            reverse=True,
        )

        return components

    def non_trivial_components(self) -> list[list[int]]:
        """
        Returns only components containing at least two nodes.

        This is typically what the duplicate detector needs.
        """

        return [
            component
            for component in self.components()
            if len(component) > 1
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Number of elements.
        """
        return len(self._parent)

    @property
    def set_count(self) -> int:
        """
        Current number of disjoint sets.
        """
        return self._set_count