"""Build embodiment > task > primitive trees from action metadata.

Each trajectory carries labels ``{"embodiment": ..., "task": ..., "primitive": ...}``. We build the
rooted tree ``root -> embodiment -> task -> primitive`` and expose its hop-count metric and
adjacency, which are the ground truth for hierarchy reconstruction (``metrics/distortion.py``)
and a 0-hyperbolic reference for the Gromov estimator tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

LEVELS: tuple[str, ...] = ("embodiment", "task", "primitive")


@dataclass
class Hierarchy:
    """A rooted tree stored as parent pointers.

    Attributes:
        names: node label per index; index ``0`` is the root ``"<root>"``.
        parent: parent index per node; ``-1`` for the root.
        depth: depth per node (root ``0``).
        levels: level name per node (``"root"``, then :data:`LEVELS`).
    """

    names: list[str] = field(default_factory=lambda: ["<root>"])
    parent: list[int] = field(default_factory=lambda: [-1])
    depth: list[int] = field(default_factory=lambda: [0])
    levels: list[str] = field(default_factory=lambda: ["root"])
    _index: dict[tuple[str, ...], int] = field(default_factory=lambda: {(): 0}, repr=False)

    # ------------------------------------------------------------------ construction
    def add_path(self, path: Sequence[str]) -> int:
        """Insert ``path`` (e.g. ``("arm_a", "task_1", "grasp")``); returns the leaf index."""
        node = 0
        for lvl, label in enumerate(path):
            key = tuple(path[: lvl + 1])
            if key not in self._index:
                self.names.append(label)
                self.parent.append(node)
                self.depth.append(lvl + 1)
                self.levels.append(LEVELS[lvl] if lvl < len(LEVELS) else f"level_{lvl}")
                self._index[key] = len(self.names) - 1
            node = self._index[key]
        return node

    def index_of(self, path: Sequence[str]) -> int:
        return self._index[tuple(path)]

    @property
    def n_nodes(self) -> int:
        return len(self.names)

    # ------------------------------------------------------------------ metric structure
    def adjacency(self) -> Tensor:
        """Symmetric boolean ``(n, n)`` edge matrix."""
        n = self.n_nodes
        adj = torch.zeros(n, n, dtype=torch.bool)
        for child, par in enumerate(self.parent):
            if par >= 0:
                adj[child, par] = adj[par, child] = True
        return adj

    def ancestors(self, node: int) -> list[int]:
        """Path from ``node`` up to and including the root."""
        out = [node]
        while self.parent[out[-1]] >= 0:
            out.append(self.parent[out[-1]])
        return out

    def lca_depth(self, a: int, b: int) -> int:
        anc_a = set(self.ancestors(a))
        for node in self.ancestors(b):
            if node in anc_a:
                return self.depth[node]
        raise RuntimeError("nodes do not share a root")  # unreachable for a tree

    def tree_distance_matrix(self) -> Tensor:
        """Hop-count distance ``(n, n)`` via lowest common ancestors; a 0-hyperbolic metric."""
        n = self.n_nodes
        d = torch.zeros(n, n, dtype=torch.float64)
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.depth[i] + self.depth[j] - 2 * self.lca_depth(i, j)
                d[i, j] = d[j, i] = float(dist)
        return d

    def leaves(self) -> list[int]:
        children = set(p for p in self.parent if p >= 0)
        return [i for i in range(self.n_nodes) if i not in children]


def build_hierarchy(metadata: Iterable[Mapping[str, str]], levels: Sequence[str] = LEVELS) -> Hierarchy:
    """Build the tree from per-trajectory metadata dicts.

    Args:
        metadata: iterable of dicts containing every key in ``levels``.
        levels: ordered keys, outermost first. Defaults to embodiment > task > primitive.

    Raises:
        KeyError: if a record is missing a level key.
    """
    tree = Hierarchy()
    for rec in metadata:
        tree.add_path([str(rec[k]) for k in levels])
    return tree


def leaf_index_for_metadata(tree: Hierarchy, metadata: Iterable[Mapping[str, str]], levels: Sequence[str] = LEVELS) -> Tensor:
    """Map each metadata record to its leaf node index (for aligning latents with tree nodes)."""
    return torch.tensor([tree.index_of([str(rec[k]) for k in levels]) for rec in metadata], dtype=torch.long)


__all__ = ["LEVELS", "Hierarchy", "build_hierarchy", "leaf_index_for_metadata"]
