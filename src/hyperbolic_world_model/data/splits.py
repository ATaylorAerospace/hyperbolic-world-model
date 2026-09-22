"""Held-out (embodiment, primitive) splits shared by the datasets and the compositional task.

One implementation of the safety rules so the Cosmos loader and the metadata-only path can never
drift apart: every held-out combination must occur in the data, and holding it out must not remove
an embodiment or a primitive entirely (that would test extrapolation, not composition).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

COMBINATION_KEYS: tuple[str, str] = ("embodiment", "primitive")


def normalise_holdout(holdout: Iterable[Sequence[str]] | None) -> tuple[tuple[str, str], ...]:
    """``[[embodiment, primitive], ...]`` (lists, tuples or Hydra containers) -> tuple of pairs."""
    pairs = []
    for combo in holdout or ():
        combo = tuple(str(x) for x in combo)
        if len(combo) != 2:
            raise ValueError(
                f"each held-out combination must be [embodiment, primitive], got {combo}"
            )
        pairs.append(combo)
    return tuple(pairs)


def split_by_combination(
    metadata: Sequence[Mapping[str, str]],
    holdout: Iterable[Sequence[str]],
    keys: tuple[str, str] = COMBINATION_KEYS,
) -> tuple[list[int], list[int]]:
    """``(seen, held_out)`` item indices from per-item metadata.

    Raises:
        ValueError: if a held-out combination never occurs, or if holding it out removes an
            embodiment or a primitive entirely.
    """
    held = set(normalise_holdout(holdout))
    combos = [(str(rec[keys[0]]), str(rec[keys[1]])) for rec in metadata]
    present = set(combos)
    missing = held - present
    if missing:
        raise ValueError(f"held-out combinations not present in the data: {sorted(missing)}")
    seen_combos = present - held
    if held and (
        {e for e, _ in seen_combos} != {e for e, _ in present}
        or {p for _, p in seen_combos} != {p for _, p in present}
    ):
        raise ValueError(
            "holding out these combinations removes an embodiment or primitive entirely"
        )
    seen, out = [], []
    for i, combo in enumerate(combos):
        (out if combo in held else seen).append(i)
    return seen, out


__all__ = ["COMBINATION_KEYS", "normalise_holdout", "split_by_combination"]
