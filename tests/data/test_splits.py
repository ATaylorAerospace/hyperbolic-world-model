"""One implementation of the held-out split, shared by the datasets and the compositional task."""

from __future__ import annotations

import pytest

from hyperbolic_world_model.data import splits
from hyperbolic_world_model.tasks import compositional_generalization as task_mod


def test_task_module_reuses_the_shared_split() -> None:
    assert task_mod.split_by_combination is splits.split_by_combination
    assert task_mod.normalise_holdout is splits.normalise_holdout
    assert task_mod.COMBINATION_KEYS == splits.COMBINATION_KEYS == ("embodiment", "primitive")


def test_split_by_combination_rules() -> None:
    meta = [{"embodiment": e, "primitive": p} for e in ("a", "b") for p in ("x", "y")] * 2
    seen, held = splits.split_by_combination(meta, [["b", "y"]])
    assert held == [3, 7] and seen == [0, 1, 2, 4, 5, 6]
    assert splits.split_by_combination(meta, []) == (list(range(8)), [])
    with pytest.raises(ValueError, match="not present"):
        splits.split_by_combination(meta, [["b", "z"]])
    with pytest.raises(ValueError, match="removes"):
        splits.split_by_combination(meta, [["a", "x"], ["a", "y"]])
    with pytest.raises(ValueError, match="embodiment, primitive"):
        splits.normalise_holdout([["only-one"]])
    custom = [{"robot": r, "skill": s} for r in ("r1", "r2") for s in ("s1", "s2")]
    assert splits.split_by_combination(custom, [["r2", "s1"]], keys=("robot", "skill")) == (
        [0, 1, 3],
        [2],
    )
