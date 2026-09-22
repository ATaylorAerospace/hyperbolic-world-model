"""Dimension-efficiency curves and the numpy 1.x/2.x trapezoid compatibility."""

from __future__ import annotations

import numpy as np
import pytest

from hyperbolic_world_model.metrics.dimension_efficiency import (
    DimensionCurve,
    area_under_curve,
    dimension_curve,
    dimension_to_reach,
)


def test_area_under_curve_matches_manual_trapezoid() -> None:
    c = DimensionCurve("poincare", (8, 16, 32), (1.0, 0.5, 0.25))
    x = np.log2([8, 16, 32])
    manual = 0.5 * (1.0 + 0.5) * (x[1] - x[0]) + 0.5 * (0.5 + 0.25) * (x[2] - x[1])
    assert area_under_curve(c) == pytest.approx(manual)
    assert area_under_curve(c, log_dims=False) == pytest.approx(0.5 * 1.5 * 8 + 0.5 * 0.75 * 16)
    with pytest.raises(ValueError):
        area_under_curve(DimensionCurve("e", (8,), (1.0,)))
    with pytest.raises(ValueError):
        DimensionCurve("e", (16, 8), (1.0, 2.0))


def test_dimension_curve_table_and_threshold() -> None:
    runs = [
        {"geometry": "poincare", "latent_dim": 16, "err": 0.4},
        {"geometry": "poincare", "latent_dim": 8, "err": 0.9},
        {"geometry": "euclidean", "latent_dim": 8, "err": 1.2},
    ]
    df = dimension_curve(runs, "err")
    assert list(df["latent_dim"]) == [8, 8, 16] and list(df["geometry"]) == [
        "euclidean",
        "poincare",
        "poincare",
    ]
    with pytest.raises(KeyError):
        dimension_curve([{"geometry": "x"}], "err")
    c = DimensionCurve("poincare", (8, 16), (0.9, 0.4))
    assert dimension_to_reach(c, 0.5) == 16 and dimension_to_reach(c, 0.1) is None
