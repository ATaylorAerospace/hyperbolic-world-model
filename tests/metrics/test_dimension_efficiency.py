"""Dimension curves, their log2 area and the dimension needed to reach a threshold."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hyperbolic_world_model.metrics import (
    DimensionCurve,
    area_under_curve,
    curves_from_table,
    dimension_curve,
    dimension_to_reach,
    efficiency_table,
)


def _runs() -> list[dict]:
    rows = []
    for geometry, base in (("euclidean", 1.0), ("poincare", 0.5)):
        for seed in (0, 1):
            for dim in (8, 16, 32):
                rows.append(
                    {
                        "geometry": geometry,
                        "latent_dim": dim,
                        "seed": seed,
                        "err": base / math.log2(dim) + 0.01 * seed,
                    }
                )
    return rows


def test_dimension_curve_builds_a_sorted_long_table() -> None:
    df = dimension_curve(reversed(_runs()), metric="err")
    assert list(df.columns) == ["geometry", "latent_dim", "value"]
    assert len(df) == 12
    assert df["geometry"].tolist() == ["euclidean"] * 6 + ["poincare"] * 6
    assert df["latent_dim"].is_monotonic_increasing or True  # within geometry:
    for _, grp in df.groupby("geometry"):
        assert grp["latent_dim"].is_monotonic_increasing
    with pytest.raises(KeyError, match="missing key"):
        dimension_curve([{"geometry": "euclidean", "latent_dim": 8}], metric="err")


def test_dimension_curve_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        DimensionCurve("g", (1, 2), (1.0,))
    with pytest.raises(ValueError, match="sorted"):
        DimensionCurve("g", (4, 2), (1.0, 2.0))
    with pytest.raises(ValueError, match="unique"):
        DimensionCurve("g", (2, 2), (1.0, 2.0))
    with pytest.raises(ValueError, match="positive"):
        DimensionCurve("g", (0, 2), (1.0, 2.0))


def test_curves_from_table_averages_seeds() -> None:
    curves = curves_from_table(dimension_curve(_runs(), metric="err"))
    assert [c.geometry for c in curves] == ["euclidean", "poincare"]
    euc = curves[0]
    assert euc.dims == (8, 16, 32)
    assert euc.values[0] == pytest.approx(1.0 / 3 + 0.005)  # mean over the two seeds
    assert curves_from_table(pd.DataFrame(columns=["geometry", "latent_dim", "value"])) == []


def test_area_under_curve_on_log2_axis() -> None:
    flat = DimensionCurve("flat", (2, 4, 8), (1.0, 1.0, 1.0))
    assert area_under_curve(flat) == pytest.approx(2.0)  # x = 1, 2, 3
    assert area_under_curve(flat, log_dims=False) == pytest.approx(6.0)
    tri = DimensionCurve("tri", (2, 4), (0.0, 2.0))
    assert area_under_curve(tri) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="two dimensions"):
        area_under_curve(DimensionCurve("one", (8,), (1.0,)))


def test_dimension_to_reach_in_both_directions() -> None:
    err = DimensionCurve("err", (8, 16, 32), (0.9, 0.5, 0.2))
    assert dimension_to_reach(err, 0.5) == 16
    assert dimension_to_reach(err, 0.1) is None
    score = DimensionCurve("map", (8, 16, 32), (0.3, 0.6, 0.9))
    assert dimension_to_reach(score, 0.6, lower_is_better=False) == 16
    assert dimension_to_reach(score, 0.95, lower_is_better=False) is None


def test_efficiency_table_summarises_every_curve() -> None:
    curves = curves_from_table(dimension_curve(_runs(), metric="err"))
    euc_at_max = curves[0].values[-1]
    table = efficiency_table(curves, threshold=euc_at_max)
    assert list(table.columns) == [
        "label",
        "n_dims",
        "auc_log2",
        "best_value",
        "best_dim",
        "dim_to_reach",
    ]
    euc, poi = table.iloc[0], table.iloc[1]
    assert euc["label"] == "euclidean" and euc["dim_to_reach"] == 32 and euc["best_dim"] == 32
    # The hyperbolic curve is uniformly better, so it reaches the Euclidean value earlier.
    assert poi["dim_to_reach"] < euc["dim_to_reach"] and poi["auc_log2"] < euc["auc_log2"]
    single = efficiency_table([DimensionCurve("one", (8,), (0.4,))])
    assert np.isnan(single["auc_log2"].iloc[0]) and np.isnan(single["dim_to_reach"].iloc[0])
