"""Performance-versus-latent-dimension curves.

The hyperbolic hypothesis predicts that a curved latent reaches a given rollout error at a lower
dimension than the flat baseline. This module turns per-run metrics into a tidy table and the
area-under-curve summary that ``reporting/`` plots.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# numpy 2.0 renamed ``trapz`` to ``trapezoid`` and 2.4 removed the old name; support both pins.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # noqa: NPY201


@dataclass(frozen=True)
class DimensionCurve:
    """One curve: a geometry label, sorted dimensions and the metric value at each."""

    geometry: str
    dims: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.dims) != len(self.values):
            raise ValueError("dims and values must have the same length")
        if list(self.dims) != sorted(self.dims):
            raise ValueError("dims must be sorted ascending")


def dimension_curve(
    runs: Iterable[dict], metric: str, geometry_key: str = "geometry", dim_key: str = "latent_dim"
) -> pd.DataFrame:
    """Build a long-format table ``(geometry, latent_dim, value)`` from run metric dicts.

    Args:
        runs: iterables of flat dicts such as those written to ``outputs/*/metrics.json``.
            Each must contain ``geometry_key``, ``dim_key`` and ``metric``.
        metric: the metric column to extract.

    Returns:
        DataFrame with one row per run, sorted by geometry then dimension. Multiple runs at the
        same ``(geometry, dim)`` (e.g. seeds) are kept; aggregate with ``groupby`` downstream.
    """
    rows = []
    for run in runs:
        try:
            rows.append(
                {
                    "geometry": run[geometry_key],
                    "latent_dim": int(run[dim_key]),
                    "value": float(run[metric]),
                }
            )
        except KeyError as e:
            raise KeyError(f"run is missing key {e}; have {sorted(run)}") from e
    df = pd.DataFrame(rows, columns=["geometry", "latent_dim", "value"])
    return df.sort_values(["geometry", "latent_dim"]).reset_index(drop=True)


def area_under_curve(curve: DimensionCurve, log_dims: bool = True) -> float:
    """Trapezoidal area under ``value`` vs ``dim`` (optionally ``log2 dim``).

    Lower is better for error metrics. Comparing AUCs across geometries summarises dimension
    efficiency in a single number that does not depend on picking one dimension.
    """
    x = (
        np.log2(np.asarray(curve.dims, dtype=float))
        if log_dims
        else np.asarray(curve.dims, dtype=float)
    )
    y = np.asarray(curve.values, dtype=float)
    if len(x) < 2:
        raise ValueError("need at least two dimensions to compute an area")
    return float(_trapezoid(y, x))


def dimension_to_reach(curve: DimensionCurve, threshold: float) -> int | None:
    """Smallest dimension at which the metric is ``<= threshold``, or ``None`` if never reached."""
    for d, v in zip(curve.dims, curve.values, strict=True):
        if v <= threshold:
            return d
    return None


__all__ = ["DimensionCurve", "area_under_curve", "dimension_curve", "dimension_to_reach"]
