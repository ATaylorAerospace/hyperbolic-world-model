"""Performance-versus-latent-dimension curves.

The hyperbolic hypothesis predicts that a curved latent reaches a given rollout error at a lower
dimension than the flat baseline. This module turns per-run metrics into a tidy table, one
:class:`DimensionCurve` per geometry, and the summaries ``reporting/`` tabulates and plots:

* :func:`area_under_curve`: trapezoidal area under the curve on a ``log2`` dimension axis, so
  doubling the dimension is one unit of x and no single dimension has to be picked;
* :func:`dimension_to_reach`: the smallest dimension at which the metric is at least as good as a
  threshold (typically the Euclidean baseline's value at its largest dimension).

Nothing here computes a distance; the values it consumes were produced in each model's native
geometry by the other metrics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


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
        if len(set(self.dims)) != len(self.dims):
            raise ValueError("dims must be unique; aggregate seeds before building a curve")
        if any(d <= 0 for d in self.dims):
            raise ValueError("dims must be positive")


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
        same ``(geometry, dim)`` (e.g. seeds) are kept; aggregate with ``groupby`` downstream or
        with :func:`curves_from_table`.
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


def curves_from_table(
    table: pd.DataFrame,
    label_col: str = "geometry",
    dim_col: str = "latent_dim",
    value_col: str = "value",
    agg: str = "mean",
) -> list[DimensionCurve]:
    """Aggregate a long table into one :class:`DimensionCurve` per label (seeds averaged).

    Args:
        table: long table such as the output of :func:`dimension_curve` or the report's summary.
        label_col: column identifying a curve (``geometry``, or a combined ``geometry(K=...)``).
        dim_col: dimension column.
        value_col: metric column.
        agg: pandas aggregation applied to repeated ``(label, dim)`` rows.

    Returns:
        Curves sorted by label; labels with a single dimension are still returned (their AUC
        is undefined, see :func:`area_under_curve`).
    """
    if table.empty:
        return []
    grouped = table.groupby([label_col, dim_col], sort=True)[value_col].agg(agg).reset_index()
    curves = []
    for label, grp in grouped.groupby(label_col, sort=True):
        grp = grp.sort_values(dim_col)
        curves.append(
            DimensionCurve(
                geometry=str(label),
                dims=tuple(int(d) for d in grp[dim_col]),
                values=tuple(float(v) for v in grp[value_col]),
            )
        )
    return curves


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
    # Trapezoid rule written out so it runs on NumPy 1.26 (no ``trapezoid``) and 2.x alike.
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2.0))


def dimension_to_reach(
    curve: DimensionCurve, threshold: float, lower_is_better: bool = True
) -> int | None:
    """Smallest dimension at which the metric reaches ``threshold``, or ``None`` if never reached.

    "Reaches" means ``value <= threshold`` for error-like metrics and ``value >= threshold`` when
    ``lower_is_better`` is false (mAP, correlations).
    """
    for d, v in zip(curve.dims, curve.values, strict=True):
        if (v <= threshold) if lower_is_better else (v >= threshold):
            return d
    return None


def efficiency_table(
    curves: Iterable[DimensionCurve],
    threshold: float | None = None,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """One row per curve: ``label, n_dims, auc_log2, best_value, best_dim, dim_to_reach``.

    ``auc_log2`` is ``NaN`` for curves with a single dimension and ``dim_to_reach`` is ``NaN``
    when ``threshold`` is ``None`` or never reached.
    """
    rows = []
    for c in curves:
        vals = np.asarray(c.values, dtype=float)
        best = int(np.argmin(vals) if lower_is_better else np.argmax(vals))
        reach = None if threshold is None else dimension_to_reach(c, threshold, lower_is_better)
        rows.append(
            {
                "label": c.geometry,
                "n_dims": len(c.dims),
                "auc_log2": area_under_curve(c) if len(c.dims) >= 2 else float("nan"),
                "best_value": float(vals[best]),
                "best_dim": int(c.dims[best]),
                "dim_to_reach": float("nan") if reach is None else float(reach),
            }
        )
    cols = ["label", "n_dims", "auc_log2", "best_value", "best_dim", "dim_to_reach"]
    return pd.DataFrame(rows, columns=cols)


__all__ = [
    "DimensionCurve",
    "area_under_curve",
    "curves_from_table",
    "dimension_curve",
    "dimension_to_reach",
    "efficiency_table",
]
