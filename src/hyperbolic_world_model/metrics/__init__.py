"""Evaluation metrics.

Every distance-based metric here takes a :class:`~hyperbolic_world_model.geometry.Manifold` and
computes distances with ``manifold.dist``. There is deliberately no code path that computes a
Euclidean distance on hyperbolic coordinates: that would silently penalise the curved models.

The four metric families are

* :mod:`geodesic_error`: rollout error versus horizon, the static "nothing moves" baseline and
  the normalised error that is comparable across curvatures;
* :mod:`gromov_hyperbolicity`: Gromov delta of a point cloud (tree-likeness), absolute and
  diameter-normalised;
* :mod:`distortion`: scale-fitted average distortion and mAP of an embedding against a tree;
* :mod:`dimension_efficiency`: metric-versus-dimension curves, their log2 area and the dimension
  needed to reach a threshold.
"""

from __future__ import annotations

from hyperbolic_world_model.metrics.dimension_efficiency import (
    DimensionCurve,
    area_under_curve,
    curves_from_table,
    dimension_curve,
    dimension_to_reach,
    efficiency_table,
)
from hyperbolic_world_model.metrics.distortion import average_distortion, mean_average_precision
from hyperbolic_world_model.metrics.geodesic_error import (
    geodesic_error,
    geodesic_error_per_horizon,
    normalised_geodesic_error,
    normalised_geodesic_error_per_horizon,
    static_baseline_error,
)
from hyperbolic_world_model.metrics.gromov_hyperbolicity import (
    delta_hyperbolicity,
    delta_hyperbolicity_from_distances,
    relative_delta_hyperbolicity,
)

__all__ = [
    "DimensionCurve",
    "area_under_curve",
    "average_distortion",
    "curves_from_table",
    "delta_hyperbolicity",
    "delta_hyperbolicity_from_distances",
    "dimension_curve",
    "dimension_to_reach",
    "efficiency_table",
    "geodesic_error",
    "geodesic_error_per_horizon",
    "mean_average_precision",
    "normalised_geodesic_error",
    "normalised_geodesic_error_per_horizon",
    "relative_delta_hyperbolicity",
    "static_baseline_error",
]
