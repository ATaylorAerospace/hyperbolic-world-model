"""Evaluation metrics.

Every distance-based metric here takes a :class:`~hyperbolic_world_model.geometry.Manifold` and
computes distances with ``manifold.dist``. There is deliberately no code path that computes a
Euclidean distance on hyperbolic coordinates: that would silently penalise the curved models.
"""

from __future__ import annotations

from hyperbolic_world_model.metrics.dimension_efficiency import (
    DimensionCurve,
    area_under_curve,
    dimension_curve,
)
from hyperbolic_world_model.metrics.distortion import average_distortion, mean_average_precision
from hyperbolic_world_model.metrics.geodesic_error import (
    geodesic_error,
    geodesic_error_per_horizon,
)
from hyperbolic_world_model.metrics.gromov_hyperbolicity import (
    delta_hyperbolicity,
    relative_delta_hyperbolicity,
)

__all__ = [
    "DimensionCurve",
    "area_under_curve",
    "average_distortion",
    "delta_hyperbolicity",
    "dimension_curve",
    "geodesic_error",
    "geodesic_error_per_horizon",
    "mean_average_precision",
    "relative_delta_hyperbolicity",
]
