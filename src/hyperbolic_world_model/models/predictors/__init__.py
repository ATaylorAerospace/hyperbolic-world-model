"""Action-conditioned predictor heads.

Two heads share the :class:`ActionConditionedPredictor` interface. The Euclidean head is the
baseline; the hyperbolic head is generic over any :class:`~hyperbolic_world_model.geometry.Manifold`
(including ``Euclidean``, which is used as a consistency test).
"""

from __future__ import annotations

from hyperbolic_world_model.models.predictors.base import ActionConditionedPredictor
from hyperbolic_world_model.models.predictors.euclidean_head import EuclideanHead
from hyperbolic_world_model.models.predictors.hyperbolic_head import HyperbolicHead

__all__ = ["ActionConditionedPredictor", "EuclideanHead", "HyperbolicHead"]
