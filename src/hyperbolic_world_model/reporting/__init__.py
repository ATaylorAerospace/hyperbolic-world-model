"""Tables and figures regenerated from ``outputs/``. Nothing here is hand-edited.

``python -m hyperbolic_world_model.reporting.make_report`` (or ``scripts/make_report.sh``) is the
single command that rebuilds every table and curvature-sweep figure; see :mod:`make_report`.
"""

from __future__ import annotations

from hyperbolic_world_model.reporting.make_report import make_report
from hyperbolic_world_model.reporting.tables import (
    best_curvature_table,
    collect_runs,
    dimension_efficiency_table,
    results_table,
    summary_table,
    sweep_table,
    to_markdown,
)

__all__ = [
    "best_curvature_table",
    "collect_runs",
    "dimension_efficiency_table",
    "make_report",
    "results_table",
    "summary_table",
    "sweep_table",
    "to_markdown",
]
