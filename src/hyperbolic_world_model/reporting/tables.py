"""Collect ``outputs/**/metrics.json`` into tidy tables.

Every row carries ``geometry`` and ``curvature`` so no number is ever shown without the space it
was computed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def collect_runs(outputs_root: Path) -> list[dict]:
    """Load every ``metrics.json`` below ``outputs_root`` (recursively)."""
    runs = []
    for path in sorted(Path(outputs_root).rglob("metrics.json")):
        payload = json.loads(path.read_text())
        payload["_path"] = str(path.parent)
        runs.append(payload)
    return runs


def results_table(runs: list[dict]) -> pd.DataFrame:
    """Long table: one row per (run, task, metric)."""
    rows = []
    for run in runs:
        for task, res in run.get("tasks", {}).items():
            for metric, value in res.get("metrics", {}).items():
                rows.append(
                    {
                        "experiment": run.get("experiment"),
                        "model": run.get("model"),
                        "geometry": run.get("geometry"),
                        "curvature": run.get("curvature"),
                        "latent_dim": run.get("latent_dim"),
                        "task": task,
                        "metric": metric,
                        "value": value,
                        "path": run["_path"],
                    }
                )
    cols = [
        "experiment",
        "model",
        "geometry",
        "curvature",
        "latent_dim",
        "task",
        "metric",
        "value",
        "path",
    ]
    return pd.DataFrame(rows, columns=cols)


def to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table without extra dependencies."""
    if df.empty:
        return "_(no runs found)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(float_fmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = ["collect_runs", "results_table", "to_markdown"]
