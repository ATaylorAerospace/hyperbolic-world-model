"""Collect ``outputs/**/metrics.json`` into tidy tables.

Every row carries ``geometry`` and ``curvature`` so no number is ever shown without the space it
was computed in. The tables are pure functions of the run payloads and are sorted on every key,
so regenerating a report from the same ``outputs/`` gives byte-identical Markdown.

Tables:

* :func:`results_table`: one row per ``(run, task, metric)``, the long form everything else is
  derived from;
* :func:`summary_table`: mean, std and count over seeds per configuration;
* :func:`sweep_table`: one metric as a ``(geometry, curvature) x latent_dim`` grid of
  ``mean ± std`` cells, the "full curve" the methodology requires;
* :func:`best_curvature_table`: per latent dimension, the Euclidean baseline against the best
  swept curvature, with the methodology's one-std rule applied;
* :func:`dimension_efficiency_table`: log2 area under the metric-vs-dimension curve per geometry
  and curvature, and the dimension at which each reaches the Euclidean baseline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from hyperbolic_world_model.metrics.dimension_efficiency import (
    curves_from_table,
    efficiency_table,
)

RUN_KEYS = ["experiment", "model", "geometry", "curvature", "latent_dim"]
CONFIG_KEYS = [*RUN_KEYS, "task", "metric"]
INT_COLS = frozenset(
    {
        "latent_dim",
        "seed",
        "n_seeds",
        "n_dims",
        "best_dim",
        "dim_to_reach",
        "horizon",
        "n_pairs",
        "n_nodes",
        "depth",
    }
)

#: Metrics where a larger value is better; everything else is treated as an error (lower better).
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {
        "map",
        "depth_spearman",
        "divergence_spearman",
        "divergence_spearman_pixels",
        "embedded_divergence_spearman",
        "saturation_horizon",
        "saturation_fraction_of_horizon",
        "latent_divergence_growth",
    }
)

#: Bookkeeping scalars that are not results and are left out of sweep tables and figures.
COUNT_METRICS: frozenset[str] = frozenset(
    {
        "n_items",
        "n_nodes",
        "n_leaves",
        "n_trajectories",
        "n_bootstrap",
        "n_pairs",
        "n_seen",
        "n_unseen",
        "n_holdout_combinations",
        "horizon",
    }
)


def lower_is_better(metric: str) -> bool:
    """Direction of a metric: false for the ranking/correlation metrics, true for errors."""
    return metric not in HIGHER_IS_BETTER


def is_result_metric(metric: str) -> bool:
    """Whether ``metric`` is a result (as opposed to a count) worth a table row or a figure."""
    return metric not in COUNT_METRICS and not metric.endswith("_std")


def collect_runs(outputs_root: Path) -> list[dict]:
    """Load every ``metrics.json`` below ``outputs_root`` (recursively, sorted by path)."""
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
                        "seed": run.get("seed"),
                        "task": task,
                        "metric": metric,
                        "value": value,
                        "path": run["_path"],
                    }
                )
    cols = [*RUN_KEYS[:3], "curvature", "latent_dim", "seed", "task", "metric", "value", "path"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    return df.sort_values(cols[:-2], kind="stable").reset_index(drop=True)


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and count over seeds per ``(experiment, model, geometry, K, dim, task, metric)``."""
    cols = [*CONFIG_KEYS, "mean", "std", "n_seeds"]
    if table.empty:
        return pd.DataFrame(columns=cols)
    grp = table.groupby(CONFIG_KEYS, dropna=False, sort=True)["value"]
    out = grp.agg(mean="mean", std="std", n_seeds="count").reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out[cols].sort_values(CONFIG_KEYS, kind="stable").reset_index(drop=True)


def format_mean_std(mean: float, std: float, n: int, float_fmt: str = "{:.4f}") -> str:
    """``mean ± std`` when more than one seed contributed, ``mean`` otherwise."""
    if n > 1:
        return f"{float_fmt.format(mean)} ± {float_fmt.format(std)}"
    return float_fmt.format(mean)


def sweep_table(summary: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """``(geometry, curvature)`` rows x ``latent_dim`` columns of ``mean ± std`` for one metric."""
    df = summary[(summary["task"] == task) & (summary["metric"] == metric)]
    dims = sorted(int(d) for d in df["latent_dim"].dropna().unique())
    rows = []
    for (geometry, curvature), grp in df.groupby(
        ["geometry", "curvature"], dropna=False, sort=True
    ):
        row = {"geometry": geometry, "curvature": curvature}
        for d in dims:
            cell = grp[grp["latent_dim"] == d]
            row[f"dim={d}"] = (
                ""
                if cell.empty
                else format_mean_std(
                    float(cell["mean"].iloc[0]),
                    float(cell["std"].iloc[0]),
                    int(cell["n_seeds"].iloc[0]),
                )
            )
        rows.append(row)
    cols = ["geometry", "curvature", *(f"dim={d}" for d in dims)]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["geometry", "curvature"], kind="stable").reset_index(drop=True)


def best_curvature_table(summary: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """Per latent dimension: Euclidean baseline vs the best swept curvature for one metric.

    ``exceeds_one_std`` applies the statistical protocol of ``docs/methodology.md``: a difference
    counts only if it exceeds one seed standard deviation at *both* configurations, which needs at
    least two seeds on each side (``n_seeds`` is the smaller of the two counts; with fewer than two
    the flag is always ``no``). Rows with no Euclidean run, or no hyperbolic run, at that dimension
    carry ``NaN`` in the missing side.
    """
    lower = lower_is_better(metric)
    df = summary[(summary["task"] == task) & (summary["metric"] == metric)]
    cols = [
        "task",
        "metric",
        "latent_dim",
        "euclidean_mean",
        "euclidean_std",
        "best_geometry",
        "best_curvature",
        "best_mean",
        "best_std",
        "n_seeds",
        "improvement",
        "exceeds_one_std",
    ]
    rows = []
    for d in sorted(int(x) for x in df["latent_dim"].dropna().unique()):
        at_d = df[df["latent_dim"] == d]
        euc = at_d[at_d["geometry"] == "euclidean"]
        hyp = at_d[at_d["geometry"] != "euclidean"]
        row: dict = {"task": task, "metric": metric, "latent_dim": d}
        n_seeds = math.inf
        if not euc.empty:
            row["euclidean_mean"] = float(euc["mean"].mean())
            row["euclidean_std"] = float(euc["std"].mean())
            n_seeds = min(n_seeds, int(euc["n_seeds"].min()))
        else:
            row["euclidean_mean"] = row["euclidean_std"] = math.nan
        if not hyp.empty:
            best = hyp.sort_values(["mean", "geometry", "curvature"], kind="stable")
            best = best.iloc[0] if lower else best.iloc[-1]
            row["best_geometry"] = best["geometry"]
            row["best_curvature"] = float(best["curvature"])
            row["best_mean"], row["best_std"] = float(best["mean"]), float(best["std"])
            n_seeds = min(n_seeds, int(best["n_seeds"]))
        else:
            row["best_geometry"] = ""
            row["best_curvature"] = row["best_mean"] = row["best_std"] = math.nan
        row["n_seeds"] = math.nan if math.isinf(n_seeds) else float(n_seeds)
        if not euc.empty and not hyp.empty:
            gain = row["euclidean_mean"] - row["best_mean"]
            row["improvement"] = gain if lower else -gain
            row["exceeds_one_std"] = bool(
                n_seeds >= 2 and row["improvement"] > max(row["euclidean_std"], row["best_std"])
            )
        else:
            row["improvement"], row["exceeds_one_std"] = math.nan, False
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def dimension_efficiency_table(summary: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """Log2 AUC and dimension-to-reach per ``geometry(K=...)`` for one metric.

    The threshold for ``dim_to_reach`` is the Euclidean baseline's value at its largest
    dimension, so the column reads "the dimension at which this geometry matches the flat
    baseline at full width"; it is ``NaN`` when there is no Euclidean run.
    """
    lower = lower_is_better(metric)
    df = summary[(summary["task"] == task) & (summary["metric"] == metric)].copy()
    cols = [
        "task",
        "metric",
        "label",
        "n_dims",
        "auc_log2",
        "best_value",
        "best_dim",
        "dim_to_reach",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["label"] = [
        f"{g}" if g == "euclidean" else f"{g}(K={k})"
        for g, k in zip(df["geometry"], df["curvature"], strict=True)
    ]
    curves = curves_from_table(df, label_col="label", dim_col="latent_dim", value_col="mean")
    euc = df[df["geometry"] == "euclidean"].sort_values("latent_dim")
    threshold = float(euc["mean"].iloc[-1]) if not euc.empty else None
    out = efficiency_table(curves, threshold=threshold, lower_is_better=lower)
    out.insert(0, "metric", metric)
    out.insert(0, "task", task)
    return out[cols]


def _format_cell(col: str, v: object, float_fmt: str) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if col in INT_COLS and v.is_integer():
            return str(int(v))
        return float_fmt.format(v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table without extra dependencies."""
    if df.empty:
        return "_(no runs found)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [_format_cell(c, row[c], float_fmt) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = [
    "COUNT_METRICS",
    "HIGHER_IS_BETTER",
    "best_curvature_table",
    "collect_runs",
    "dimension_efficiency_table",
    "format_mean_std",
    "is_result_metric",
    "lower_is_better",
    "results_table",
    "summary_table",
    "sweep_table",
    "to_markdown",
]
