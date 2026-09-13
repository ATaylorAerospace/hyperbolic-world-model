"""Figures for curvature sweeps, dimension curves and per-task curves (matplotlib, headless).

Every figure is a pure function of the tables it is given: groups are iterated in sorted order
and the PNG metadata is stripped, so the same ``outputs/`` produces byte-identical files.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURVE_KEYS = ["experiment", "geometry", "curvature", "latent_dim"]
_SAVE = {"dpi": 150, "metadata": {"Software": None}, "bbox_inches": "tight"}


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **_SAVE)
    plt.close(fig)
    return out_path


def plot_curvature_sweep(summary: pd.DataFrame, metric: str, out_path: Path) -> Path:
    """Metric vs curvature: one line per (geometry, latent_dim); Euclidean as dashed rules.

    Args:
        summary: rows of :func:`~hyperbolic_world_model.reporting.tables.summary_table` for one
            task (any metric; filtered here).
        metric: metric to plot.
        out_path: PNG path.
    """
    df = summary[summary["metric"] == metric]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (geometry, dim), grp in df.groupby(["geometry", "latent_dim"], dropna=False, sort=True):
        label = f"{geometry}, dim={int(dim)}" if pd.notna(dim) else str(geometry)
        if geometry == "euclidean":
            ax.axhline(float(grp["mean"].mean()), linestyle="--", label=f"{label} (K=0)")
            continue
        grp = grp.sort_values("curvature")
        ax.errorbar(
            grp["curvature"], grp["mean"], yerr=grp["std"], marker="o", capsize=2, label=label
        )
    ax.set_xlabel("curvature K")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs curvature (native geometry of each model)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_dimension_curves(summary: pd.DataFrame, metric: str, out_path: Path) -> Path:
    """Metric vs latent dimension (log2 axis): one line per (geometry, curvature)."""
    df = summary[summary["metric"] == metric]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (geometry, k), grp in df.groupby(["geometry", "curvature"], dropna=False, sort=True):
        grp = grp.sort_values("latent_dim")
        label = "euclidean (K=0)" if geometry == "euclidean" else f"{geometry} (K={k})"
        ax.errorbar(
            grp["latent_dim"],
            grp["mean"],
            yerr=grp["std"],
            marker="o",
            capsize=2,
            linestyle="--" if geometry == "euclidean" else "-",
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("latent dimension")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs latent dimension (native geometry of each model)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out_path)


def aggregate_curves(
    curves: pd.DataFrame, x: str, keys: Sequence[str] = CURVE_KEYS
) -> pd.DataFrame:
    """Mean of every numeric column over seeds per ``(keys..., x)``."""
    numeric = [
        c
        for c in curves.columns
        if c not in {*keys, x, "seed"} and pd.api.types.is_numeric_dtype(curves[c])
    ]
    grp = curves.groupby([*keys, x], dropna=False, sort=True)[numeric].mean().reset_index()
    return grp


def plot_task_curves(curves: pd.DataFrame, x: str, task: str, out_path: Path) -> Path:
    """One subplot per curve column, one line per run configuration (seeds averaged).

    Args:
        curves: concatenation of a task's ``*_curves.csv`` files with the run keys
            (:data:`CURVE_KEYS` and ``seed``) added as columns.
        x: the x column (``horizon`` or ``depth``).
        task: task name, for the title.
        out_path: PNG path.
    """
    agg = aggregate_curves(curves, x)
    cols = [
        c
        for c in agg.columns
        if c not in {*CURVE_KEYS, x}
        and not c.startswith("n_")  # counts (n_pairs, n_nodes) are bookkeeping, not curves
        and pd.api.types.is_numeric_dtype(agg[c])
    ]
    n = max(len(cols), 1)
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.6 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, cols, strict=False):
        for key, grp in agg.groupby(CURVE_KEYS, dropna=False, sort=True):
            exp, geometry, k, dim = key
            label = (
                f"{exp}/{geometry}"
                + ("" if geometry == "euclidean" else f"(K={k})")
                + f" dim={int(dim)}"
            )
            ax.plot(grp[x], grp[col], marker="o", markersize=3, label=label)
        ax.set_xlabel(x)
        ax.set_ylabel(col)
        ax.set_title(f"{task}: {col}", fontsize=9)
    for ax in list(axes.flat)[len(cols) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f"{task} curves (native geometry of each model)", fontsize=10)
    fig.tight_layout()
    if handles:
        fig.legend(
            handles, labels, fontsize=6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0)
        )
    return _save(fig, out_path)


def plot_rollout_curves(curve_files: dict[str, Path], out_path: Path) -> Path:
    """Geodesic error vs horizon, one line per labelled ``latent_rollout_curves.csv``."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, path in sorted(curve_files.items()):
        df = pd.read_csv(path)
        ax.plot(df["horizon"], df["geodesic_error"], marker="o", label=label)
    ax.set_xlabel("horizon")
    ax.set_ylabel("geodesic error (native geometry)")
    if curve_files:
        ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out_path)


__all__ = [
    "CURVE_KEYS",
    "aggregate_curves",
    "plot_curvature_sweep",
    "plot_dimension_curves",
    "plot_rollout_curves",
    "plot_task_curves",
]
