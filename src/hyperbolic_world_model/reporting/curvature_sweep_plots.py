"""Figures for curvature sweeps and rollout curves (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_curvature_sweep(table: pd.DataFrame, metric: str, out_path: Path) -> Path:
    """Metric vs curvature, one line per geometry, Euclidean baseline as a horizontal rule."""
    df = table[table["metric"] == metric]
    fig, ax = plt.subplots(figsize=(6, 4))
    for geometry, grp in df.groupby("geometry"):
        if geometry == "euclidean":
            ax.axhline(grp["value"].mean(), linestyle="--", label="euclidean (K=0)")
            continue
        agg = grp.groupby("curvature")["value"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["curvature"], agg["mean"], yerr=agg["std"].fillna(0), marker="o", label=geometry
        )
    ax.set_xlabel("curvature K")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs curvature (native geometry of each model)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_rollout_curves(curve_files: dict[str, Path], out_path: Path) -> Path:
    """Geodesic error vs horizon, one line per labelled run."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, path in curve_files.items():
        df = pd.read_csv(path)
        ax.plot(df["horizon"], df["geodesic_error"], marker="o", label=label)
    ax.set_xlabel("horizon")
    ax.set_ylabel("geodesic error (native geometry)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


__all__ = ["plot_curvature_sweep", "plot_rollout_curves"]
