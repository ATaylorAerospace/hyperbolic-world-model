"""Regenerate every table and figure from ``outputs/``.

Usage::

    python -m hyperbolic_world_model.reporting.make_report --outputs outputs --report outputs/report

Writes ``outputs/report/tables/results.md`` and one figure per (task, metric) that has more than
one curvature, plus the rollout curves for every run that produced them. Running it twice on the
same ``outputs/`` gives byte-identical results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hyperbolic_world_model.reporting.curvature_sweep_plots import (
    plot_curvature_sweep,
    plot_rollout_curves,
)
from hyperbolic_world_model.reporting.tables import collect_runs, results_table, to_markdown


def make_report(outputs: Path, report: Path) -> list[Path]:
    runs = collect_runs(outputs)
    table = results_table(runs)
    written: list[Path] = []

    tables_dir = report / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    md = tables_dir / "results.md"
    md.write_text(
        "# Results (all distances in each model's native geometry)\n\n" + to_markdown(table) + "\n"
    )
    written.append(md)

    figs = report / "figures"
    for (task, metric), grp in table.groupby(["task", "metric"]):
        if grp["curvature"].nunique() > 1:
            written.append(
                plot_curvature_sweep(grp, metric, figs / f"{task}_{metric}_vs_curvature.png")
            )

    curve_files = {}
    for run in runs:
        for csv in sorted(Path(run["_path"]).glob("*_curves.csv")):
            curve_files[
                f"{run.get('experiment')}/{run.get('geometry')}(K={run.get('curvature')})"
            ] = csv
    if curve_files:
        written.append(plot_rollout_curves(curve_files, figs / "rollout_error_vs_horizon.png"))
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    p.add_argument("--report", type=Path, default=Path("outputs/report"))
    args = p.parse_args(argv)
    for path in make_report(args.outputs, args.report):
        print(path)


if __name__ == "__main__":
    main()
