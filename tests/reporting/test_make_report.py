"""One command regenerates every table and figure, deterministically, from ``outputs/``."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hyperbolic_world_model.reporting.make_report import main, make_report
from tests.reporting.helpers import write_outputs_tree

EXPECTED_TABLES = {
    "results.md",
    "summary.md",
    "best_curvature_vs_euclidean.md",
    "dimension_efficiency.md",
    "latent_rollout__geodesic_error_hmax__curvature_sweep.md",
    "hierarchy_reconstruction__map__curvature_sweep.md",
}
EXPECTED_FIGURES = {
    "latent_rollout__geodesic_error_hmax__vs_curvature.png",
    "latent_rollout__geodesic_error_hmax__vs_dimension.png",
    "hierarchy_reconstruction__map__vs_curvature.png",
    "hierarchy_reconstruction__map__vs_dimension.png",
    "latent_rollout__curves.png",
    "hierarchy_reconstruction__curves.png",
}


def _digest(paths: list[Path]) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_make_report_writes_every_table_and_figure(tmp_path: Path) -> None:
    outputs = write_outputs_tree(tmp_path / "outputs")
    report = outputs / "report"
    written = make_report(outputs, report)
    tables = {p.name for p in written if p.parent.name == "tables"}
    figures = {p.name for p in written if p.parent.name == "figures"}
    assert tables == EXPECTED_TABLES
    assert figures == EXPECTED_FIGURES
    assert (report / "README.md") in written
    index = (report / "README.md").read_text()
    assert "Runs found: 12" in index
    for name in EXPECTED_TABLES | EXPECTED_FIGURES:
        assert name in index
    # Count metrics and *_std never get their own sweep table or figure.
    assert not any("n_items" in p.name or "map_std" in p.name for p in written)
    summary = (report / "tables" / "summary.md").read_text()
    assert "| geometry |" not in summary.splitlines()[0]  # title first, table later
    assert "poincare | -1.0000 |" in summary and "euclidean | 0.0000 |" in summary
    best = (report / "tables" / "best_curvature_vs_euclidean.md").read_text()
    assert "| yes |" in best and "hierarchy_reconstruction | map |" in best


def test_report_is_a_pure_function_of_outputs(tmp_path: Path) -> None:
    outputs = write_outputs_tree(tmp_path / "outputs")
    report = tmp_path / "report"
    first = _digest(make_report(outputs, report))
    stale = report / "tables" / "stale.md"
    stale.write_text("left over from an earlier run")
    (report / "figures" / "old.png").write_bytes(b"png")
    (report / "notes.txt").write_text("not ours")
    second = _digest(make_report(outputs, report))
    assert first == second
    assert not stale.exists() and not (report / "figures" / "old.png").exists()
    assert (report / "notes.txt").exists()  # only generated files are removed


def test_without_a_sweep_there_are_no_curvature_figures(tmp_path: Path) -> None:
    outputs = write_outputs_tree(tmp_path / "outputs", with_sweep=False)
    written = make_report(outputs, tmp_path / "report")
    names = {p.name for p in written}
    assert not any("curvature" in n and n.endswith(".png") for n in names)
    assert "latent_rollout__geodesic_error_hmax__vs_dimension.png" in names
    best = (tmp_path / "report" / "tables" / "best_curvature_vs_euclidean.md").read_text()
    assert "| no |" in best and "| yes |" not in best


def test_empty_outputs_still_produce_a_report(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    written = make_report(tmp_path / "outputs", tmp_path / "report")
    names = {p.name for p in written}
    assert names == {
        "results.md",
        "summary.md",
        "best_curvature_vs_euclidean.md",
        "dimension_efficiency.md",
        "README.md",
    }
    assert "_(no runs found)_" in (tmp_path / "report" / "tables" / "results.md").read_text()
    assert "Runs found: 0" in (tmp_path / "report" / "README.md").read_text()


def test_cli_prints_every_written_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outputs = write_outputs_tree(tmp_path / "outputs", with_sweep=False)
    main(["--outputs", str(outputs), "--report", str(tmp_path / "report")])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(set(lines)) >= 5
    assert all(Path(line).exists() for line in lines)
    assert lines == [*sorted(lines[:-1]), lines[-1]] and lines[-1].endswith("README.md")
