"""Unit tests for the coordinate CSV evaluation CLI."""

import pytest

from detection_3d.evaluate_positions_csv import (
    format_metrics,
    load_positions_csv,
    main,
)
from detection_3d.evaluation import summarize_positions


def test_load_positions_csv_default_columns(tmp_path):
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("x,y,z\n0.1,0.2,0.5\n0.2,0.3,0.6\n", encoding="utf-8")

    positions = load_positions_csv(csv_path)

    assert positions == [(0.1, 0.2, 0.5), (0.2, 0.3, 0.6)]


def test_load_positions_csv_custom_columns(tmp_path):
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(
        "pos_x,pos_y,pos_z\n0.1,0.2,0.5\n",
        encoding="utf-8",
    )

    positions = load_positions_csv(csv_path, columns=("pos_x", "pos_y", "pos_z"))

    assert positions == [(0.1, 0.2, 0.5)]


def test_load_positions_csv_missing_columns(tmp_path):
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("x,y\n0.1,0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_positions_csv(csv_path)


def test_format_metrics_includes_accuracy_when_ground_truth_is_present():
    metrics = summarize_positions(
        [(0.1, 0.2, 0.5), (0.2, 0.2, 0.5)],
        ground_truth=(0.1, 0.2, 0.5),
    )

    report = format_metrics(metrics)

    assert "count: 2" in report
    assert "max_std_m:" in report
    assert "rmse_m:" in report


def test_main_prints_report(tmp_path, capsys):
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("x,y,z\n0.1,0.2,0.5\n0.2,0.2,0.5\n", encoding="utf-8")

    result = main([str(csv_path), "--ground-truth", "0.1", "0.2", "0.5"])
    output = capsys.readouterr().out

    assert result == 0
    assert "count: 2" in output
    assert "rmse_m:" in output
