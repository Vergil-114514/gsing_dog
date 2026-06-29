"""Tests for the coordinate CSV comparison CLI."""

from detection_3d.compare_positions_csv import format_comparison, main
from detection_3d.evaluation import compare_position_sets


def test_format_comparison_includes_baseline_candidate_and_improvement():
    comparison = compare_position_sets(
        baseline_positions=[
            (0.10, 0.20, 0.50),
            (0.14, 0.20, 0.54),
            (0.06, 0.20, 0.46),
        ],
        candidate_positions=[
            (0.10, 0.20, 0.50),
            (0.11, 0.20, 0.51),
            (0.09, 0.20, 0.49),
        ],
        ground_truth=(0.10, 0.20, 0.50),
    )

    report = format_comparison(comparison)

    assert "baseline:" in report
    assert "candidate:" in report
    assert "improvement:" in report
    assert "max_std_m:" in report
    assert "rmse_m:" in report


def test_main_compares_two_csv_files(tmp_path, capsys):
    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"
    baseline_csv.write_text(
        "x,y,z\n0.10,0.20,0.50\n0.14,0.20,0.54\n0.06,0.20,0.46\n",
        encoding="utf-8",
    )
    candidate_csv.write_text(
        "x,y,z\n0.10,0.20,0.50\n0.11,0.20,0.51\n0.09,0.20,0.49\n",
        encoding="utf-8",
    )

    result = main([
        str(baseline_csv),
        str(candidate_csv),
        "--ground-truth",
        "0.10",
        "0.20",
        "0.50",
    ])
    output = capsys.readouterr().out

    assert result == 0
    assert "baseline:" in output
    assert "candidate:" in output
    assert "improvement:" in output


def test_main_compares_two_column_sets_from_one_csv(tmp_path, capsys):
    paired_csv = tmp_path / "paired.csv"
    paired_csv.write_text(
        (
            "baseline_x,baseline_y,baseline_z,optimized_x,optimized_y,optimized_z\n"
            "0.10,0.20,0.50,0.10,0.20,0.50\n"
            "0.14,0.20,0.54,0.11,0.20,0.51\n"
            "0.06,0.20,0.46,0.09,0.20,0.49\n"
        ),
        encoding="utf-8",
    )

    result = main([
        str(paired_csv),
        "--baseline-columns",
        "baseline_x",
        "baseline_y",
        "baseline_z",
        "--candidate-columns",
        "optimized_x",
        "optimized_y",
        "optimized_z",
        "--ground-truth",
        "0.10",
        "0.20",
        "0.50",
    ])
    output = capsys.readouterr().out

    assert result == 0
    assert "improvement:" in output
    assert "n/a" not in output


def test_main_passes_when_minimum_improvement_thresholds_are_met(tmp_path, capsys):
    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"
    baseline_csv.write_text(
        "x,y,z\n0.10,0.20,0.50\n0.14,0.20,0.54\n0.06,0.20,0.46\n",
        encoding="utf-8",
    )
    candidate_csv.write_text(
        "x,y,z\n0.10,0.20,0.50\n0.11,0.20,0.51\n0.09,0.20,0.49\n",
        encoding="utf-8",
    )

    result = main([
        str(baseline_csv),
        str(candidate_csv),
        "--ground-truth",
        "0.10",
        "0.20",
        "0.50",
        "--min-stability-improvement-pct",
        "50",
        "--min-rmse-improvement-pct",
        "50",
    ])
    output = capsys.readouterr().out

    assert result == 0
    assert "threshold_failed" not in output


def test_main_fails_when_minimum_improvement_threshold_is_not_met(tmp_path, capsys):
    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"
    baseline_csv.write_text(
        "x,y,z\n0.10,0.20,0.50\n0.14,0.20,0.54\n0.06,0.20,0.46\n",
        encoding="utf-8",
    )
    candidate_csv.write_text(
        "x,y,z\n0.10,0.20,0.50\n0.11,0.20,0.51\n0.09,0.20,0.49\n",
        encoding="utf-8",
    )

    result = main([
        str(baseline_csv),
        str(candidate_csv),
        "--ground-truth",
        "0.10",
        "0.20",
        "0.50",
        "--min-stability-improvement-pct",
        "90",
    ])
    output = capsys.readouterr().out

    assert result == 1
    assert "threshold_failed: max_std_m=" in output
