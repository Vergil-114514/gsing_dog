"""Unit tests for 3D coordinate evaluation metrics."""

import pytest

from detection_3d.evaluation import (
    compare_position_sets,
    improvement_pct,
    summarize_positions,
)


def test_summarize_positions_stability_metrics():
    samples = [
        (0.10, 0.20, 0.50),
        (0.11, 0.19, 0.52),
        (0.09, 0.21, 0.48),
    ]

    metrics = summarize_positions(samples)

    assert metrics.count == 3
    assert metrics.mean_xyz == pytest.approx((0.10, 0.20, 0.50))
    assert metrics.std_xyz[0] == pytest.approx(0.0081649658)
    assert metrics.std_xyz[1] == pytest.approx(0.0081649658)
    assert metrics.std_xyz[2] == pytest.approx(0.0163299316)
    assert metrics.max_std_m == pytest.approx(metrics.std_xyz[2])
    assert metrics.rmse_m is None


def test_summarize_positions_accuracy_metrics_with_ground_truth():
    samples = [
        (0.10, 0.20, 0.50),
        (0.11, 0.20, 0.50),
        (0.10, 0.22, 0.50),
    ]

    metrics = summarize_positions(samples, ground_truth=(0.10, 0.20, 0.50))

    assert metrics.rmse_m == pytest.approx(((0.0 ** 2 + 0.01 ** 2 + 0.02 ** 2) / 3) ** 0.5)
    assert metrics.mean_error_m == pytest.approx((0.0 + 0.01 + 0.02) / 3)
    assert metrics.max_error_m == pytest.approx(0.02)


def test_summarize_positions_single_sample_has_zero_std():
    metrics = summarize_positions([(1.0, 2.0, 3.0)])

    assert metrics.count == 1
    assert metrics.std_xyz == pytest.approx((0.0, 0.0, 0.0))
    assert metrics.max_std_m == 0.0


def test_summarize_positions_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        summarize_positions([])


def test_summarize_positions_rejects_invalid_ground_truth():
    with pytest.raises(ValueError, match="3-element"):
        summarize_positions([(0.0, 0.0, 0.0)], ground_truth=(0.0, 0.0))


def test_improvement_pct_reports_reduction_as_positive():
    assert improvement_pct(0.10, 0.04) == pytest.approx(60.0)
    assert improvement_pct(0.10, 0.12) == pytest.approx(-20.0)
    assert improvement_pct(0.0, 0.01) is None


def test_compare_position_sets_reports_stability_and_accuracy_improvement():
    baseline = [
        (0.10, 0.20, 0.50),
        (0.14, 0.20, 0.54),
        (0.06, 0.20, 0.46),
    ]
    candidate = [
        (0.10, 0.20, 0.50),
        (0.11, 0.20, 0.51),
        (0.09, 0.20, 0.49),
    ]

    comparison = compare_position_sets(
        baseline,
        candidate,
        ground_truth=(0.10, 0.20, 0.50),
    )

    assert comparison.baseline.count == 3
    assert comparison.candidate.count == 3
    assert comparison.max_std_improvement_pct > 0.0
    assert comparison.rmse_improvement_pct > 0.0
    assert comparison.mean_error_improvement_pct > 0.0
