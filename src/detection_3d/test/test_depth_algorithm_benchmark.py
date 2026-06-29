"""Tests for the synthetic depth estimator A/B benchmark."""

import pytest

from detection_3d.depth_algorithm_benchmark import (
    build_synthetic_depth_samples,
    format_benchmark_report,
    run_synthetic_benchmark,
)


def test_synthetic_benchmark_shows_cluster_estimator_improves_metrics():
    result = run_synthetic_benchmark(sample_count=40, seed=7)

    assert result.baseline_metrics.count == 40
    assert result.optimized_metrics.count == 40
    assert result.optimized_metrics.max_std_m < result.baseline_metrics.max_std_m * 0.5
    assert result.optimized_metrics.rmse_m < result.baseline_metrics.rmse_m * 0.1
    assert result.stability_improvement_pct > 50.0
    assert result.rmse_improvement_pct is not None
    assert result.rmse_improvement_pct > 90.0


def test_format_benchmark_report_contains_old_new_metric_names():
    result = run_synthetic_benchmark(sample_count=5, seed=1)

    report = format_benchmark_report(result)

    assert "baseline_center_median" in report
    assert "optimized_cluster_centroid" in report
    assert "max_std_m:" in report
    assert "rmse_m:" in report


def test_synthetic_samples_include_disconnected_same_depth_noise():
    sample = build_synthetic_depth_samples(sample_count=1, seed=7)[0]

    assert sample.roi[1:6, 4:8].mean() == pytest.approx(1000, abs=10)
    assert sample.roi[7:9, 0:2].mean() == pytest.approx(1000, abs=10)
