"""Metrics for proving 3D coordinate stability and accuracy."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositionMetrics:
    """
    Summary metrics for a sequence of 3D positions.

    std_xyz / max_std_m measure stability. rmse_m / mean_error_m /
    max_error_m measure accuracy when a ground-truth coordinate is provided.
    """

    count: int
    mean_xyz: tuple[float, float, float]
    std_xyz: tuple[float, float, float]
    max_std_m: float
    rmse_m: float | None
    mean_error_m: float | None
    max_error_m: float | None


@dataclass(frozen=True)
class PositionMetricsComparison:
    """
    Comparison between baseline and candidate coordinate metrics.

    Positive improvement percentages mean the candidate is more stable or more
    accurate than the baseline.
    """

    baseline: PositionMetrics
    candidate: PositionMetrics
    max_std_improvement_pct: float | None
    rmse_improvement_pct: float | None
    mean_error_improvement_pct: float | None


def improvement_pct(
    baseline_value: float | None,
    candidate_value: float | None,
) -> float | None:
    """
    Calculate percentage reduction from baseline to candidate.

    Args:
        baseline_value: Baseline metric where lower is better.
        candidate_value: Candidate metric where lower is better.

    Returns:
        Positive percentage when candidate improved, negative when it regressed.
    """
    if baseline_value is None or candidate_value is None or baseline_value <= 0.0:
        return None
    return (baseline_value - candidate_value) / baseline_value * 100.0


def summarize_positions(
    positions: list[tuple[float, float, float]],
    ground_truth: tuple[float, float, float] | None = None,
) -> PositionMetrics:
    """
    Summarize stability and optional accuracy for 3D coordinate samples.

    Args:
        positions: Sequence of (x, y, z) samples in meters.
        ground_truth: Optional measured target coordinate in meters.

    Returns:
        PositionMetrics with per-axis standard deviation and optional errors.

    Raises:
        ValueError: If positions is empty or not shaped as Nx3.
    """
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("positions must be a non-empty sequence of (x, y, z)")

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    rmse_m = None
    mean_error_m = None
    max_error_m = None
    if ground_truth is not None:
        truth = np.asarray(ground_truth, dtype=np.float64)
        if truth.shape != (3,):
            raise ValueError("ground_truth must be a 3-element coordinate")
        errors = np.linalg.norm(arr - truth, axis=1)
        rmse_m = float(np.sqrt(np.mean(errors ** 2)))
        mean_error_m = float(np.mean(errors))
        max_error_m = float(np.max(errors))

    return PositionMetrics(
        count=int(arr.shape[0]),
        mean_xyz=(float(mean[0]), float(mean[1]), float(mean[2])),
        std_xyz=(float(std[0]), float(std[1]), float(std[2])),
        max_std_m=float(np.max(std)),
        rmse_m=rmse_m,
        mean_error_m=mean_error_m,
        max_error_m=max_error_m,
    )


def compare_position_sets(
    baseline_positions: list[tuple[float, float, float]],
    candidate_positions: list[tuple[float, float, float]],
    ground_truth: tuple[float, float, float] | None = None,
) -> PositionMetricsComparison:
    """
    Compare baseline and candidate coordinate samples.

    Args:
        baseline_positions: Samples from the old or reference algorithm.
        candidate_positions: Samples from the algorithm under test.
        ground_truth: Optional measured target coordinate in meters.

    Returns:
        PositionMetricsComparison with stability and optional accuracy deltas.
    """
    baseline = summarize_positions(baseline_positions, ground_truth=ground_truth)
    candidate = summarize_positions(candidate_positions, ground_truth=ground_truth)

    return PositionMetricsComparison(
        baseline=baseline,
        candidate=candidate,
        max_std_improvement_pct=improvement_pct(
            baseline.max_std_m,
            candidate.max_std_m,
        ),
        rmse_improvement_pct=improvement_pct(
            baseline.rmse_m,
            candidate.rmse_m,
        ),
        mean_error_improvement_pct=improvement_pct(
            baseline.mean_error_m,
            candidate.mean_error_m,
        ),
    )
