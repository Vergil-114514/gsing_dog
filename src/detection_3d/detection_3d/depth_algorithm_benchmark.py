"""Synthetic benchmark for comparing depth target-point estimators."""

import argparse
from dataclasses import dataclass

import numpy as np

from detection_3d.depth_processor import (
    estimate_target_point_from_roi,
    filter_depth_roi,
)
from detection_3d.evaluation import PositionMetrics, improvement_pct, summarize_positions
from detection_3d.geometry import project_pixel_to_xyz


@dataclass(frozen=True)
class DepthBenchmarkSample:
    """
    One depth ROI sample with a known target point.

    The benchmark keeps the YOLO box center fixed and moves only the target
    surface inside the ROI. This isolates the algorithmic difference between
    projecting the box center and projecting the detected foreground cluster.
    """

    roi: np.ndarray
    truth_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class DepthBenchmarkResult:
    """
    A/B metrics for the old center-median estimator and current estimator.

    max_std_m measures stability. rmse_m measures accuracy against the known
    target coordinate in the synthetic scene.
    """

    baseline_metrics: PositionMetrics
    optimized_metrics: PositionMetrics
    stability_improvement_pct: float
    rmse_improvement_pct: float | None


def _project_with_offset(
    center_u_px: float,
    center_v_px: float,
    offset_x_px: float,
    offset_y_px: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    return project_pixel_to_xyz(
        center_u_px + offset_x_px,
        center_v_px + offset_y_px,
        depth_m,
        fx,
        fy,
        cx,
        cy,
    )


def build_synthetic_depth_samples(
    sample_count: int = 40,
    seed: int = 7,
    center_u_px: float = 320.0,
    center_v_px: float = 240.0,
    fx: float = 600.0,
    fy: float = 600.0,
    cx: float = 320.0,
    cy: float = 240.0,
) -> list[DepthBenchmarkSample]:
    """
    Build deterministic ROI samples for old/new estimator comparison.

    The synthetic scene models common grasping failure cases: the YOLO box
    contains more background than foreground, the true cube surface is shifted
    away from the box center, and a small same-depth blob appears away from the
    target surface.
    """
    rng = np.random.default_rng(seed)
    truth_depth_m = 1.0
    truth_offset_x_px = 1.5
    truth_offset_y_px = -1.0
    truth_xyz = _project_with_offset(
        center_u_px,
        center_v_px,
        truth_offset_x_px,
        truth_offset_y_px,
        truth_depth_m,
        fx,
        fy,
        cx,
        cy,
    )

    samples: list[DepthBenchmarkSample] = []
    for _ in range(sample_count):
        background_mm = rng.normal(loc=3000.0, scale=25.0, size=(9, 9))
        target_mm = rng.normal(loc=1000.0, scale=3.0, size=(5, 4))
        same_depth_noise_mm = rng.normal(loc=1000.0, scale=3.0, size=(2, 2))

        roi = np.clip(background_mm, 1.0, 65535.0).astype(np.uint16)
        roi[1:6, 4:8] = np.clip(target_mm, 1.0, 65535.0).astype(np.uint16)
        roi[7:9, 0:2] = np.clip(same_depth_noise_mm, 1.0, 65535.0).astype(np.uint16)

        # Simulate occasional invalid background depth without changing target truth.
        invalid_mask = rng.random((9, 9)) < 0.04
        invalid_mask[1:6, 4:8] = False
        invalid_mask[7:9, 0:2] = False
        roi[invalid_mask] = 0

        samples.append(DepthBenchmarkSample(roi=roi, truth_xyz=truth_xyz))

    return samples


def _baseline_center_median_position(
    roi: np.ndarray,
    depth_scale: float,
    center_u_px: float,
    center_v_px: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float] | None:
    depth_m, quality = filter_depth_roi(roi, depth_scale=depth_scale)
    if depth_m <= 0.0 or quality <= 0.0:
        return None
    return _project_with_offset(
        center_u_px,
        center_v_px,
        0.0,
        0.0,
        depth_m,
        fx,
        fy,
        cx,
        cy,
    )


def _optimized_cluster_position(
    roi: np.ndarray,
    depth_scale: float,
    center_u_px: float,
    center_v_px: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float] | None:
    estimate = estimate_target_point_from_roi(roi, depth_scale=depth_scale)
    if estimate is None or estimate.depth_m <= 0.0 or estimate.quality <= 0.0:
        return None
    return _project_with_offset(
        center_u_px,
        center_v_px,
        estimate.offset_x_px,
        estimate.offset_y_px,
        estimate.depth_m,
        fx,
        fy,
        cx,
        cy,
    )


def run_synthetic_benchmark(
    sample_count: int = 40,
    seed: int = 7,
    depth_scale: float = 0.001,
) -> DepthBenchmarkResult:
    """
    Compare old and current algorithms on deterministic synthetic depth ROIs.

    Returns:
        DepthBenchmarkResult with stability and accuracy improvement ratios.

    Raises:
        ValueError: If either estimator fails all benchmark samples.
    """
    center_u_px = 320.0
    center_v_px = 240.0
    fx = 600.0
    fy = 600.0
    cx = 320.0
    cy = 240.0

    samples = build_synthetic_depth_samples(
        sample_count=sample_count,
        seed=seed,
        center_u_px=center_u_px,
        center_v_px=center_v_px,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    baseline_positions: list[tuple[float, float, float]] = []
    optimized_positions: list[tuple[float, float, float]] = []
    truth_positions: list[tuple[float, float, float]] = []

    for sample in samples:
        baseline = _baseline_center_median_position(
            sample.roi,
            depth_scale,
            center_u_px,
            center_v_px,
            fx,
            fy,
            cx,
            cy,
        )
        optimized = _optimized_cluster_position(
            sample.roi,
            depth_scale,
            center_u_px,
            center_v_px,
            fx,
            fy,
            cx,
            cy,
        )
        if baseline is None or optimized is None:
            continue
        baseline_positions.append(baseline)
        optimized_positions.append(optimized)
        truth_positions.append(sample.truth_xyz)

    if not baseline_positions or not optimized_positions:
        raise ValueError("benchmark produced no comparable estimator outputs")

    truth = truth_positions[0]
    baseline_metrics = summarize_positions(baseline_positions, ground_truth=truth)
    optimized_metrics = summarize_positions(optimized_positions, ground_truth=truth)

    stability_improvement = improvement_pct(
        baseline_metrics.max_std_m,
        optimized_metrics.max_std_m,
    )
    rmse_improvement = improvement_pct(
        baseline_metrics.rmse_m,
        optimized_metrics.rmse_m,
    )

    return DepthBenchmarkResult(
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
        stability_improvement_pct=stability_improvement or 0.0,
        rmse_improvement_pct=rmse_improvement,
    )


def format_benchmark_report(result: DepthBenchmarkResult) -> str:
    """
    Format benchmark metrics as a compact report.

    Args:
        result: Synthetic benchmark result.

    Returns:
        Multiline text report.
    """
    baseline = result.baseline_metrics
    optimized = result.optimized_metrics
    lines = [
        "synthetic_depth_estimator_benchmark",
        f"samples: {baseline.count}",
        "baseline_center_median:",
        f"  max_std_m: {baseline.max_std_m:.6f}",
        f"  rmse_m: {baseline.rmse_m:.6f}",
        "optimized_cluster_centroid:",
        f"  max_std_m: {optimized.max_std_m:.6f}",
        f"  rmse_m: {optimized.rmse_m:.6f}",
        "improvement:",
        f"  max_std_m: {result.stability_improvement_pct:.2f}%",
    ]
    if result.rmse_improvement_pct is not None:
        lines.append(f"  rmse_m: {result.rmse_improvement_pct:.2f}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """
    Run the synthetic depth-estimator benchmark.

    Args:
        argv: Optional CLI arguments for tests.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Compare center-median and cluster-centroid depth estimators."
    )
    parser.add_argument("--samples", type=int, default=40, help="Number of synthetic frames")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args(argv)

    result = run_synthetic_benchmark(sample_count=args.samples, seed=args.seed)
    print(format_benchmark_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
