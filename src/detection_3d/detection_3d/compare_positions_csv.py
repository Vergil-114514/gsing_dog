"""Command-line A/B comparison for 3D coordinate sample CSV files."""

import argparse

from detection_3d.evaluate_positions_csv import format_metrics, load_positions_csv
from detection_3d.evaluation import PositionMetricsComparison, compare_position_sets


def _format_optional_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _check_min_improvement(
    metric_name: str,
    actual_pct: float | None,
    required_pct: float | None,
) -> str | None:
    if required_pct is None:
        return None
    if actual_pct is None:
        return f"{metric_name}=n/a, required>={required_pct:.2f}%"
    if actual_pct < required_pct:
        return f"{metric_name}={actual_pct:.2f}%, required>={required_pct:.2f}%"
    return None


def format_comparison(comparison: PositionMetricsComparison) -> str:
    """
    Format baseline/candidate metrics and improvement percentages.

    Args:
        comparison: Metrics comparison for two coordinate sample sets.

    Returns:
        Multiline text report.
    """
    lines = [
        "baseline:",
        format_metrics(comparison.baseline),
        "",
        "candidate:",
        format_metrics(comparison.candidate),
        "",
        "improvement:",
        f"max_std_m: {_format_optional_pct(comparison.max_std_improvement_pct)}",
        f"rmse_m: {_format_optional_pct(comparison.rmse_improvement_pct)}",
        f"mean_error_m: {_format_optional_pct(comparison.mean_error_improvement_pct)}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """
    Compare two CSV coordinate sample sets.

    Args:
        argv: Optional CLI arguments for tests.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate 3D coordinate CSV samples."
    )
    parser.add_argument("baseline_csv", help="CSV file from the baseline algorithm")
    parser.add_argument(
        "candidate_csv",
        nargs="?",
        help="CSV file from the candidate algorithm; defaults to baseline_csv",
    )
    parser.add_argument(
        "--columns",
        nargs=3,
        default=("x", "y", "z"),
        metavar=("X_COL", "Y_COL", "Z_COL"),
        help="Default column names for both baseline and candidate coordinates",
    )
    parser.add_argument(
        "--baseline-columns",
        nargs=3,
        metavar=("X_COL", "Y_COL", "Z_COL"),
        help="Column names for baseline x/y/z coordinates",
    )
    parser.add_argument(
        "--candidate-columns",
        nargs=3,
        metavar=("X_COL", "Y_COL", "Z_COL"),
        help="Column names for candidate x/y/z coordinates",
    )
    parser.add_argument(
        "--ground-truth",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional measured target coordinate in meters",
    )
    parser.add_argument(
        "--min-stability-improvement-pct",
        type=float,
        help="Fail if max_std_m improvement is below this percentage",
    )
    parser.add_argument(
        "--min-rmse-improvement-pct",
        type=float,
        help="Fail if rmse_m improvement is below this percentage",
    )
    args = parser.parse_args(argv)

    baseline_columns = tuple(args.baseline_columns or args.columns)
    candidate_columns = tuple(args.candidate_columns or args.columns)
    candidate_csv = args.candidate_csv or args.baseline_csv

    baseline_positions = load_positions_csv(args.baseline_csv, baseline_columns)
    candidate_positions = load_positions_csv(candidate_csv, candidate_columns)
    ground_truth = tuple(args.ground_truth) if args.ground_truth is not None else None
    comparison = compare_position_sets(
        baseline_positions,
        candidate_positions,
        ground_truth=ground_truth,
    )
    print(format_comparison(comparison))

    failures = [
        failure
        for failure in (
            _check_min_improvement(
                "max_std_m",
                comparison.max_std_improvement_pct,
                args.min_stability_improvement_pct,
            ),
            _check_min_improvement(
                "rmse_m",
                comparison.rmse_improvement_pct,
                args.min_rmse_improvement_pct,
            ),
        )
        if failure is not None
    ]
    for failure in failures:
        print(f"threshold_failed: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
