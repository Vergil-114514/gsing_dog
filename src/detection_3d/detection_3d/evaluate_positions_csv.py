"""Command-line CSV evaluator for 3D detection coordinate samples."""

import argparse
import csv
from pathlib import Path

from detection_3d.evaluation import PositionMetrics, summarize_positions


def load_positions_csv(
    csv_path: str | Path,
    columns: tuple[str, str, str] = ("x", "y", "z"),
) -> list[tuple[float, float, float]]:
    """
    Load (x, y, z) samples from a CSV file.

    Args:
        csv_path: Path to a CSV file with a header row.
        columns: Column names for x, y, z coordinates.

    Returns:
        List of coordinate tuples in meters.

    Raises:
        ValueError: If required columns are missing or values are invalid.
    """
    path = Path(csv_path)
    positions: list[tuple[float, float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file must contain a header row")
        missing = [name for name in columns if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        for row_num, row in enumerate(reader, start=2):
            try:
                positions.append(
                    (
                        float(row[columns[0]]),
                        float(row[columns[1]]),
                        float(row[columns[2]]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid coordinate at CSV row {row_num}") from exc

    return positions


def format_metrics(metrics: PositionMetrics) -> str:
    """
    Format PositionMetrics as a compact human-readable report.

    Args:
        metrics: Calculated coordinate metrics.

    Returns:
        Multiline text report.
    """
    lines = [
        f"count: {metrics.count}",
        (
            "mean_xyz_m: "
            f"{metrics.mean_xyz[0]:.6f}, "
            f"{metrics.mean_xyz[1]:.6f}, "
            f"{metrics.mean_xyz[2]:.6f}"
        ),
        (
            "std_xyz_m: "
            f"{metrics.std_xyz[0]:.6f}, "
            f"{metrics.std_xyz[1]:.6f}, "
            f"{metrics.std_xyz[2]:.6f}"
        ),
        f"max_std_m: {metrics.max_std_m:.6f}",
    ]
    if metrics.rmse_m is not None:
        lines.extend([
            f"rmse_m: {metrics.rmse_m:.6f}",
            f"mean_error_m: {metrics.mean_error_m:.6f}",
            f"max_error_m: {metrics.max_error_m:.6f}",
        ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """
    Evaluate a CSV file of 3D coordinate samples.

    Args:
        argv: Optional CLI arguments for tests.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate stability and accuracy of 3D coordinate samples."
    )
    parser.add_argument("csv_path", help="CSV file containing coordinate samples")
    parser.add_argument(
        "--columns",
        nargs=3,
        default=("x", "y", "z"),
        metavar=("X_COL", "Y_COL", "Z_COL"),
        help="Column names for x/y/z coordinates",
    )
    parser.add_argument(
        "--ground-truth",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional measured target coordinate in meters",
    )
    args = parser.parse_args(argv)

    positions = load_positions_csv(args.csv_path, tuple(args.columns))
    ground_truth = tuple(args.ground_truth) if args.ground_truth is not None else None
    metrics = summarize_positions(positions, ground_truth=ground_truth)
    print(format_metrics(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
