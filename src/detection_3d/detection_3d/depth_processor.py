"""
Adaptive ROI depth extraction with outlier rejection and quality scoring.

All functions are pure (no side effects), making them easy to unit-test.
"""

import numpy as np


def compute_roi_size(
    box_w_px: float,
    box_h_px: float,
    roi_ratio: float = 0.3,
    min_size: int = 5,
) -> int:
    """
    Compute ROI size from detection box dimensions.

    Args:
        box_w_px, box_h_px: Detection box width/height in pixels.
        roi_ratio: Fraction of the smaller box dimension to use as ROI.
        min_size: Minimum ROI size (pixels), prevents degenerate ROI on tiny boxes.

    Returns:
        Odd integer >= min_size representing ROI side length.
    """
    size = int(min(box_w_px, box_h_px) * roi_ratio)
    if size < min_size:
        size = min_size
    if size % 2 == 0:
        size += 1
    return size


def extract_roi(
    depth_image: np.ndarray,
    cx_px: int,
    cy_px: int,
    roi_size: int,
) -> np.ndarray | None:
    """
    Extract a square ROI from the depth image centered at (cx_px, cy_px).

    Clamps to image bounds. Returns None if the ROI is completely out of bounds.

    Args:
        depth_image: 2D depth array (H, W).
        cx_px, cy_px: Center pixel coordinates.
        roi_size: Side length (odd).

    Returns:
        ROI as 2D ndarray or None.
    """
    h, w = depth_image.shape[:2]
    half = roi_size // 2

    x1 = max(0, cx_px - half)
    y1 = max(0, cy_px - half)
    x2 = min(w, cx_px + half + 1)
    y2 = min(h, cy_px + half + 1)

    if x2 <= x1 or y2 <= y1:
        return None
    return depth_image[y1:y2, x1:x2]


def filter_depth_roi(
    roi: np.ndarray,
    depth_scale: float = 0.001,
    min_valid_ratio: float = 0.3,
    outlier_sigma: float = 2.0,
) -> tuple[float, float]:
    """
    Filter a depth ROI and return (depth_meters, quality_score).

    Processing:
      1. Exclude zero/invalid pixels.
      2. If valid ratio < min_valid_ratio, return (0.0, 0.0).
      3. Filter outliers: values > outlier_sigma * std from median.
      4. Return median depth and quality score (0–1).

    Quality score = valid_ratio * (1 - cv) where cv = std/mean of inliers.

    Returns:
        (depth_m, quality) — depth in meters, quality in [0, 1].
    """
    valid_mask = roi > 0
    valid_count = int(valid_mask.sum())
    total_count = int(roi.size)
    valid_ratio = valid_count / total_count if total_count > 0 else 0.0

    if valid_ratio < min_valid_ratio or valid_count < 3:
        return 0.0, 0.0

    valid_values = roi[valid_mask].astype(np.float64)
    median_val = float(np.median(valid_values))
    std_val = float(np.std(valid_values))

    # Outlier rejection: keep values within outlier_sigma * std of the median
    if std_val > 0 and valid_count > 3:
        lo = median_val - outlier_sigma * std_val
        hi = median_val + outlier_sigma * std_val
        inliers = valid_values[(valid_values >= lo) & (valid_values <= hi)]
        if len(inliers) >= 3:
            valid_values = inliers
            median_val = float(np.median(valid_values))
            std_val = float(np.std(valid_values))

    depth_m = median_val * depth_scale

    # Quality: valid_ratio * stability (1 - coefficient_of_variation)
    mean_val = float(np.mean(valid_values))
    cv = std_val / mean_val if mean_val > 0 else 1.0
    stability = max(0.0, 1.0 - cv)
    quality = valid_ratio * stability

    return depth_m, float(np.clip(quality, 0.0, 1.0))
