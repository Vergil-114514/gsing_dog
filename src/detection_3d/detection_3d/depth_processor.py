"""
Adaptive ROI depth extraction with outlier rejection and quality scoring.

All functions are pure (no side effects), making them easy to unit-test.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthEstimate:
    """
    Estimated target surface inside a depth ROI.

    offset_x_px / offset_y_px are relative to the ROI center. They let callers
    project the depth-cluster centroid instead of blindly projecting the YOLO
    box center.
    """

    depth_m: float
    quality: float
    offset_x_px: float
    offset_y_px: float
    valid_ratio: float
    cluster_ratio: float


def compute_roi_size(
    box_w_px: float,
    box_h_px: float,
    roi_ratio: float = 0.15,
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


def _normalize_kernel_size(kernel_size: int) -> int:
    size = max(3, int(kernel_size))
    if size % 2 == 0:
        size += 1
    return size


def _assign_depth_value(target: np.ndarray, y: int, x: int, value: float) -> None:
    if np.issubdtype(target.dtype, np.integer):
        info = np.iinfo(target.dtype)
        value = float(np.clip(round(value), info.min, info.max))
    target[y, x] = value


def clean_depth_roi(
    roi: np.ndarray,
    depth_scale: float = 0.001,
    hole_fill_enabled: bool = True,
    hole_fill_kernel_size: int = 3,
    hole_fill_min_neighbors: int = 4,
    spatial_outlier_threshold_m: float = 0.05,
) -> np.ndarray:
    """
    Clean a small depth ROI before estimating the target point.

    The filter is intentionally local to the ROI: it fills small Astra depth
    holes from nearby valid pixels and removes isolated depth spikes before
    they can become a wrong 3D target.
    """
    if roi.size == 0:
        return roi.copy()

    cleaned = roi.copy()
    kernel_size = _normalize_kernel_size(hole_fill_kernel_size)
    half = kernel_size // 2
    min_neighbors = max(1, int(hole_fill_min_neighbors))
    threshold_raw = float(spatial_outlier_threshold_m) / max(float(depth_scale), 1e-12)
    h, w = cleaned.shape[:2]

    if hole_fill_enabled:
        source = cleaned.copy()
        for y in range(h):
            for x in range(w):
                if source[y, x] != 0:
                    continue

                y1 = max(0, y - half)
                y2 = min(h, y + half + 1)
                x1 = max(0, x - half)
                x2 = min(w, x + half + 1)
                neighbors = source[y1:y2, x1:x2]
                valid = neighbors[neighbors > 0]
                if len(valid) >= min_neighbors:
                    _assign_depth_value(cleaned, y, x, float(np.median(valid)))

    if spatial_outlier_threshold_m <= 0.0:
        return cleaned

    source = cleaned.copy()
    for y in range(h):
        for x in range(w):
            value = float(source[y, x])
            if value <= 0.0:
                continue

            y1 = max(0, y - half)
            y2 = min(h, y + half + 1)
            x1 = max(0, x - half)
            x2 = min(w, x + half + 1)
            neighbors = source[y1:y2, x1:x2].astype(np.float64)
            valid_mask = neighbors > 0.0
            valid_mask[y - y1, x - x1] = False
            valid = neighbors[valid_mask]
            if len(valid) < min_neighbors:
                continue

            same_depth_neighbors = valid[np.abs(valid - value) <= threshold_raw]
            if len(same_depth_neighbors) < 1:
                cleaned[y, x] = 0

    return cleaned


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


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_component: list[tuple[int, int]] = []

    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []

        while stack:
            y, x = stack.pop()
            component.append((y, x))

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if len(component) > len(best_component):
            best_component = component

    component_mask = np.zeros_like(mask, dtype=bool)
    for y, x in best_component:
        component_mask[y, x] = True
    return component_mask


def estimate_target_point_from_roi(
    roi: np.ndarray,
    depth_scale: float = 0.001,
    min_valid_ratio: float = 0.3,
    outlier_sigma: float = 2.0,
    cluster_tolerance_m: float = 0.03,
    min_cluster_ratio: float = 0.15,
) -> DepthEstimate | None:
    """
    Estimate target depth and pixel centroid from a depth ROI.

    The existing center-point algorithm assumes the YOLO box center is the
    target point. This estimator finds the closest reliable depth cluster, then
    computes the centroid of pixels that belong to that cluster. That is more
    accurate when the box center is biased, contains depth holes, or includes
    background pixels behind the target.

    Args:
        roi: 2D depth array.
        depth_scale: Raw depth unit to meters.
        min_valid_ratio: Minimum ratio of non-zero depth pixels.
        outlier_sigma: Robust spread multiplier used for the depth cluster.
        cluster_tolerance_m: Depth gap used to split foreground/background clusters.
        min_cluster_ratio: Minimum valid-pixel ratio for a reliable depth cluster.

    Returns:
        DepthEstimate or None if the ROI has too little reliable depth.
    """
    valid_mask = roi > 0
    valid_count = int(valid_mask.sum())
    total_count = int(roi.size)
    valid_ratio = valid_count / total_count if total_count > 0 else 0.0
    if valid_ratio < min_valid_ratio or valid_count < 3:
        return None

    depth_m = roi.astype(np.float64) * depth_scale
    valid_depths = depth_m[valid_mask]
    sorted_depths = np.sort(valid_depths)
    tolerance_m = max(float(cluster_tolerance_m), 1e-6)
    min_cluster_count = max(3, int(np.ceil(valid_count * max(0.0, min_cluster_ratio))))

    clusters: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(sorted_depths)):
        if sorted_depths[idx] - sorted_depths[idx - 1] > tolerance_m:
            clusters.append((start, idx))
            start = idx
    clusters.append((start, len(sorted_depths)))

    selected_mask: np.ndarray | None = None
    for start, end in clusters:
        cluster = sorted_depths[start:end]
        if len(cluster) < min_cluster_count:
            continue
        cluster_min = float(cluster[0] - tolerance_m)
        cluster_max = float(cluster[-1] + tolerance_m)
        cluster_mask = valid_mask & (depth_m >= cluster_min) & (depth_m <= cluster_max)
        cluster_mask = _largest_connected_component(cluster_mask)
        if int(cluster_mask.sum()) >= min_cluster_count:
            selected_mask = cluster_mask
            break

    if selected_mask is None:
        anchor_depth_m, base_quality = filter_depth_roi(
            roi,
            depth_scale=depth_scale,
            min_valid_ratio=min_valid_ratio,
            outlier_sigma=outlier_sigma,
        )
        if anchor_depth_m <= 0.0 or base_quality <= 0.0:
            return None
        cluster_min = float(anchor_depth_m - tolerance_m)
        cluster_max = float(anchor_depth_m + tolerance_m)
        selected_mask = valid_mask & (depth_m >= cluster_min) & (depth_m <= cluster_max)
        selected_mask = _largest_connected_component(selected_mask)

    cluster_count = int(selected_mask.sum())
    if cluster_count < 3:
        return None

    ys, xs = np.nonzero(selected_mask)
    cluster_depths = depth_m[selected_mask]
    cluster_depth_m = float(np.median(cluster_depths))

    center_x = (roi.shape[1] - 1) / 2.0
    center_y = (roi.shape[0] - 1) / 2.0
    offset_x = float(np.mean(xs) - center_x)
    offset_y = float(np.mean(ys) - center_y)

    cluster_ratio = cluster_count / valid_count
    mean_depth = float(np.mean(cluster_depths))
    cv = float(np.std(cluster_depths) / mean_depth) if mean_depth > 0 else 1.0
    stability = max(0.0, 1.0 - cv)
    quality = valid_ratio * cluster_ratio * stability

    return DepthEstimate(
        depth_m=cluster_depth_m,
        quality=float(np.clip(quality, 0.0, 1.0)),
        offset_x_px=offset_x,
        offset_y_px=offset_y,
        valid_ratio=valid_ratio,
        cluster_ratio=cluster_ratio,
    )
