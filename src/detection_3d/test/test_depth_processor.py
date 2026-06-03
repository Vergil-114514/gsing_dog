"""Unit tests for depth_processor module."""

import numpy as np
import pytest
from detection_3d.depth_processor import (
    compute_roi_size,
    extract_roi,
    filter_depth_roi,
)


# ---------------------------------------------------------------------------
# compute_roi_size
# ---------------------------------------------------------------------------

class TestComputeRoiSize:
    def test_default_ratio(self):
        """ROI = 0.3 × smaller box dimension, odd, >= min_size."""
        size = compute_roi_size(100, 80, roi_ratio=0.3, min_size=5)
        assert size >= 5
        assert size % 2 == 1
        assert size == 25  # 80 * 0.3 = 24 → 25 (odd)

    def test_clamps_to_min(self):
        """Tiny boxes get min_size at minimum."""
        size = compute_roi_size(10, 10, roi_ratio=0.3, min_size=5)
        assert size == 5

    def test_even_gets_bumped_to_odd(self):
        """If computed size is even, bump to odd."""
        size = compute_roi_size(20, 20, roi_ratio=0.5, min_size=3)
        assert size == 11  # 10 → 11

    def test_horizontal_box(self):
        """ROI uses smaller dimension, both orientations give same size."""
        size_w = compute_roi_size(200, 50, roi_ratio=0.3)
        size_h = compute_roi_size(50, 200, roi_ratio=0.3)
        # Both use min(200,50)=50, 50*0.3=15 (already odd)
        assert size_w == 15
        assert size_h == 15


# ---------------------------------------------------------------------------
# extract_roi
# ---------------------------------------------------------------------------

class TestExtractRoi:
    def test_centered_extraction(self):
        """ROI centered, fully within bounds."""
        img = np.arange(100).reshape(10, 10).astype(np.uint16)
        roi = extract_roi(img, 5, 5, 5)
        assert roi is not None
        assert roi.shape == (5, 5)
        # Center pixel should be 55
        assert roi[2, 2] == 55

    def test_edge_clamping(self):
        """ROI near edge is clamped to image bounds."""
        img = np.ones((10, 10), dtype=np.uint16) * 100
        roi = extract_roi(img, 0, 0, 5)
        assert roi is not None
        # At (0,0) with half=2, we get [0:3, 0:3]
        assert roi.shape == (3, 3)

    def test_out_of_bounds_returns_none(self):
        """ROI completely outside image returns None."""
        img = np.ones((10, 10), dtype=np.uint16) * 100
        roi = extract_roi(img, -20, -20, 5)
        assert roi is None

    def test_large_roi_clamps_to_full_image(self):
        """ROI larger than image returns full image."""
        img = np.ones((10, 10), dtype=np.uint16) * 100
        roi = extract_roi(img, 5, 5, 99)
        assert roi is not None
        assert roi.shape == (10, 10)


# ---------------------------------------------------------------------------
# filter_depth_roi
# ---------------------------------------------------------------------------

class TestFilterDepthRoi:
    def test_all_valid_pixels(self):
        """All pixels > 0, good quality."""
        roi = np.full((5, 5), 500, dtype=np.uint16)  # 500mm
        depth_m, quality = filter_depth_roi(roi, depth_scale=0.001)
        assert depth_m == pytest.approx(0.5)
        assert quality > 0.8  # near-perfect quality

    def test_all_zero_returns_zero_quality(self):
        """All pixels = 0, no valid depth."""
        roi = np.zeros((5, 5), dtype=np.uint16)
        depth_m, quality = filter_depth_roi(roi, depth_scale=0.001)
        assert depth_m == 0.0
        assert quality == 0.0

    def test_below_min_valid_ratio(self):
        """Too few valid pixels → rejected."""
        roi = np.zeros((10, 10), dtype=np.uint16)
        roi[0, 0] = 500
        roi[0, 1] = 510
        depth_m, quality = filter_depth_roi(roi, depth_scale=0.001, min_valid_ratio=0.3)
        assert depth_m == 0.0
        assert quality == 0.0

    def test_outlier_rejection(self):
        """Single outlier far from median is excluded."""
        roi = np.full((5, 5), 500, dtype=np.uint16)
        roi[0, 0] = 50  # outlier (10× lower)
        depth_m, quality = filter_depth_roi(roi, depth_scale=0.001, outlier_sigma=2.0)
        # Median should be ~500mm unaffected by outlier
        assert depth_m == pytest.approx(0.5, abs=0.01)
        assert quality > 0.7

    def test_mixed_depths_with_outliers(self):
        """Realistic scenario: mostly cube at 1m, some background at 3m."""
        roi = np.full((7, 7), 1000, dtype=np.uint16)  # cube at 1m
        roi[5:, :] = 3000  # background
        roi[6, :] = 0      # invalid edge
        depth_m, quality = filter_depth_roi(roi, depth_scale=0.001)
        # Should filter out 3000 as outlier, median ~1000
        assert depth_m == pytest.approx(1.0, abs=0.05)
        assert quality > 0.5

    def test_quality_perfect(self):
        """All identical valid values → quality near 1.0."""
        roi = np.full((10, 10), 2000, dtype=np.uint16)
        _, quality = filter_depth_roi(roi, depth_scale=0.001)
        assert quality >= 0.99

    def test_quality_low_for_noisy_roi(self):
        """High variance → lower quality."""
        roi = np.array([
            [100, 200, 500],
            [200, 500, 800],
            [500, 800, 1000],
        ], dtype=np.uint16)
        _, quality = filter_depth_roi(roi, depth_scale=0.001)
        assert 0.0 < quality < 0.5
