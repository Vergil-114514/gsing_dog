"""Unit tests for target_selector module."""

import pytest
from detection_3d.target_selector import (
    TargetCandidate,
    compute_composite_score,
    select_best_target,
    detect_coordinate_jump,
)


class TestComputeCompositeScore:
    def test_perfect(self):
        """Both scores 1.0 → composite 1.0."""
        assert compute_composite_score(1.0, 1.0) == pytest.approx(1.0)

    def test_weighted_formula(self):
        """Verify 0.6 × conf + 0.4 × quality."""
        assert compute_composite_score(0.8, 0.5) == pytest.approx(0.6 * 0.8 + 0.4 * 0.5)

    def test_both_zero(self):
        assert compute_composite_score(0.0, 0.0) == 0.0


class TestSelectBestTarget:
    def test_empty_returns_none(self):
        assert select_best_target([]) is None

    def test_single_returns_it(self):
        c = TargetCandidate(0, 0, 1, "Cube_food", 0.9, 0.8, 0.86)
        result = select_best_target([c])
        assert result is c

    def test_highest_composite_wins(self):
        c1 = TargetCandidate(0, 0, 1, "Cube_food", 0.9, 0.9, 0.9)
        c2 = TargetCandidate(0, 0, 1, "Cube_ins", 0.95, 0.95, 0.95)
        c3 = TargetCandidate(0, 0, 1, "Cube_medicine", 0.5, 0.5, 0.5)
        result = select_best_target([c1, c2, c3])
        assert result is c2  # highest composite

    def test_same_score_first_wins(self):
        c1 = TargetCandidate(0, 0, 1, "Cube_food", 0.8, 0.8, 0.8)
        c2 = TargetCandidate(0, 0, 1, "Cube_ins", 0.8, 0.8, 0.8)
        result = select_best_target([c1, c2])
        assert result is c1  # max() returns first on tie


class TestDetectCoordinateJump:
    def test_no_previous(self):
        """First frame, no previous → no jump."""
        assert not detect_coordinate_jump((0.1, 0.2, 0.5), None, 0.05)

    def test_small_movement(self):
        """Tiny movement within threshold → no jump."""
        assert not detect_coordinate_jump(
            (0.101, 0.200, 0.500),
            (0.100, 0.200, 0.500),
            0.05,
        )

    def test_large_jump_detected(self):
        """Movement exceeds threshold → jump."""
        assert detect_coordinate_jump(
            (0.2, 0.2, 0.5),
            (0.1, 0.2, 0.5),
            0.05,
        )

    def test_exactly_at_threshold(self):
        """Distance exactly equals threshold → not a jump (> not >=)."""
        assert not detect_coordinate_jump(
            (0.15, 0.2, 0.5),
            (0.10, 0.2, 0.5),
            0.05,
        )
