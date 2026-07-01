"""Unit tests for coordinate_stabilizer module."""

import pytest
from detection_3d.coordinate_stabilizer import CoordinateStabilizer


class TestCoordinateStabilizer:
    def test_not_ready_before_window_full(self):
        """Returns None until window_size frames collected."""
        s = CoordinateStabilizer(window_size=3, max_variance_m2=1.0)
        assert s.update((0.1, 0.2, 0.5)) is None
        assert s.update((0.1, 0.2, 0.5)) is None
        assert s.window_fill == 2
        assert not s.is_ready

    def test_outputs_when_stable(self):
        """Window full + low variance → returns average."""
        s = CoordinateStabilizer(window_size=3, max_variance_m2=0.0004)
        s.update((0.1, 0.2, 0.5))
        s.update((0.11, 0.21, 0.51))
        result = s.update((0.12, 0.19, 0.49))
        assert result is not None
        x, y, z = result
        assert x == pytest.approx(0.11, abs=0.01)
        assert y == pytest.approx(0.20, abs=0.01)
        assert z == pytest.approx(0.50, abs=0.01)

    def test_average_output_keeps_residual_jitter_visible(self):
        """Average output matches the original stabilizer behavior."""
        s = CoordinateStabilizer(window_size=3, max_variance_m2=0.01)
        s.update((0.1, 0.2, 0.5))
        s.update((0.1, 0.2, 0.5))
        result = s.update((0.16, 0.2, 0.5))
        assert result == pytest.approx((0.12, 0.2, 0.5))

    def test_high_variance_rejected(self):
        """Window full but positions vary too much → None."""
        s = CoordinateStabilizer(window_size=3, max_variance_m2=0.0001)  # very tight
        s.update((0.1, 0.2, 0.5))
        s.update((0.2, 0.3, 0.6))
        result = s.update((0.3, 0.4, 0.7))
        assert result is None

    def test_reset_clears_window(self):
        """Reset empties the buffer."""
        s = CoordinateStabilizer(window_size=3, max_variance_m2=1.0)
        s.update((0.1, 0.2, 0.5))
        s.update((0.1, 0.2, 0.5))
        s.reset()
        assert s.window_fill == 0
        assert not s.is_ready

    def test_sliding_window_behavior(self):
        """Oldest value drops out as new values come in."""
        s = CoordinateStabilizer(window_size=3, max_variance_m2=0.0004)
        # Fill with stable values
        s.update((0.1, 0.2, 0.5))
        s.update((0.1, 0.2, 0.5))
        s.update((0.1, 0.2, 0.5))  # window full → outputs

        # Now add a wild outlier
        result = s.update((10.0, 10.0, 10.0))  # drops oldest 0.1, adds 10.0
        # With values [0.1, 0.1, 10.0], variance is huge → None
        assert result is None

        # Add more stable values to flush out the outlier
        s.update((0.1, 0.2, 0.5))
        s.update((0.1, 0.2, 0.5))
        result = s.update((0.1, 0.2, 0.5))
        assert result is not None  # window is now [0.1, 0.1, 0.1]

    def test_default_window_size(self):
        """Default window_size=5."""
        s = CoordinateStabilizer()
        for _ in range(4):
            assert s.update((0.0, 0.0, 0.0)) is None
        assert s.is_ready is False
        # 5th frame fills the window
        result = s.update((0.0, 0.0, 0.0))
        assert result is not None
        assert s.is_ready is True
