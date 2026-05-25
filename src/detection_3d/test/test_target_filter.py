import pytest
from detection_3d.target_filter import distance, EMAFilter, StabilityFilter


def test_distance_same():
    assert distance((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == 0.0


def test_distance_unit():
    assert distance((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == 1.0


def test_distance_3d():
    d = distance((1.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    assert d == pytest.approx(3.0)  # sqrt(1+4+4)


class TestEMAFilter:

    def test_first_update_returns_raw(self):
        ema = EMAFilter(alpha=0.25)
        result = ema.update((1.0, 2.0, 3.0))
        assert result == (1.0, 2.0, 3.0)

    def test_ema_converges_toward_input(self):
        ema = EMAFilter(alpha=0.5)
        ema.update((0.0, 0.0, 0.0))
        result = ema.update((1.0, 0.0, 0.0))
        assert result[0] == pytest.approx(0.5)

    def test_ema_reset(self):
        ema = EMAFilter(alpha=0.5)
        ema.update((1.0, 1.0, 1.0))
        ema.reset()
        result = ema.update((2.0, 2.0, 2.0))
        assert result == (2.0, 2.0, 2.0)

    def test_alpha_clamped(self):
        ema = EMAFilter(alpha=1.5)
        assert ema.alpha == 1.0
        ema = EMAFilter(alpha=-0.5)
        assert ema.alpha == 0.0


class TestStabilityFilter:

    def test_first_frame_no_output(self):
        sf = StabilityFilter(radius_m=0.02, required_frames=3)
        assert sf.update((1.0, 0.0, 0.0)) is None

    def test_stable_after_required_frames(self):
        sf = StabilityFilter(radius_m=0.1, required_frames=3)
        assert sf.update((1.0, 0.0, 0.0)) is None
        assert sf.update((1.0, 0.0, 0.0)) is None
        result = sf.update((1.0, 0.0, 0.0))
        assert result is not None

    def test_resets_on_large_movement(self):
        sf = StabilityFilter(radius_m=0.05, required_frames=3)
        sf.update((1.0, 0.0, 0.0))
        sf.update((1.0, 0.0, 0.0))  # count=2
        # Large jump should reset
        result = sf.update((5.0, 0.0, 0.0))
        assert result is None  # reset, count=1

    def test_reset_method(self):
        sf = StabilityFilter(required_frames=2)
        sf.update((1.0, 0.0, 0.0))
        sf.reset()
        assert sf.update((2.0, 0.0, 0.0)) is None
