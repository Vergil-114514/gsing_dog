import pytest
from detection_3d.geometry import project_pixel_to_xyz


def test_project_pixel_to_xyz_center():
    x, y, z = project_pixel_to_xyz(320.0, 240.0, 1.5, fx=640.0, fy=640.0, cx=320.0, cy=240.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(1.5)


def test_project_pixel_to_xyz_off_center():
    fx, fy, cx, cy = 640.0, 640.0, 320.0, 240.0
    x, y, z = project_pixel_to_xyz(640.0, 480.0, 2.0, fx, fy, cx, cy)
    assert x == pytest.approx((640.0 - 320.0) * 2.0 / 640.0)  # 1.0
    assert y == pytest.approx((480.0 - 240.0) * 2.0 / 640.0)  # 0.75
    assert z == pytest.approx(2.0)


def test_project_pixel_to_xyz_zero_depth():
    x, y, z = project_pixel_to_xyz(100.0, 100.0, 0.0, 640.0, 640.0, 320.0, 240.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.0)
