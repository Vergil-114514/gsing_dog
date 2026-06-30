import pytest
from detection_3d.geometry import map_source_pixel_to_depth_pixel, project_pixel_to_xyz


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


def test_map_source_pixel_to_depth_pixel_scales_and_offsets():
    u, v = map_source_pixel_to_depth_pixel(
        source_x=320.0,
        source_y=240.0,
        source_width=640,
        source_height=480,
        depth_width=640,
        depth_height=400,
        offset_x_px=2.0,
        offset_y_px=-3.0,
        clamp_half_size=2,
    )

    assert (u, v) == (322, 197)


def test_map_source_pixel_to_depth_pixel_clamps_to_roi_margin():
    u, v = map_source_pixel_to_depth_pixel(
        source_x=-100.0,
        source_y=999.0,
        source_width=640,
        source_height=480,
        depth_width=640,
        depth_height=400,
        clamp_half_size=5,
    )

    assert (u, v) == (5, 394)


def test_map_source_pixel_to_depth_pixel_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        map_source_pixel_to_depth_pixel(
            source_x=0.0,
            source_y=0.0,
            source_width=0,
            source_height=480,
            depth_width=640,
            depth_height=400,
        )
