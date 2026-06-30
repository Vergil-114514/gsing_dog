"""Tests for camera-to-arm-base coordinate transform."""

import math

import pytest

from detection_3d.vision_transform import (
    VisionTransformConfig,
    transform_camera_to_arm_base,
)


def test_transform_matches_vision_transform_c_with_zero_base_angle():
    config = VisionTransformConfig(
        camera_offset_x_m=0.105,
        camera_offset_y_m=0.0,
        camera_offset_z_m=-0.078,
        camera_tilt_forward_deg=0.0,
    )

    result = transform_camera_to_arm_base(
        (0.01, 0.02, 0.15),
        config,
        theta1_rad=0.0,
        current_end_xyz_m=(0.2, -0.1, 0.3),
    )

    assert result == pytest.approx((0.325, -0.09, 0.072))


def test_transform_rotates_local_target_by_theta1():
    config = VisionTransformConfig(
        camera_offset_x_m=0.105,
        camera_offset_y_m=0.0,
        camera_offset_z_m=-0.078,
        camera_tilt_forward_deg=0.0,
    )

    result = transform_camera_to_arm_base(
        (0.01, 0.02, 0.15),
        config,
        theta1_rad=math.pi / 2.0,
        current_end_xyz_m=(0.2, -0.1, 0.3),
    )

    assert result == pytest.approx((0.19, 0.025, 0.072))


def test_transform_applies_forward_camera_tilt():
    config = VisionTransformConfig(
        camera_offset_x_m=0.105,
        camera_offset_y_m=0.0,
        camera_offset_z_m=-0.078,
        camera_tilt_forward_deg=45.0,
    )

    result = transform_camera_to_arm_base(
        (0.01, 0.02, 0.15),
        config,
        theta1_rad=0.0,
        current_end_xyz_m=(0.2, -0.1, 0.3),
    )

    expected_x = 0.2 + 0.105 + (0.02 * math.sqrt(0.5)) + (0.15 * math.sqrt(0.5))
    expected_y = -0.1 + 0.01
    expected_z = 0.3 - 0.078 + (0.02 * math.sqrt(0.5)) - (0.15 * math.sqrt(0.5))
    assert result == pytest.approx((expected_x, expected_y, expected_z))


def test_transform_rotates_tilted_local_target_by_theta1():
    config = VisionTransformConfig(
        camera_offset_x_m=0.105,
        camera_offset_y_m=0.0,
        camera_offset_z_m=-0.078,
        camera_tilt_forward_deg=45.0,
    )

    result = transform_camera_to_arm_base(
        (0.01, 0.02, 0.15),
        config,
        theta1_rad=math.pi / 2.0,
        current_end_xyz_m=(0.2, -0.1, 0.3),
    )

    local_x = 0.105 + (0.02 * math.sqrt(0.5)) + (0.15 * math.sqrt(0.5))
    local_y = 0.01
    local_z = -0.078 + (0.02 * math.sqrt(0.5)) - (0.15 * math.sqrt(0.5))
    assert result == pytest.approx((0.2 - local_y, -0.1 + local_x, 0.3 + local_z))


def test_transform_uses_current_end_xyz_from_call_argument():
    config = VisionTransformConfig(
        camera_offset_x_m=0.105,
        camera_offset_y_m=0.0,
        camera_offset_z_m=-0.078,
        camera_tilt_forward_deg=45.0,
    )

    first = transform_camera_to_arm_base(
        (0.01, 0.02, 0.15),
        config,
        theta1_rad=0.0,
        current_end_xyz_m=(0.2, -0.1, 0.3),
    )
    second = transform_camera_to_arm_base(
        (0.01, 0.02, 0.15),
        config,
        theta1_rad=0.0,
        current_end_xyz_m=(0.4, 0.2, 0.1),
    )

    assert second[0] - first[0] == pytest.approx(0.2)
    assert second[1] - first[1] == pytest.approx(0.3)
    assert second[2] - first[2] == pytest.approx(-0.2)
