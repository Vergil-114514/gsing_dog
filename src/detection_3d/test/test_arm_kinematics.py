import pytest
from detection_3d.arm_kinematics import joints_to_camera_arm_transform


BASE_XYZ = (0.15, 0.0, 0.05)
BASE_RPY = (0.0, 0.0, 0.0)


def test_placeholder_returns_base_offset():
    x, y, z, r, p, yw = joints_to_camera_arm_transform(
        0.0, 0.0, 0.0, 0.0, BASE_XYZ, BASE_RPY
    )
    assert x == pytest.approx(0.15)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.05)
    assert r == pytest.approx(0.0)
    assert p == pytest.approx(0.0)
    assert yw == pytest.approx(0.0)


def test_placeholder_ignores_joint_angles():
    # Placeholder: joint angles don't affect output
    a = joints_to_camera_arm_transform(1.0, 2.0, 3.0, 4.0, BASE_XYZ, BASE_RPY)
    b = joints_to_camera_arm_transform(0.0, 0.0, 0.0, 0.0, BASE_XYZ, BASE_RPY)
    assert a == b


def test_placeholder_respects_rpy():
    x, y, z, r, p, yw = joints_to_camera_arm_transform(
        0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), (1.57, 0.0, 0.0)
    )
    assert r == pytest.approx(1.57)
    assert p == pytest.approx(0.0)
    assert yw == pytest.approx(0.0)
