"""Forward kinematics for the 4-axis arm.

Converts joint angles to camera-frame -> arm_base transform.

The camera is mounted near the end effector. This module computes the
transform from camera optical frame to arm base frame, given the four
joint angles and the known base (static) offset.

=== PLACEHOLDER ===
Replace this function with your actual arm FK once DH parameters /
link lengths are measured. Current implementation returns the base
offset unchanged, so the TF defaults to camera_frame -> arm_base
via the static camera_to_arm_xyz/rpy params only.
"""

import math


def joints_to_camera_arm_transform(
    j1: float, j2: float, j3: float, j4: float,
    base_xyz: tuple[float, float, float],
    base_rpy: tuple[float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Compute camera_frame -> arm_base transform from joint angles.

    Args:
        j1..j4: Joint angles in radians.
        base_xyz: Base translation (camera_to_arm_xyz) in meters.
        base_rpy: Base rotation (camera_to_arm_rpy) in radians.

    Returns:
        (x, y, z, roll, pitch, yaw) — translation in meters,
        rotation in radians.
    """
    # TODO: implement actual forward kinematics.
    # For now, return the base offset unchanged.
    return (*base_xyz, *base_rpy)
