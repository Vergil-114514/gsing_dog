"""Camera-to-arm-base coordinate transform used before sending targets to MCU."""

from dataclasses import dataclass
from math import cos, sin


@dataclass(frozen=True)
class VisionTransformConfig:
    """
    Parameters for reproducing the MCU vision coordinate transform.

    Only fixed camera mounting offsets belong in config. The current arm pose
    and theta1 must come from fresh MCU feedback at send time.
    """

    camera_offset_x_m: float = 0.105
    camera_offset_y_m: float = 0.0
    camera_offset_z_m: float = -0.078


def transform_camera_to_arm_base(
    camera_xyz_m: tuple[float, float, float],
    config: VisionTransformConfig,
    theta1_rad: float,
    current_end_xyz_m: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Transform camera-frame target coordinates into arm-base coordinates.

    Args:
        camera_xyz_m: Target coordinate from the 3D detector in meters.
        config: Fixed camera mounting offsets.
        theta1_rad: Current base yaw from MCU feedback.
        current_end_xyz_m: Current end-effector xyz from MCU feedback.

    Returns:
        Arm-base target coordinate in meters, ready for the 0x12 protocol frame.
    """
    cam_x_m, cam_y_m, cam_z_m = camera_xyz_m

    x_e = cam_y_m + config.camera_offset_x_m
    y_e = cam_x_m + config.camera_offset_y_m
    z_e = -cam_z_m + config.camera_offset_z_m

    cos_t1 = cos(theta1_rad)
    sin_t1 = sin(theta1_rad)

    arm_x = (x_e * cos_t1) - (y_e * sin_t1) + current_end_xyz_m[0]
    arm_y = (x_e * sin_t1) + (y_e * cos_t1) + current_end_xyz_m[1]
    arm_z = z_e + current_end_xyz_m[2]

    return (arm_x, arm_y, arm_z)
