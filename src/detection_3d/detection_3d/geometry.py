"""Coordinate transforms: ROS optical frame ↔ STM32 camera frame, pixel → 3D."""


def optical_to_stm32_camera(optical: tuple[float, float, float]) -> tuple[float, float, float]:
    """ROS optical (+X right, +Y down, +Z forward) → STM32 (+X right, -Y forward, +Z down)."""
    return (optical[0], -optical[2], optical[1])


def project_pixel_to_xyz(u: float, v: float, depth: float,
                         fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    """Project a pixel coordinate + depth value into camera-frame 3D (optical frame)."""
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return (x, y, z)
