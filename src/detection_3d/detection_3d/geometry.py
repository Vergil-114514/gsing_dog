"""Coordinate transforms: pixel → 3D camera-frame projection."""


def project_pixel_to_xyz(u: float, v: float, depth: float,
                         fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    """Project a pixel coordinate + depth value into camera-frame 3D (optical frame)."""
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return (x, y, z)
