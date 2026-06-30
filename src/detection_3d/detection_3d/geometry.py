"""Coordinate transforms: pixel → 3D camera-frame projection."""

def map_source_pixel_to_depth_pixel(
    source_x: float,
    source_y: float,
    source_width: int,
    source_height: int,
    depth_width: int,
    depth_height: int,
    offset_x_px: float = 0.0,
    offset_y_px: float = 0.0,
    clamp_half_size: int = 0,
) -> tuple[int, int]:
    """
    Map a detector-image pixel into the depth image with a tunable calibration offset.

    RGB detection and depth sampling are produced by different camera streams.
    A small measured pixel offset is the lightest correction for residual
    RGB/depth alignment error after camera registration.

    Args:
        source_x: Pixel x coordinate in the source detector image.
        source_y: Pixel y coordinate in the source detector image.
        source_width: Width of the source detector image.
        source_height: Height of the source detector image.
        depth_width: Width of the depth image.
        depth_height: Height of the depth image.
        offset_x_px: Extra x offset in depth-image pixels.
        offset_y_px: Extra y offset in depth-image pixels.
        clamp_half_size: Border margin needed by later ROI extraction.

    Returns:
        Integer depth-image pixel coordinate after scaling, offset, and clamp.
    """
    if source_width <= 0 or source_height <= 0:
        raise ValueError('source image dimensions must be positive')
    if depth_width <= 0 or depth_height <= 0:
        raise ValueError('depth image dimensions must be positive')

    scale_x = depth_width / source_width
    scale_y = depth_height / source_height
    half_x = max(0, min(int(clamp_half_size), (depth_width - 1) // 2))
    half_y = max(0, min(int(clamp_half_size), (depth_height - 1) // 2))

    u = int(round(source_x * scale_x + offset_x_px))
    v = int(round(source_y * scale_y + offset_y_px))
    u = max(half_x, min(u, depth_width - 1 - half_x))
    v = max(half_y, min(v, depth_height - 1 - half_y))
    return u, v


def project_pixel_to_xyz(u: float, v: float, depth: float,
                         fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    """Project a pixel coordinate + depth value into camera-frame 3D (optical frame)."""
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return (x, y, z)
