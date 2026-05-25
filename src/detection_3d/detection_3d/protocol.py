"""STM32 USB CDC binary protocol helpers.

Frame format (all directions):
  [0x55][0xAA][func_id][payload_len][payload...][checksum]

All multi-byte values are little-endian. Coordinates in meters.
"""

import struct


# ---- Function IDs ----
FUNC_ARM_TARGET_XYZ = 0x12   # host -> MCU: camera-frame xyz
FUNC_ARM_FLAG = 0x21         # MCU -> host: 1-byte flag (0=grasp, 1=place)


def build_frame(func_id: int, payload: bytes) -> bytes:
    header = bytes([0x55, 0xAA, func_id, len(payload)])
    frame = header + payload
    checksum = sum(frame) & 0xFF
    return frame + bytes([checksum])


def pack_arm_target_xyz(x: float, y: float, z: float) -> bytes:
    """Pack cartesian arm target in camera frame.

    payload: 3×f32 (x,y,z) camera-frame, meters  (12 bytes)
    """
    return build_frame(FUNC_ARM_TARGET_XYZ, struct.pack('<fff', x, y, z))


def parse_arm_flag(payload: bytes) -> int:
    """Parse a 0x21 flag frame.

    payload: 1-byte flag (0 = grasp, 1 = place)
    raises ValueError if payload length != 1
    """
    if len(payload) != 1:
        raise ValueError(f"Expected 1-byte payload for 0x21, got {len(payload)}")
    return payload[0]
