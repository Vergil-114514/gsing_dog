"""STM32 USB CDC binary protocol helpers.

Frame format (all directions):
  [0x55][0xAA][func_id][payload_len][payload...][checksum]

All multi-byte values are little-endian. Coordinates in meters, angles in radians.
"""

import struct


# ---- Function IDs (host -> MCU) ----
FUNC_LEG_CONTROL = 0x10
FUNC_ARM_JOINT_PUMP = 0x11
FUNC_ARM_TARGET_XYZ = 0x12
FUNC_SUCTION_CONTROL = 0x13

# Backwards-compatible alias
FUNC_ARM_CONTROL = FUNC_ARM_TARGET_XYZ


# ---- Function IDs (MCU -> host) ----
FUNC_ROBOT_STATE = 0x20
FUNC_ARM_POSE = 0x21       # 4x f32 joint angles + u8 flag


def build_frame(func_id: int, payload: bytes) -> bytes:
    header = bytes([0x55, 0xAA, func_id, len(payload)])
    frame = header + payload
    checksum = sum(frame) & 0xFF
    return frame + bytes([checksum])


# ------------------------------------------------------------------
# Host -> MCU packers
# ------------------------------------------------------------------

def pack_leg_command(
    joint_positions: list[float],
    joint_velocities: list[float],
    control_mode: int,
) -> bytes:
    """Pack leg joint control command.

    payload: 12 x f32 pos + 12 x f32 vel + u8 mode = 97 bytes
    """
    payload = struct.pack(
        '<12f12fB',
        *joint_positions,
        *joint_velocities,
        control_mode,
    )
    return build_frame(FUNC_LEG_CONTROL, payload)


def pack_arm_joint_pump_command(
    joint_positions: list[float],
    pump_on: bool,
    pump_pressure: float,
) -> bytes:
    """Pack 6-DOF arm joint + pump control command.

    payload: 6 x f32 pos + u8 pump + f32 pressure = 29 bytes
    """
    payload = struct.pack(
        '<6fBf',
        *joint_positions,
        1 if pump_on else 0,
        pump_pressure,
    )
    return build_frame(FUNC_ARM_JOINT_PUMP, payload)


def pack_arm_target_xyz(flag: int, x: float, y: float, z: float) -> bytes:
    """Pack cartesian arm target with grasp/place flag.

    flag = 0 → grasp (default)
    flag = 1 → place

    payload: u8 flag + 3×f32 (x,y,z) arm_base, meters  (13 bytes)
    """
    return build_frame(FUNC_ARM_TARGET_XYZ, struct.pack('<Bfff', flag, x, y, z))


def pack_suction(enabled: bool) -> bytes:
    """Pack suction on/off command."""
    return build_frame(FUNC_SUCTION_CONTROL, bytes([1 if enabled else 0]))


# ------------------------------------------------------------------
# MCU -> Host parsers
# ------------------------------------------------------------------

def parse_arm_pose(payload: bytes) -> tuple[list[float], int]:
    """Parse a 0x21 arm-pose frame.

    payload: 4 x f32 joint_angles + u8 flag  (17 bytes)
    returns: (joints: list[4 floats], flag: int)
       flag = 0  grasp mode
       flag = 1  place mode
    """
    joints = list(struct.unpack('<4f', payload[:16]))
    flag = payload[16]
    return joints, flag
