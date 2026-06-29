"""STM32 USB CDC binary protocol helpers."""

from dataclasses import dataclass
import struct


# ---- Frame constants ----
HEADER_0 = 0x55
HEADER_1 = 0xAA

# ---- Function IDs ----
FUNC_ARM_TARGET = 0x12       # host -> MCU: target_type + arm-base xyz
FUNC_ARM_FEEDBACK = 0x21     # MCU -> host: state + end xyz + theta1

# Backward-compatible alias for older imports. New code should use
# FUNC_ARM_TARGET and pack_arm_target().
FUNC_ARM_TARGET_XYZ = FUNC_ARM_TARGET

# ---- Target types ----
TARGET_TYPE_GRASP = 0
TARGET_TYPE_PLACE = 1

# ---- Arm states reported by MCU ----
ARM_STATE_IDLE = 0
ARM_STATE_MOVING = 1
ARM_STATE_REACHED = 2
ARM_STATE_ERROR = 3


@dataclass(frozen=True)
class ArmFeedback:
    """Structured MCU feedback used by the host-side arm state machine."""

    arm_state: int
    end_xyz_m: tuple[float, float, float]
    theta1_rad: float


def build_frame(func_id: int, payload: bytes) -> bytes:
    """Build one framed protocol packet with low-8-bit additive checksum."""
    header = bytes([HEADER_0, HEADER_1, func_id, len(payload)])
    frame = header + payload
    checksum = sum(frame) & 0xFF
    return frame + bytes([checksum])


def pack_arm_target(target_type: int, x: float, y: float, z: float) -> bytes:
    """Pack an arm target command in arm_base coordinates.

    target_type: 0 = grasp, 1 = place.
    payload: u8 target_type + 3xf32 xyz, meters (13 bytes).
    """
    if target_type not in (TARGET_TYPE_GRASP, TARGET_TYPE_PLACE):
        raise ValueError(f'unsupported arm target type: {target_type}')
    payload = struct.pack('<Bfff', int(target_type), x, y, z)
    return build_frame(FUNC_ARM_TARGET, payload)


def pack_arm_target_xyz(x: float, y: float, z: float) -> bytes:
    """Pack a grasp arm target in arm_base coordinates.

    Kept for older callers; the main path should use pack_arm_target().
    """
    return pack_arm_target(TARGET_TYPE_GRASP, x, y, z)


def parse_arm_feedback(payload: bytes) -> ArmFeedback:
    """Parse a 0x21 ARM_FEEDBACK payload.

    payload: u8 arm_state + 3xf32 end_xyz_m + f32 theta1_rad (17 bytes).
    """
    expected_len = struct.calcsize('<Bffff')
    if len(payload) != expected_len:
        raise ValueError(
            f'Expected {expected_len}-byte payload for 0x21, got {len(payload)}'
        )
    arm_state, end_x, end_y, end_z, theta1_rad = struct.unpack('<Bffff', payload)
    return ArmFeedback(
        arm_state=int(arm_state),
        end_xyz_m=(float(end_x), float(end_y), float(end_z)),
        theta1_rad=float(theta1_rad),
    )


def parse_arm_flag(payload: bytes) -> int:
    """Parse an old 0x21 flag payload.

    Kept only for compatibility with older tools. The active bridge state
    machine uses parse_arm_feedback().
    """
    if len(payload) != 1:
        raise ValueError(f'Expected 1-byte payload for 0x21, got {len(payload)}')
    return payload[0]
