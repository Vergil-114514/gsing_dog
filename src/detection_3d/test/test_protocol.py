import struct
import pytest
from detection_3d.protocol import (
    build_frame,
    pack_arm_target_xyz,
    pack_arm_joint_pump_command,
    pack_leg_command,
    pack_suction,
    parse_arm_pose,
    FUNC_ARM_CONTROL,
    FUNC_ARM_JOINT_PUMP,
    FUNC_ARM_POSE,
    FUNC_ARM_TARGET_XYZ,
    FUNC_LEG_CONTROL,
    FUNC_ROBOT_STATE,
    FUNC_SUCTION_CONTROL,
)


def test_build_frame_checksum():
    payload = struct.pack('<fff', 1.0, 2.0, 3.0)
    frame = build_frame(0x12, payload)
    # Header: [0x55, 0xAA, func_id, len_payload]
    assert frame[0] == 0x55
    assert frame[1] == 0xAA
    assert frame[2] == 0x12
    assert frame[3] == 12  # 3 * float32
    # Checksum is last byte
    expected_checksum = sum(frame[:-1]) & 0xFF
    assert frame[-1] == expected_checksum


def test_pack_arm_target_xyz_payload_length():
    frame = pack_arm_target_xyz(1.0, 2.0, 3.0)
    # 4 header + 12 payload + 1 checksum = 17 bytes
    assert len(frame) == 17
    assert frame[2] == FUNC_ARM_CONTROL
    assert frame[3] == 12


def test_pack_arm_target_xyz_roundtrip():
    x, y, z = 0.5, -0.25, 1.75
    frame = pack_arm_target_xyz(x, y, z)
    payload = frame[4:16]
    rx, ry, rz = struct.unpack('<fff', payload)
    assert rx == pytest.approx(x)
    assert ry == pytest.approx(y)
    assert rz == pytest.approx(z)


def test_pack_suction_on():
    frame = pack_suction(True)
    assert frame[2] == FUNC_SUCTION_CONTROL
    assert frame[3] == 1
    assert frame[4] == 1  # payload


def test_pack_suction_off():
    frame = pack_suction(False)
    assert frame[2] == FUNC_SUCTION_CONTROL
    assert frame[3] == 1
    assert frame[4] == 0  # payload


def test_build_frame_empty_payload():
    frame = build_frame(0x12, b'')
    assert len(frame) == 5  # 4 header + 1 checksum
    assert frame[3] == 0


def test_build_frame_checksum_zero_sum():
    """Verify checksum wraps at 0xFF correctly."""
    # Choose bytes so sum wraps past 255
    payload = bytes([0xFF, 0xFF, 0xFF])
    frame = build_frame(0x12, payload)
    expected = (0x55 + 0xAA + 0x12 + 3 + 0xFF + 0xFF + 0xFF) & 0xFF
    assert frame[-1] == expected


# ---- func 0x10: pack_leg_command ----

def test_pack_leg_command_func_id():
    positions = [0.0] * 12
    velocities = [0.0] * 12
    frame = pack_leg_command(positions, velocities, 0)
    assert frame[2] == FUNC_LEG_CONTROL


def test_pack_leg_command_payload_length():
    positions = [0.0] * 12
    velocities = [0.0] * 12
    frame = pack_leg_command(positions, velocities, 0)
    # 12×4 + 12×4 + 1 = 97 bytes
    assert frame[3] == 97


def test_pack_leg_command_frame_length():
    positions = [0.0] * 12
    velocities = [0.0] * 12
    frame = pack_leg_command(positions, velocities, 0)
    # 4 header + 97 payload + 1 checksum = 102
    assert len(frame) == 102


def test_pack_leg_command_roundtrip():
    positions = [float(i) * 0.1 for i in range(12)]
    velocities = [float(i) * 0.01 for i in range(12)]
    control_mode = 2
    frame = pack_leg_command(positions, velocities, control_mode)
    payload = frame[4:-1]
    fmt = '<12f12fB'
    unpacked = struct.unpack(fmt, payload)
    assert unpacked[:12] == pytest.approx(tuple(positions))
    assert unpacked[12:24] == pytest.approx(tuple(velocities))
    assert unpacked[24] == control_mode


def test_pack_leg_command_checksum():
    positions = [0.0] * 12
    velocities = [0.0] * 12
    frame = pack_leg_command(positions, velocities, 0)
    expected = sum(frame[:-1]) & 0xFF
    assert frame[-1] == expected


# ---- func 0x11: pack_arm_joint_pump_command ----

def test_pack_arm_joint_pump_func_id():
    joints = [0.0] * 6
    frame = pack_arm_joint_pump_command(joints, False, 0.0)
    assert frame[2] == FUNC_ARM_JOINT_PUMP


def test_pack_arm_joint_pump_payload_length():
    joints = [0.0] * 6
    frame = pack_arm_joint_pump_command(joints, False, 0.0)
    # 6×4 + 1 + 4 = 29 bytes
    assert frame[3] == 29


def test_pack_arm_joint_pump_frame_length():
    frame = pack_arm_joint_pump_command([0.0] * 6, False, 0.0)
    # 4 header + 29 payload + 1 checksum = 34
    assert len(frame) == 34


def test_pack_arm_joint_pump_roundtrip():
    joints = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    pump_on = True
    pressure = 1.5
    frame = pack_arm_joint_pump_command(joints, pump_on, pressure)
    payload = frame[4:-1]
    unpacked = struct.unpack('<6fBf', payload)
    assert unpacked[:6] == pytest.approx(tuple(joints))
    assert unpacked[6] == 1  # pump_on → 1
    assert unpacked[7] == pytest.approx(pressure)


def test_pack_arm_joint_pump_pump_off():
    frame = pack_arm_joint_pump_command([0.0] * 6, False, 0.0)
    payload = frame[4:-1]
    unpacked = struct.unpack('<6fBf', payload)
    assert unpacked[6] == 0


def test_pack_arm_joint_pump_checksum():
    frame = pack_arm_joint_pump_command([0.0] * 6, False, 0.0)
    expected = sum(frame[:-1]) & 0xFF
    assert frame[-1] == expected


# ---- func ID constants ----

def test_func_ids_are_distinct():
    ids = {
        FUNC_LEG_CONTROL, FUNC_ARM_JOINT_PUMP,
        FUNC_ARM_TARGET_XYZ, FUNC_SUCTION_CONTROL,
        FUNC_ROBOT_STATE, FUNC_ARM_POSE,
    }
    assert len(ids) == 6


def test_backwards_compat_alias():
    assert FUNC_ARM_CONTROL == FUNC_ARM_TARGET_XYZ


# ---- func 0x21: parse_arm_pose ----

def test_parse_arm_pose_joints():
    import struct
    payload = struct.pack('<4fB', 0.1, -0.2, 0.3, 1.5, 0)
    joints, flag = parse_arm_pose(payload)
    assert joints == pytest.approx([0.1, -0.2, 0.3, 1.5])


def test_parse_arm_pose_flag_0():
    import struct
    payload = struct.pack('<4fB', 0.0, 0.0, 0.0, 0.0, 0)
    _, flag = parse_arm_pose(payload)
    assert flag == 0


def test_parse_arm_pose_flag_1():
    import struct
    payload = struct.pack('<4fB', 0.0, 0.0, 0.0, 0.0, 1)
    _, flag = parse_arm_pose(payload)
    assert flag == 1


def test_parse_arm_pose_flag_255():
    import struct
    payload = struct.pack('<4fB', 0.0, 0.0, 0.0, 0.0, 255)
    _, flag = parse_arm_pose(payload)
    assert flag == 255
