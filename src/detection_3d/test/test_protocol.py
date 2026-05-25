import struct
import pytest
from detection_3d.protocol import (
    build_frame,
    pack_arm_target_xyz,
    parse_arm_flag,
    FUNC_ARM_TARGET_XYZ,
    FUNC_ARM_FLAG,
)


def test_build_frame_checksum():
    payload = struct.pack('<fff', 1.0, 2.0, 3.0)
    frame = build_frame(0x12, payload)
    assert frame[0] == 0x55
    assert frame[1] == 0xAA
    assert frame[2] == 0x12
    assert frame[3] == 12  # 3 * float32
    expected_checksum = sum(frame[:-1]) & 0xFF
    assert frame[-1] == expected_checksum


def test_build_frame_empty_payload():
    frame = build_frame(0x12, b'')
    assert len(frame) == 5  # 4 header + 1 checksum
    assert frame[3] == 0


def test_build_frame_checksum_zero_sum():
    payload = bytes([0xFF, 0xFF, 0xFF])
    frame = build_frame(0x12, payload)
    expected = (0x55 + 0xAA + 0x12 + 3 + 0xFF + 0xFF + 0xFF) & 0xFF
    assert frame[-1] == expected


# ---- func 0x12: pack_arm_target_xyz ----

def test_pack_arm_target_xyz_payload_length():
    frame = pack_arm_target_xyz(1.0, 2.0, 3.0)
    # 4 header + 12 payload + 1 checksum = 17 bytes
    assert len(frame) == 17
    assert frame[2] == FUNC_ARM_TARGET_XYZ
    assert frame[3] == 12  # 3×f32


def test_pack_arm_target_xyz_roundtrip():
    x, y, z = 0.5, -0.25, 1.75
    frame = pack_arm_target_xyz(x, y, z)
    payload = frame[4:-1]
    rx, ry, rz = struct.unpack('<fff', payload)
    assert rx == pytest.approx(x)
    assert ry == pytest.approx(y)
    assert rz == pytest.approx(z)


def test_pack_arm_target_xyz_values():
    frame = pack_arm_target_xyz(1.0, 2.0, 3.0)
    payload = frame[4:-1]
    rx, ry, rz = struct.unpack('<fff', payload)
    assert rx == pytest.approx(1.0)
    assert ry == pytest.approx(2.0)
    assert rz == pytest.approx(3.0)


# ---- func 0x21: parse_arm_flag ----

def test_parse_arm_flag_grasp():
    flag = parse_arm_flag(bytes([0]))
    assert flag == 0


def test_parse_arm_flag_place():
    flag = parse_arm_flag(bytes([1]))
    assert flag == 1


def test_parse_arm_flag_invalid_length():
    with pytest.raises(ValueError, match="Expected 1-byte"):
        parse_arm_flag(b'')
    with pytest.raises(ValueError, match="Expected 1-byte"):
        parse_arm_flag(bytes([0, 0]))
