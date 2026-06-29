import struct

import pytest

from detection_3d.protocol import (
    ARM_STATE_MOVING,
    FUNC_ARM_FEEDBACK,
    FUNC_ARM_TARGET,
    TARGET_TYPE_GRASP,
    TARGET_TYPE_PLACE,
    build_frame,
    pack_arm_target,
    parse_arm_feedback,
)


def test_build_frame_checksum():
    payload = struct.pack('<fff', 1.0, 2.0, 3.0)
    frame = build_frame(0x12, payload)
    assert frame[0] == 0x55
    assert frame[1] == 0xAA
    assert frame[2] == 0x12
    assert frame[3] == 12
    expected_checksum = sum(frame[:-1]) & 0xFF
    assert frame[-1] == expected_checksum


def test_build_frame_empty_payload():
    frame = build_frame(0x12, b'')
    assert len(frame) == 5
    assert frame[3] == 0


def test_build_frame_checksum_zero_sum():
    payload = bytes([0xFF, 0xFF, 0xFF])
    frame = build_frame(0x12, payload)
    expected = (0x55 + 0xAA + 0x12 + 3 + 0xFF + 0xFF + 0xFF) & 0xFF
    assert frame[-1] == expected


def test_pack_arm_target_payload_and_frame_length():
    frame = pack_arm_target(TARGET_TYPE_GRASP, 1.0, 2.0, 3.0)
    assert len(frame) == 18
    assert frame[2] == FUNC_ARM_TARGET
    assert frame[3] == 13


def test_pack_arm_target_roundtrip():
    x, y, z = 0.5, -0.25, 1.75
    frame = pack_arm_target(TARGET_TYPE_PLACE, x, y, z)
    target_type, rx, ry, rz = struct.unpack('<Bfff', frame[4:-1])

    assert target_type == TARGET_TYPE_PLACE
    assert rx == pytest.approx(x)
    assert ry == pytest.approx(y)
    assert rz == pytest.approx(z)


def test_pack_arm_target_rejects_unknown_type():
    with pytest.raises(ValueError, match='unsupported arm target type'):
        pack_arm_target(99, 1.0, 2.0, 3.0)


def test_parse_arm_feedback_roundtrip():
    payload = struct.pack('<Bffff', ARM_STATE_MOVING, 0.1, -0.2, 0.3, 1.57)

    feedback = parse_arm_feedback(payload)

    assert feedback.arm_state == ARM_STATE_MOVING
    assert feedback.end_xyz_m == pytest.approx((0.1, -0.2, 0.3))
    assert feedback.theta1_rad == pytest.approx(1.57)


def test_parse_arm_feedback_invalid_length():
    with pytest.raises(ValueError, match='Expected 17-byte payload'):
        parse_arm_feedback(b'')
    with pytest.raises(ValueError, match='Expected 17-byte payload'):
        parse_arm_feedback(bytes([FUNC_ARM_FEEDBACK]))
