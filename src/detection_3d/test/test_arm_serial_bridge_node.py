import pytest

from detection_3d.arm_serial_bridge_node import ArmSerialBridgeNode, BridgeState
from detection_3d.target_filter import distance
from detection_3d.protocol import (
    ARM_STATE_ERROR,
    ARM_STATE_MOVING,
    ARM_STATE_REACHED,
    ArmFeedback,
    FUNC_PUMP_CONTROL,
    PUMP_OFF,
    PUMP_ON,
    TARGET_TYPE_GRASP,
    TARGET_TYPE_PLACE,
)
from detection_3d.vision_transform import VisionTransformConfig


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, *_args, **_kwargs):
        self.messages.append(('info', _args[0] if _args else ''))

    def warn(self, *_args, **_kwargs):
        self.messages.append(('warn', _args[0] if _args else ''))

    def error(self, *_args, **_kwargs):
        self.messages.append(('error', _args[0] if _args else ''))

    def debug(self, *_args, **_kwargs):
        self.messages.append(('debug', _args[0] if _args else ''))


class _IdentityFilter:
    def __init__(self):
        self.reset_count = 0

    def update(self, value):
        return value

    def reset(self):
        self.reset_count += 1


class _Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class _Hypothesis:
    def __init__(self, class_id='', score=1.0):
        self.class_id = class_id
        self.score = score


class _Result:
    def __init__(self, xyz, class_id='', score=1.0):
        self.hypothesis = _Hypothesis(class_id=class_id, score=score)
        self.pose = type('PoseWithCovariance', (), {})()
        self.pose.pose = type('Pose', (), {})()
        self.pose.pose.position = _Point(*xyz)


class _Detection:
    def __init__(self, xyz, class_id='', score=1.0):
        self.results = [_Result(xyz, class_id=class_id, score=score)]


class _DetectionArray:
    def __init__(self, detections):
        self.detections = detections


def _make_node() -> ArmSerialBridgeNode:
    node = ArmSerialBridgeNode.__new__(ArmSerialBridgeNode)
    node._bridge_state = BridgeState.WAIT_DETECTION
    node._latest_stable_camera_target = None
    node._last_detection_time = None
    node._latest_feedback = None
    node._last_feedback_time = None
    node._active_target = None
    node._hold_target = None
    node._reach_count = 0
    node._mcu_reached_count = 0
    node._stall_count = 0
    node._last_arrival_end_xyz = None
    node._delay_start_time = None
    node._pump_command_sent = False
    node._last_feedback_warn_time = 0.0
    node._last_serial_reopen_time = 0.0
    node.read_feedback = False
    node.last_send_time = -100.0
    node.last_sent_target = None
    node.max_send_rate = 1000.0
    node.reach_tolerance_m = 0.02
    node.reach_stable_frames = 2
    node.arrival_delay_sec = 1.0
    node.feedback_timeout_sec = 0.5
    node.detection_timeout_sec = 0.5
    node.arrival_accept_mcu_reached = True
    node.arrival_stall_enabled = True
    node.arrival_stall_epsilon_m = 0.05
    node.arrival_stall_frames = 4
    node.arrival_stall_max_distance_m = 0.10
    node.target_class = ''
    node.camera_to_arm_transform_enabled = True
    node.vision_transform = VisionTransformConfig(
        camera_offset_x_m=0.105,
        camera_offset_y_m=0.0,
        camera_offset_z_m=-0.078,
    )
    node.command_offset_m = (0.0, 0.0, 0.09)
    node.command_abs_y_offset_m = 0.0
    node.serial_tx_log = True
    node.serial_tx_log_hex = False
    node._place_target = (-0.257, -0.19, 0.3835)
    node._place_target_command = node._place_target
    node.ema = _IdentityFilter()
    node.stability = _IdentityFilter()
    node._sent = []
    node._last_detection_warn_time = 0.0
    node._write_frame = lambda frame: True
    node._logger = _Logger()
    node.get_logger = lambda: node._logger
    return node


class _ClosedSerial:
    is_open = False


def test_read_feedback_reopens_closed_serial_with_throttle(monkeypatch):
    node = _make_node()
    node.read_feedback = True
    node.serial = _ClosedSerial()
    calls = []
    monkeypatch.setattr(node, '_open_serial', lambda: calls.append('open') or False)

    node._read_and_parse_feedback()
    node._read_and_parse_feedback()

    assert calls == ['open']


def test_detection_callback_caches_camera_frame_target_only():
    node = _make_node()

    node.detection_callback(_DetectionArray([_Detection((0.01, 0.02, 0.15))]))

    assert node._latest_stable_camera_target == pytest.approx((0.01, 0.02, 0.15))
    assert node._last_detection_time is not None
    assert node._bridge_state == BridgeState.SEND_GRASP


def test_grasp_target_not_sent_without_detection(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.1, 0.3),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda *args, **kwargs: sent.append((args, kwargs)) or True,
    )

    node._maybe_send_grasp(now=10.1)

    assert sent == []


def test_grasp_send_uses_latest_feedback_transform(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 10.0
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.1, 0.3),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_grasp(now=10.1)

    assert len(sent) == 1
    assert sent[0][0] == pytest.approx((0.425208153, -0.09, 0.220076118))
    assert sent[0][1] == TARGET_TYPE_GRASP
    assert sent[0][2] == 'grasp'


def test_grasp_target_timeout_clears_target_and_waits(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 9.0
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.1, 0.3),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda *args, **kwargs: sent.append((args, kwargs)) or True,
    )

    node._maybe_send_grasp(now=10.0)

    assert sent == []
    assert node._bridge_state == BridgeState.WAIT_DETECTION
    assert node._latest_stable_camera_target is None
    assert node._last_detection_time is None


def test_empty_detection_message_stops_grasp_send():
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 10.0

    node.detection_callback(_DetectionArray([]))

    assert node._bridge_state == BridgeState.WAIT_DETECTION
    assert node._latest_stable_camera_target is None


def test_grasp_target_not_sent_when_feedback_is_stale(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 10.0
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.1, 0.3),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 9.0
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda *args, **kwargs: sent.append((args, kwargs)) or True,
    )

    node._maybe_send_grasp(now=10.0)

    assert sent == []


def test_reaching_grasp_target_enters_delay_then_place(monkeypatch):
    node = _make_node()
    node.reach_stable_frames = 2
    node._bridge_state = BridgeState.SEND_GRASP
    node.vision_transform = VisionTransformConfig(
        camera_offset_x_m=0.0,
        camera_offset_y_m=0.0,
        camera_offset_z_m=0.0,
    )
    node._latest_stable_camera_target = (0.0, 0.0, 0.0)
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.1, 0.3),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0
    node._last_detection_time = 10.0
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_grasp(now=10.0)

    assert len(sent) == 1
    assert sent[0][0] == pytest.approx((0.2, -0.1, 0.39))
    assert sent[0][1:] == (TARGET_TYPE_GRASP, 'grasp')

    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=sent[0][0],
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.1
    node._update_arrival(
        now=10.1,
        target=sent[0][0],
        reached_state=BridgeState.GRASP_DELAY,
    )
    node._update_arrival(
        now=10.2,
        target=sent[0][0],
        reached_state=BridgeState.GRASP_DELAY,
    )

    assert node._bridge_state == BridgeState.GRASP_DELAY

    node._pump_command_sent = True
    node._delay_start_time = 10.0
    node._send_delay_hold(
        now=11.2,
        target_type=TARGET_TYPE_GRASP,
        tag='grasp_hold',
        next_state=BridgeState.SEND_PLACE,
    )

    assert node._bridge_state == BridgeState.SEND_PLACE


def test_mcu_reached_state_can_complete_arrival_when_short_of_target():
    node = _make_node()
    node.reach_stable_frames = 2
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_REACHED,
        end_xyz_m=(0.0, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0
    target = (0.05, 0.0, 0.0)

    node._update_arrival(
        now=10.0,
        target=target,
        reached_state=BridgeState.GRASP_DELAY,
    )
    assert node._bridge_state == BridgeState.SEND_GRASP

    node._update_arrival(
        now=10.1,
        target=target,
        reached_state=BridgeState.GRASP_DELAY,
    )

    assert node._bridge_state == BridgeState.GRASP_DELAY
    assert ('info', 'Arrival detected by mcu_reached') in node._logger.messages


def test_grasp_arrival_is_checked_even_when_target_send_is_skipped(monkeypatch):
    node = _make_node()
    node.reach_stable_frames = 1
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.0, 0.0, 0.0)
    node._last_detection_time = 10.0
    node.vision_transform = VisionTransformConfig(
        camera_offset_x_m=0.0,
        camera_offset_y_m=0.0,
        camera_offset_z_m=0.0,
        camera_tilt_forward_deg=0.0,
    )
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_REACHED,
        end_xyz_m=(0.2, -0.1, 0.3),
        theta1_rad=0.0,
    )
    node._active_target = (0.2, -0.1, 0.39)
    node._hold_target = node._active_target
    node._last_feedback_time = 10.0
    monkeypatch.setattr(node, '_send_target_immediate', lambda *args, **kwargs: False)

    node._maybe_send_grasp(now=10.0)

    assert node._bridge_state == BridgeState.GRASP_DELAY
    assert ('info', 'Arrival detected by mcu_reached') in node._logger.messages


def test_stalled_end_effector_can_complete_arrival_near_target():
    node = _make_node()
    node.arrival_stall_frames = 2
    node._bridge_state = BridgeState.SEND_GRASP
    target = (0.05, 0.0, 0.0)

    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.02, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0
    node._update_arrival(
        now=10.0,
        target=target,
        reached_state=BridgeState.GRASP_DELAY,
    )

    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.021, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.1
    node._update_arrival(
        now=10.1,
        target=target,
        reached_state=BridgeState.GRASP_DELAY,
    )
    assert node._bridge_state == BridgeState.SEND_GRASP

    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.0215, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.2
    node._update_arrival(
        now=10.2,
        target=target,
        reached_state=BridgeState.GRASP_DELAY,
    )

    assert node._bridge_state == BridgeState.GRASP_DELAY
    assert ('info', 'Arrival detected by stall') in node._logger.messages


def test_stalled_end_effector_too_far_from_target_does_not_arrive():
    node = _make_node()
    node.arrival_stall_frames = 2
    node._bridge_state = BridgeState.SEND_GRASP
    target = (0.20, 0.0, 0.0)

    for now in (10.0, 10.1, 10.2):
        node._latest_feedback = ArmFeedback(
            arm_state=ARM_STATE_MOVING,
            end_xyz_m=(0.0, 0.0, 0.0),
            theta1_rad=0.0,
        )
        node._last_feedback_time = now
        node._update_arrival(
            now=now,
            target=target,
            reached_state=BridgeState.GRASP_DELAY,
        )

    assert node._bridge_state == BridgeState.SEND_GRASP


def test_arrival_feedback_timeout_resets_extended_arrival_counts():
    node = _make_node()
    node.reach_stable_frames = 1
    node.arrival_stall_frames = 1
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_REACHED,
        end_xyz_m=(0.02, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 9.0
    node._mcu_reached_count = 1
    node._stall_count = 1
    node._last_arrival_end_xyz = (0.02, 0.0, 0.0)

    node._update_arrival(
        now=10.0,
        target=(0.05, 0.0, 0.0),
        reached_state=BridgeState.GRASP_DELAY,
    )

    assert node._bridge_state == BridgeState.SEND_GRASP
    assert node._mcu_reached_count == 0
    assert node._stall_count == 0
    assert node._last_arrival_end_xyz is None


def test_place_arrival_can_use_stall_detection():
    node = _make_node()
    node.arrival_stall_frames = 2
    node._bridge_state = BridgeState.SEND_PLACE
    target = node._place_target_command
    near_place = (target[0] + 0.03, target[1], target[2])

    for now, x_offset in ((10.0, 0.03), (10.1, 0.029), (10.2, 0.0285)):
        node._latest_feedback = ArmFeedback(
            arm_state=ARM_STATE_MOVING,
            end_xyz_m=(target[0] + x_offset, target[1], target[2]),
            theta1_rad=0.0,
        )
        node._last_feedback_time = now
        node._update_arrival(
            now=now,
            target=target,
            reached_state=BridgeState.PLACE_DELAY,
        )

    assert distance(near_place, target) <= node.arrival_stall_max_distance_m
    assert node._bridge_state == BridgeState.PLACE_DELAY


def test_place_arrival_is_checked_even_when_target_send_fails(monkeypatch):
    node = _make_node()
    node.arrival_stall_frames = 2
    node._bridge_state = BridgeState.SEND_PLACE
    target = node._place_target_command
    node._active_target = target
    node._hold_target = target
    monkeypatch.setattr(node, '_send_target_immediate', lambda *args, **kwargs: False)

    for now, x_offset in ((10.0, 0.03), (10.1, 0.029), (10.2, 0.0285)):
        node._latest_feedback = ArmFeedback(
            arm_state=ARM_STATE_MOVING,
            end_xyz_m=(target[0] + x_offset, target[1], target[2]),
            theta1_rad=0.0,
        )
        node._last_feedback_time = now
        node._maybe_send_place(now=now)

    assert node._bridge_state == BridgeState.PLACE_DELAY
    assert ('info', 'Arrival detected by stall') in node._logger.messages


def test_grasp_delay_sends_pump_on_once(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.GRASP_DELAY
    node._hold_target = (0.1, 0.2, 0.3)
    frames = []
    monkeypatch.setattr(node, '_write_frame', lambda frame: frames.append(frame) or True)

    node.send_timer_callback()
    node.send_timer_callback()

    pump_frames = [frame for frame in frames if frame[2] == FUNC_PUMP_CONTROL]
    assert len(pump_frames) == 1
    assert pump_frames[0][3] == 1
    assert pump_frames[0][4] == PUMP_ON


def test_place_delay_retries_pump_off_until_success(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.PLACE_DELAY
    node._hold_target = node._place_target
    frames = []
    attempts = {'count': 0}

    def write_frame(frame):
        if frame[2] == FUNC_PUMP_CONTROL:
            attempts['count'] += 1
            frames.append(frame)
            return attempts['count'] >= 2
        return True

    monkeypatch.setattr(node, '_write_frame', write_frame)

    node.send_timer_callback()
    assert node._pump_command_sent is False
    assert node._delay_start_time is None

    node.send_timer_callback()
    assert node._pump_command_sent is True
    assert node._delay_start_time is not None

    pump_frames = [frame for frame in frames if frame[2] == FUNC_PUMP_CONTROL]
    assert len(pump_frames) == 2
    assert pump_frames[-1][4] == PUMP_OFF


def test_delay_timer_starts_after_pump_command_success(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.GRASP_DELAY
    node._hold_target = (0.1, 0.2, 0.3)
    node.arrival_delay_sec = 1.0
    attempts = {'count': 0}

    def write_frame(frame):
        if frame[2] == FUNC_PUMP_CONTROL:
            attempts['count'] += 1
            return attempts['count'] >= 2
        return True

    monkeypatch.setattr(node, '_write_frame', write_frame)

    node.send_timer_callback()
    assert node._delay_start_time is None

    node.send_timer_callback()
    assert node._delay_start_time is not None


def test_place_target_is_not_camera_transformed(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_PLACE
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(99.0, 99.0, 99.0),
        theta1_rad=1.57,
    )
    node._last_feedback_time = 10.0
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_place(now=10.1)

    assert sent == [(node._place_target_command, TARGET_TYPE_PLACE, 'place')]


def test_place_target_command_is_final_coordinate_without_command_offset():
    node = _make_node()
    node.command_offset_m = (0.5, 0.5, 0.09)
    node.command_abs_y_offset_m = 0.25
    node._place_target = (-0.257, -0.19, 0.3835)
    node._place_target_command = node._place_target

    assert node._place_target_command == pytest.approx((-0.257, -0.19, 0.3835))


def test_reaching_place_target_waits_and_resets_to_detection(monkeypatch):
    node = _make_node()
    node.reach_stable_frames = 1
    node._bridge_state = BridgeState.SEND_PLACE
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 9.0
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=node._place_target_command,
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0
    monkeypatch.setattr(node, '_send_target_immediate', lambda *args, **kwargs: True)

    node._maybe_send_place(now=10.0)

    assert node._bridge_state == BridgeState.PLACE_DELAY

    node._pump_command_sent = True
    node._delay_start_time = 10.0
    node._send_delay_hold(
        now=11.2,
        target_type=TARGET_TYPE_PLACE,
        tag='place_hold',
        next_state=BridgeState.WAIT_DETECTION,
    )

    assert node._bridge_state == BridgeState.WAIT_DETECTION
    assert node._latest_stable_camera_target is None
    assert node._last_detection_time is None
    assert node.ema.reset_count == 1
    assert node.stability.reset_count == 1


def test_place_send_ignores_stale_detection(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_PLACE
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 1.0
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_place(now=10.0)

    assert sent == [(node._place_target_command, TARGET_TYPE_PLACE, 'place')]


def test_command_offset_applies_all_axes():
    node = _make_node()
    node.command_offset_m = (0.01, -0.02, 0.09)
    node.command_abs_y_offset_m = 0.0

    assert node._apply_command_offset((0.1, 0.2, 0.3)) == pytest.approx(
        (0.11, 0.18, 0.39)
    )


def test_command_abs_y_offset_increases_signed_magnitude():
    node = _make_node()
    node.command_offset_m = (0.0, 0.0, 0.0)
    node.command_abs_y_offset_m = 0.25

    assert node._apply_command_offset((0.0, 0.2, 0.0)) == pytest.approx(
        (0.0, 0.45, 0.0)
    )
    assert node._apply_command_offset((0.0, -0.2, 0.0)) == pytest.approx(
        (0.0, -0.45, 0.0)
    )
    assert node._apply_command_offset((0.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 0.25, 0.0)
    )


def test_serial_target_log_contains_final_command_values():
    node = _make_node()
    node.serial_tx_log = True
    node.serial_tx_log_hex = True
    frame = bytes([0x55, 0xAA, 0x12, 0x0D, 0x00, 0x24])

    node._log_target_tx(
        frame,
        target=(0.1, -0.2, 0.39),
        target_type=TARGET_TYPE_GRASP,
        tag='grasp',
    )

    assert node._logger.messages[-1] == (
        'info',
        'SERIAL_TX 0x12 ARM_TARGET target=grasp type=0 '
        'x=0.100000 y=-0.200000 z=0.390000 '
        'frame=55 aa 12 0d 00 24',
    )


def test_serial_pump_log_contains_state():
    node = _make_node()
    node.serial_tx_log = True
    node.serial_tx_log_hex = False

    node._log_pump_tx(bytes([0x55, 0xAA, 0x13, 0x01, 0x01, 0x14]), pump_on=True)

    assert node._logger.messages[-1] == (
        'info',
        'SERIAL_TX 0x13 PUMP state=1',
    )


def test_mcu_error_state_stops_sending(monkeypatch):
    node = _make_node()
    node._handle_arm_feedback(bytes([ARM_STATE_ERROR]) + b'\x00' * 16)
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda *args, **kwargs: sent.append((args, kwargs)) or True,
    )

    node.send_timer_callback()

    assert node._bridge_state == BridgeState.ERROR
    assert sent == []
