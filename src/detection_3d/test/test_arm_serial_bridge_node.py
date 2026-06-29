import pytest

from detection_3d.arm_serial_bridge_node import ArmSerialBridgeNode, BridgeState
from detection_3d.protocol import (
    ARM_STATE_ERROR,
    ARM_STATE_MOVING,
    ArmFeedback,
    FUNC_PUMP_CONTROL,
    PUMP_OFF,
    PUMP_ON,
    TARGET_TYPE_GRASP,
    TARGET_TYPE_PLACE,
)
from detection_3d.vision_transform import VisionTransformConfig


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass

    def debug(self, *_args, **_kwargs):
        pass


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
    node._latest_feedback = None
    node._last_feedback_time = None
    node._active_target = None
    node._hold_target = None
    node._reach_count = 0
    node._delay_start_time = None
    node._pump_command_sent = False
    node._last_feedback_warn_time = 0.0
    node._last_serial_reopen_time = 0.0
    node.read_feedback = False
    node.last_send_time = -100.0
    node.last_sent_target = None
    node.max_send_rate = 1000.0
    node.reach_tolerance_m = 0.015
    node.reach_stable_frames = 3
    node.arrival_delay_sec = 1.0
    node.feedback_timeout_sec = 0.5
    node.target_class = ''
    node.camera_to_arm_transform_enabled = True
    node.vision_transform = VisionTransformConfig(
        camera_offset_x_m=0.105,
        camera_offset_y_m=0.0,
        camera_offset_z_m=-0.078,
    )
    node._place_target = (-0.1, 0.111, 0.42)
    node.ema = _IdentityFilter()
    node.stability = _IdentityFilter()
    node._sent = []
    node._write_frame = lambda frame: True
    node.get_logger = lambda: _Logger()
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
    assert node._bridge_state == BridgeState.SEND_GRASP


def test_grasp_send_uses_latest_feedback_transform(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
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
    assert sent[0][0] == pytest.approx((0.325, -0.09, 0.072))
    assert sent[0][1] == TARGET_TYPE_GRASP
    assert sent[0][2] == 'grasp'


def test_grasp_target_not_sent_when_feedback_is_stale(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
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
        end_xyz_m=(0.325, -0.09, 0.072),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0
    monkeypatch.setattr(node, '_send_target_immediate', lambda *args, **kwargs: True)

    node._maybe_send_grasp(now=10.0)
    node._maybe_send_grasp(now=10.1)

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

    assert sent == [(node._place_target, TARGET_TYPE_PLACE, 'place')]


def test_reaching_place_target_waits_and_resets_to_detection(monkeypatch):
    node = _make_node()
    node.reach_stable_frames = 1
    node._bridge_state = BridgeState.SEND_PLACE
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=node._place_target,
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
    assert node.ema.reset_count == 1
    assert node.stability.reset_count == 1


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
