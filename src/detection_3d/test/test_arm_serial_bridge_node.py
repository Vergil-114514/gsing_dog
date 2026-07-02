import pytest
import struct
import time

from detection_3d.arm_serial_bridge_node import ArmSerialBridgeNode, BridgeState
from detection_3d.target_filter import distance
from detection_3d.protocol import (
    ARM_STATE_REACHED,
    ARM_STATE_MOVING,
    ArmFeedback,
    FUNC_PUMP_CONTROL,
    PUMP_OFF,
    PUMP_ON,
    TARGET_TYPE_GRASP,
    TARGET_TYPE_PLACE,
    build_frame,
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
    node._state_enter_time = time.monotonic()
    node._latest_stable_camera_target = None
    node._last_detection_time = None
    node._latest_feedback = None
    node._last_feedback_time = None
    node._active_target = None
    node._active_target_sent_time = None
    node._active_target_start_end_xyz = None
    node._hold_target = None
    node._grasp_command_history = []
    node._grasp_occlusion_start_time = None
    node._reach_count = 0
    node._mcu_reached_count = 0
    node._last_mcu_reached_feedback_time = None
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
    node.feedback_loss_abort_sec = 3.0
    node.detection_timeout_sec = 0.5
    node.grasp_occlusion_hold_enabled = True
    node.grasp_command_filter_window = 5
    node.grasp_occlusion_timeout_sec = 8.0
    node.arrival_stall_enabled = True
    node.arrival_stall_epsilon_m = 0.015
    node.arrival_stall_frames = 5
    node.arrival_stall_max_distance_m = 0.08
    node.mcu_reached_enabled = True
    node.mcu_reached_stable_frames = 2
    node.mcu_reached_max_distance_m = 0.10
    node.mcu_reached_min_motion_m = 0.01
    node.target_class = ''
    node.camera_to_arm_transform_enabled = True
    node.vision_transform = VisionTransformConfig(
        camera_offset_x_m=0.105,
        camera_offset_y_m=0.0,
        camera_offset_z_m=-0.078,
    )
    node.command_offset_m = (0.0, 0.0, 0.11)
    node.command_abs_y_offset_m = 0.03
    node.serial_tx_log = True
    node.serial_tx_log_hex = False
    node.serial_rx_log = True
    node.serial_rx_log_hex = True
    node._place_targets = [
        (-0.217, -0.22, 0.3835),   # 0: right_rear
        (-0.217, 0.22, 0.3835),    # 1: left_rear
        (0.2407, 0.21014, 0.384),  # 2: left_front
        (0.217, -0.22, 0.3835),    # 3: right_front
    ]
    node._place_target_index = 0
    node._place_target = node._place_targets[0]
    node._place_target_command = node._place_target
    node._locked_place_target = None
    node._locked_place_index = None
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
    assert sent[0][0] == pytest.approx((0.425208153, -0.12, 0.240076118))
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


def test_grasp_target_timeout_holds_filtered_target_after_successful_send(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 9.0
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.0, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0
    node.grasp_command_filter_window = 3
    node._grasp_command_history = [
        (0.10, 0.10, 0.10),
        (0.30, 0.20, 0.20),
        (0.20, 0.30, 0.40),
    ]
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_grasp(now=10.0)

    assert node._bridge_state == BridgeState.SEND_GRASP
    assert sent == [((0.20, 0.20, 0.20), TARGET_TYPE_GRASP, 'grasp_occluded')]
    assert node._active_target == pytest.approx((0.20, 0.20, 0.20))
    assert node._hold_target == pytest.approx((0.20, 0.20, 0.20))


def test_grasp_occlusion_hold_can_reach_delay_by_distance(monkeypatch):
    node = _make_node()
    node.reach_stable_frames = 1
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 9.0
    node._grasp_command_history = [(0.20, 0.10, 0.30)]
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.20, 0.10, 0.30),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0
    monkeypatch.setattr(node, '_send_target_immediate', lambda *args, **kwargs: True)

    node._maybe_send_grasp(now=10.0)

    assert node._bridge_state == BridgeState.GRASP_DELAY
    assert ('info', 'Arrival detected by distance') in node._logger.messages


def test_grasp_occlusion_hold_can_reach_delay_by_stall(monkeypatch):
    node = _make_node()
    node.arrival_stall_frames = 2
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 9.0
    node._grasp_command_history = [(0.20, 0.10, 0.30)]
    monkeypatch.setattr(node, '_send_target_immediate', lambda *args, **kwargs: True)

    for now, end_x in ((10.0, 0.16), (10.1, 0.161), (10.2, 0.162)):
        node._latest_feedback = ArmFeedback(
            arm_state=ARM_STATE_MOVING,
            end_xyz_m=(end_x, 0.10, 0.30),
            theta1_rad=0.0,
        )
        node._last_feedback_time = now
        node._maybe_send_grasp(now=now)

    assert node._bridge_state == BridgeState.GRASP_DELAY
    assert ('info', 'Arrival detected by stall') in node._logger.messages


def test_grasp_occlusion_timeout_returns_to_wait_detection(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 9.0
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.0, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 14.0
    node._grasp_command_history = [(0.20, 0.10, 0.30)]
    node._grasp_occlusion_start_time = 10.0
    node.grasp_occlusion_timeout_sec = 3.0
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda *args, **kwargs: sent.append((args, kwargs)) or True,
    )

    node._maybe_send_grasp(now=14.0)

    assert sent == []
    assert node._bridge_state == BridgeState.WAIT_DETECTION
    assert node._grasp_command_history == []
    assert node._grasp_occlusion_start_time is None


def test_grasp_occlusion_timeout_default_is_eight_seconds():
    node = _make_node()

    assert node.grasp_occlusion_timeout_sec == pytest.approx(8.0)


def test_empty_detection_message_stops_grasp_send():
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 10.0

    node.detection_callback(_DetectionArray([]))

    assert node._bridge_state == BridgeState.WAIT_DETECTION
    assert node._latest_stable_camera_target is None


def test_empty_detection_message_does_not_stop_occluded_grasp_hold():
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 10.0
    node._grasp_command_history = [(0.20, 0.10, 0.30)]

    node.detection_callback(_DetectionArray([]))

    assert node._bridge_state == BridgeState.SEND_GRASP
    assert node._latest_stable_camera_target == pytest.approx((0.01, 0.02, 0.15))


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
    assert sent[0][0] == pytest.approx((0.2, -0.13, 0.41))
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
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.15, 0.53),
        theta1_rad=0.0,
    )
    node._active_target = (0.2, -0.15, 0.53)
    node._active_target_sent_time = 9.9
    node._hold_target = node._active_target
    node._last_feedback_time = 10.0
    monkeypatch.setattr(node, '_send_target_immediate', lambda *args, **kwargs: False)

    node._maybe_send_grasp(now=10.0)

    assert node._bridge_state == BridgeState.GRASP_DELAY
    assert ('info', 'Arrival detected by distance') in node._logger.messages


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


def test_slow_moving_end_effector_does_not_count_as_stalled():
    node = _make_node()
    node.arrival_stall_frames = 2
    node._bridge_state = BridgeState.SEND_GRASP
    target = (0.07, 0.0, 0.0)

    for now, end_x in ((10.0, 0.00), (10.1, 0.02), (10.2, 0.04)):
        node._latest_feedback = ArmFeedback(
            arm_state=ARM_STATE_MOVING,
            end_xyz_m=(end_x, 0.0, 0.0),
            theta1_rad=0.0,
        )
        node._last_feedback_time = now
        node._update_arrival(
            now=now,
            target=target,
            reached_state=BridgeState.GRASP_DELAY,
        )

    assert node._bridge_state == BridgeState.SEND_GRASP
    assert ('info', 'Arrival detected by stall') not in node._logger.messages


def test_mcu_reached_state_requires_stable_near_moved_feedback():
    node = _make_node()
    node.reach_stable_frames = 5
    node.arrival_stall_enabled = False
    node._bridge_state = BridgeState.SEND_GRASP
    target = (0.05, 0.0, 0.0)
    node._active_target = target
    node._active_target_sent_time = 9.9
    node._active_target_start_end_xyz = (0.0, 0.0, 0.0)
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_REACHED,
        end_xyz_m=(0.04, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0

    node._update_arrival(
        now=10.0,
        target=target,
        reached_state=BridgeState.GRASP_DELAY,
    )
    assert node._bridge_state == BridgeState.SEND_GRASP

    node._last_feedback_time = 10.1
    node._update_arrival(
        now=10.1,
        target=target,
        reached_state=BridgeState.GRASP_DELAY,
    )

    assert node._bridge_state == BridgeState.GRASP_DELAY
    assert ('info', 'Arrival detected by mcu_reached') in node._logger.messages


def test_mcu_reached_state_does_not_count_same_feedback_twice():
    node = _make_node()
    node.reach_stable_frames = 5
    node.arrival_stall_enabled = False
    node._bridge_state = BridgeState.SEND_GRASP
    target = (0.05, 0.0, 0.0)
    node._active_target = target
    node._active_target_sent_time = 9.9
    node._active_target_start_end_xyz = (0.0, 0.0, 0.0)
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_REACHED,
        end_xyz_m=(0.04, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0

    for now in (10.0, 10.1, 10.2):
        node._update_arrival(
            now=now,
            target=target,
            reached_state=BridgeState.GRASP_DELAY,
        )

    assert node._mcu_reached_count == 1
    assert node._bridge_state == BridgeState.SEND_GRASP
    assert ('info', 'Arrival detected by mcu_reached') not in node._logger.messages


def test_stale_mcu_reached_state_does_not_complete_new_target(monkeypatch):
    node = _make_node()
    node.reach_stable_frames = 5
    node.arrival_stall_enabled = False
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_stable_camera_target = (0.01, 0.02, 0.15)
    node._last_detection_time = 10.0
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_REACHED,
        end_xyz_m=(0.0, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 9.9
    monkeypatch.setattr(node, '_write_frame', lambda frame: True)

    node._maybe_send_grasp(now=10.0)

    assert node._bridge_state == BridgeState.SEND_GRASP
    assert node._active_target_sent_time == pytest.approx(10.0)
    assert ('info', 'Arrival detected by mcu_reached') not in node._logger.messages


def test_mcu_reached_state_after_target_send_triggers_arrival():
    node = _make_node()
    node.reach_stable_frames = 5
    node.arrival_stall_enabled = False
    node._bridge_state = BridgeState.SEND_GRASP
    target = (0.05, 0.0, 0.0)
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.0, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._activate_target(target, sent_time=10.0)

    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_REACHED,
        end_xyz_m=(0.04, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.2

    node._update_arrival(
        now=10.2,
        target=target,
        reached_state=BridgeState.GRASP_DELAY,
    )
    node._last_feedback_time = 10.3
    node._update_arrival(
        now=10.3,
        target=target,
        reached_state=BridgeState.GRASP_DELAY,
    )

    assert node._bridge_state == BridgeState.GRASP_DELAY
    assert ('info', 'Arrival detected by mcu_reached') in node._logger.messages


def test_mcu_reached_state_too_far_from_target_does_not_arrive():
    node = _make_node()
    node.reach_stable_frames = 5
    node.arrival_stall_enabled = False
    node._bridge_state = BridgeState.SEND_GRASP
    target = (0.20, 0.0, 0.0)
    node._active_target = target
    node._active_target_sent_time = 9.9
    node._active_target_start_end_xyz = (0.0, 0.0, 0.0)

    for now in (10.0, 10.1, 10.2):
        node._latest_feedback = ArmFeedback(
            arm_state=ARM_STATE_REACHED,
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
    assert ('info', 'Arrival detected by mcu_reached') not in node._logger.messages


def test_mcu_reached_state_without_motion_does_not_arrive():
    node = _make_node()
    node.reach_stable_frames = 5
    node.arrival_stall_enabled = False
    node._bridge_state = BridgeState.SEND_GRASP
    target = (0.05, 0.0, 0.0)
    node._active_target = target
    node._active_target_sent_time = 9.9
    node._active_target_start_end_xyz = (0.02, 0.0, 0.0)

    for now in (10.0, 10.1, 10.2):
        node._latest_feedback = ArmFeedback(
            arm_state=ARM_STATE_REACHED,
            end_xyz_m=(0.02, 0.0, 0.0),
            theta1_rad=0.0,
        )
        node._last_feedback_time = now
        node._update_arrival(
            now=now,
            target=target,
            reached_state=BridgeState.GRASP_DELAY,
        )

    assert node._bridge_state == BridgeState.SEND_GRASP
    assert ('info', 'Arrival detected by mcu_reached') not in node._logger.messages


def test_arrival_feedback_timeout_resets_extended_arrival_counts():
    node = _make_node()
    node.reach_stable_frames = 1
    node.arrival_stall_frames = 1
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=2,
        end_xyz_m=(0.02, 0.0, 0.0),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 9.0
    node._stall_count = 1
    node._last_arrival_end_xyz = (0.02, 0.0, 0.0)

    node._update_arrival(
        now=10.0,
        target=(0.05, 0.0, 0.0),
        reached_state=BridgeState.GRASP_DELAY,
    )

    assert node._bridge_state == BridgeState.SEND_GRASP
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


def test_place_delay_sends_pump_off_after_delay_and_retries(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.PLACE_DELAY
    node._hold_target = node._place_target
    node.arrival_delay_sec = 1.0
    frames = []
    attempts = {'count': 0}

    def write_frame(frame):
        if frame[2] == FUNC_PUMP_CONTROL:
            attempts['count'] += 1
            frames.append(frame)
            return attempts['count'] >= 2
        return True

    monkeypatch.setattr(node, '_write_frame', write_frame)

    node._send_place_delay_then_pump_off(now=10.0)
    assert node._pump_command_sent is False
    assert node._delay_start_time == pytest.approx(10.0)

    node._send_place_delay_then_pump_off(now=10.5)
    assert attempts['count'] == 0

    node._send_place_delay_then_pump_off(now=11.1)
    assert node._pump_command_sent is False
    assert node._bridge_state == BridgeState.PLACE_DELAY

    node._send_place_delay_then_pump_off(now=11.2)
    assert node._pump_command_sent is False
    assert node._bridge_state == BridgeState.WAIT_DETECTION

    pump_frames = [frame for frame in frames if frame[2] == FUNC_PUMP_CONTROL]
    assert len(pump_frames) == 2
    assert pump_frames[-1][4] == PUMP_OFF


def test_place_delay_waits_for_successful_hold_target_before_timing(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.PLACE_DELAY
    node._hold_target = node._place_target
    attempts = {'target': 0}

    def write_frame(frame):
        if frame[2] != FUNC_PUMP_CONTROL:
            attempts['target'] += 1
            return attempts['target'] >= 2
        return True

    monkeypatch.setattr(node, '_write_frame', write_frame)

    node._send_place_delay_then_pump_off(now=10.0)
    assert node._delay_start_time is None

    node._send_place_delay_then_pump_off(now=10.5)
    assert node._delay_start_time == pytest.approx(10.5)


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


def test_feedback_loss_abort_requests_pump_off_and_resets(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_PLACE
    node._latest_stable_camera_target = (0.1, 0.2, 0.3)
    node._last_detection_time = 10.0
    node._active_target = (0.2, 0.0, 0.4)
    node._active_target_sent_time = 10.0
    node._active_target_start_end_xyz = (0.0, 0.0, 0.0)
    node._hold_target = node._active_target
    node._grasp_command_history = [(0.2, 0.0, 0.4)]
    node._last_feedback_time = 10.0
    frames = []
    monkeypatch.setattr(node, '_write_frame', lambda frame: frames.append(frame) or True)

    assert node._abort_on_feedback_loss(now=13.1) is True

    assert node._bridge_state == BridgeState.WAIT_DETECTION
    assert node._latest_stable_camera_target is None
    assert node._active_target is None
    assert node._grasp_command_history == []
    pump_frames = [frame for frame in frames if frame[2] == FUNC_PUMP_CONTROL]
    assert pump_frames[-1][4] == PUMP_OFF


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
    node.command_offset_m = (0.5, 0.5, 0.23)
    node.command_abs_y_offset_m = 0.25
    node._place_target = (-0.217, -0.22, 0.3835)
    node._place_target_command = node._place_target

    assert node._place_target_command == pytest.approx((-0.217, -0.22, 0.3835))


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

    pump_frames = []

    def write_frame(frame):
        if frame[2] == FUNC_PUMP_CONTROL:
            pump_frames.append(frame)
        return True

    monkeypatch.setattr(node, '_write_frame', write_frame)

    node._send_place_delay_then_pump_off(now=10.0)
    assert node._bridge_state == BridgeState.PLACE_DELAY
    assert pump_frames == []

    node._send_place_delay_then_pump_off(now=11.2)

    assert node._bridge_state == BridgeState.WAIT_DETECTION
    assert node._latest_stable_camera_target is None
    assert node._last_detection_time is None
    assert node.ema.reset_count == 1
    assert node.stability.reset_count == 1
    assert pump_frames[-1][4] == PUMP_OFF


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
    node.command_offset_m = (0.01, -0.02, 0.23)
    node.command_abs_y_offset_m = 0.0

    assert node._apply_command_offset((0.1, 0.2, 0.3)) == pytest.approx(
        (0.11, 0.18, 0.53)
    )


def test_command_abs_y_offset_changes_signed_magnitude():
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

    node.command_abs_y_offset_m = -0.15
    assert node._apply_command_offset((0.0, 0.2, 0.0)) == pytest.approx(
        (0.0, 0.05, 0.0)
    )
    assert node._apply_command_offset((0.0, -0.2, 0.0)) == pytest.approx(
        (0.0, -0.05, 0.0)
    )
    assert node._apply_command_offset((0.0, 0.1, 0.0)) == pytest.approx(
        (0.0, 0.0, 0.0)
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


def test_arm_feedback_log_contains_parsed_values_and_frame():
    node = _make_node()
    node.serial_rx_log = True
    node.serial_rx_log_hex = True
    payload = struct.pack('<Bffff', 1, 0.1, -0.2, 0.3, 1.57)
    frame = build_frame(0x21, payload)

    node._handle_arm_feedback(payload, frame)

    assert node._latest_feedback is not None
    assert node._logger.messages[-1] == (
        'info',
        'SERIAL_RX 0x21 ARM_FEEDBACK state=1 '
        'end_x=0.100000 end_y=-0.200000 end_z=0.300000 '
        f'theta1=1.570000 frame={frame.hex(" ")}',
    )


def test_arm_feedback_log_can_be_disabled():
    node = _make_node()
    node.serial_rx_log = False
    payload = struct.pack('<Bffff', 1, 0.1, -0.2, 0.3, 1.57)

    node._handle_arm_feedback(payload, build_frame(0x21, payload))

    assert node._latest_feedback is not None
    assert node._logger.messages == []


def test_no_serial_data_does_not_print_rx_log():
    node = _make_node()
    node.read_feedback = True
    node.serial = type(
        'OpenSerial',
        (),
        {'is_open': True, 'read_available': lambda *args, **kwargs: b''},
    )()
    node._rx_buffer = bytearray()

    node._read_and_parse_feedback()

    assert node._logger.messages == []


def test_non_feedback_mcu_frame_log_contains_payload_and_frame():
    node = _make_node()
    node.serial_rx_log = True
    node.serial_rx_log_hex = True
    payload = b'\x01\x02'
    frame = build_frame(0x30, payload)

    node._log_mcu_frame_rx(0x30, payload, frame)

    assert node._logger.messages[-1] == (
        'info',
        f'SERIAL_RX 0x30 payload=01 02 frame={frame.hex(" ")}',
    )

def test_arm_state_error_does_not_stop_host_state_machine(monkeypatch):
    node = _make_node()
    node._bridge_state = BridgeState.SEND_PLACE
    node._handle_arm_feedback(bytes([3]) + b'\x00' * 16)
    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda *args, **kwargs: sent.append((args, kwargs)) or True,
    )

    node._maybe_send_place(now=10.0)

    assert node._bridge_state == BridgeState.SEND_PLACE
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# Dynamic place target selection tests
# ---------------------------------------------------------------------------


def test_grasp_arrival_locks_right_front_when_end_y_negative(monkeypatch):
    """end_y < 0 at grasp arrival → place target locked to right_front (idx 3)."""
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.15, 0.3),  # end_y < 0
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0

    node._set_state(BridgeState.GRASP_DELAY)

    assert node._locked_place_target == pytest.approx((0.217, -0.22, 0.3835))
    assert node._locked_place_index == 3

    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_place(now=10.1)

    assert len(sent) == 1
    assert sent[0][0] == pytest.approx((0.217, -0.22, 0.3835))
    assert sent[0][1] == TARGET_TYPE_PLACE


def test_grasp_arrival_locks_left_rear_when_end_y_positive(monkeypatch):
    """end_y > 0 at grasp arrival → place target locked to left_rear (idx 1)."""
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, 0.15, 0.3),  # end_y > 0
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0

    node._set_state(BridgeState.GRASP_DELAY)

    assert node._locked_place_target == pytest.approx((-0.217, 0.22, 0.3835))
    assert node._locked_place_index == 1

    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_place(now=10.1)

    assert len(sent) == 1
    assert sent[0][0] == pytest.approx((-0.217, 0.22, 0.3835))
    assert sent[0][1] == TARGET_TYPE_PLACE


def test_grasp_arrival_uses_place_target_index_when_end_y_zero(monkeypatch):
    """end_y == 0 → fallback to place_target_index target (idx 0 = right_rear)."""
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, 0.0, 0.3),  # end_y == 0
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0

    node._set_state(BridgeState.GRASP_DELAY)

    assert node._locked_place_target == pytest.approx((-0.217, -0.22, 0.3835))
    assert node._locked_place_index == 0

    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_place(now=10.1)

    assert len(sent) == 1
    assert sent[0][0] == pytest.approx((-0.217, -0.22, 0.3835))


def test_locked_place_target_unchanged_by_subsequent_feedback(monkeypatch):
    """Once locked on GRASP_DELAY, feedback y changes do not switch the target."""
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.15, 0.3),  # end_y < 0 → right_front
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0

    node._set_state(BridgeState.GRASP_DELAY)

    locked_before = node._locked_place_target
    assert locked_before == pytest.approx((0.217, -0.22, 0.3835))

    # Simulate feedback y flipping positive mid-flight
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.3, 0.2, 0.35),  # end_y > 0 now
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.5

    locked_after = node._locked_place_target
    assert locked_after == locked_before

    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_place(now=10.6)

    assert sent[0][0] == pytest.approx((0.217, -0.22, 0.3835))


def test_place_target_locked_cleared_on_wait_detection():
    """Locked target is cleared when state resets to WAIT_DETECTION."""
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.15, 0.3),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0

    node._set_state(BridgeState.GRASP_DELAY)

    assert node._locked_place_target is not None
    assert node._locked_place_index is not None

    node._set_state(BridgeState.SEND_PLACE)
    assert node._locked_place_target is not None  # persists through SEND_PLACE

    node._set_state(BridgeState.PLACE_DELAY)
    assert node._locked_place_target is not None  # persists through PLACE_DELAY

    node._set_state(BridgeState.WAIT_DETECTION)
    assert node._locked_place_target is None
    assert node._locked_place_index is None


def test_place_target_falls_back_when_feedback_is_none(monkeypatch):
    """If _latest_feedback is None at lock time, fallback to place_target_index."""
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = None

    node._set_state(BridgeState.GRASP_DELAY)

    # Should keep fallback (idx 0)
    assert node._locked_place_target is None
    assert node._locked_place_index is None

    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_place(now=10.1)

    assert sent[0][0] == pytest.approx((-0.217, -0.22, 0.3835))


def test_place_target_fewer_than_four_uses_place_target_index(monkeypatch):
    """When place_targets has <4 entries, dynamic selection is disabled."""
    node = _make_node()
    node._place_targets = [(-0.3, -0.3, 0.3), (-0.2, 0.2, 0.35)]
    node._place_target_index = 0
    node._place_target = node._place_targets[0]
    node._place_target_command = node._place_target
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.15, 0.3),  # end_y < 0
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0

    node._set_state(BridgeState.GRASP_DELAY)

    # Should lock to idx 0 despite end_y < 0, because < 4 targets
    assert node._locked_place_target == pytest.approx((-0.3, -0.3, 0.3))
    assert node._locked_place_index == 0
    assert ('warn', "place_targets_m has only 2 target(s); dynamic selection disabled, using place_target_index=0") in node._logger.messages


def test_place_coordinate_not_camera_transformed_with_dynamic_target(monkeypatch):
    """Place target (even dynamically selected) is NOT camera-transformed."""
    node = _make_node()
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.15, 0.3),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0

    node._set_state(BridgeState.GRASP_DELAY)

    # Explicitly test that the locked target is the raw coordinate,
    # not transformed by any camera feedback
    assert node._locked_place_target == pytest.approx((0.217, -0.22, 0.3835))


def test_place_coordinate_no_command_offset_with_dynamic_target(monkeypatch):
    """Dynamic place target does not get command offsets applied."""
    node = _make_node()
    node.command_offset_m = (0.5, 0.5, 0.23)
    node.command_abs_y_offset_m = 0.25
    node._bridge_state = BridgeState.SEND_GRASP
    node._latest_feedback = ArmFeedback(
        arm_state=ARM_STATE_MOVING,
        end_xyz_m=(0.2, -0.15, 0.3),
        theta1_rad=0.0,
    )
    node._last_feedback_time = 10.0

    node._set_state(BridgeState.GRASP_DELAY)

    # Locked target must be the raw coordinate, no offset applied
    assert node._locked_place_target == pytest.approx((0.217, -0.22, 0.3835))

    sent = []
    monkeypatch.setattr(
        node,
        '_send_target_immediate',
        lambda target, target_type, tag, now=None: sent.append(
            (target, target_type, tag)
        ) or True,
    )

    node._maybe_send_place(now=10.1)

    assert sent[0][0] == pytest.approx((0.217, -0.22, 0.3835))
