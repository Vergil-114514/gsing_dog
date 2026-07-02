"""
ROS2 node: USB CDC serial bridge to STM32 MCU.

Host -> MCU:  func 0x12, target_type + arm_base target xyz.
Host -> MCU:  func 0x13, pump switch.
MCU -> Host:  func 0x21, arm state + current end xyz + theta1.
"""

from enum import Enum
import glob
import os
import select
from statistics import median
import time

try:
    import termios
except ImportError:  # pragma: no cover - Windows unit-test environment
    termios = None

try:
    import rclpy
    from rclpy.node import Node
except ImportError:  # pragma: no cover - unit tests without ROS2 installed
    rclpy = None

    class Node:  # type: ignore[no-redef]
        """Placeholder so pure Python state-machine tests can import this file."""

try:
    from vision_msgs.msg import Detection3DArray
except ImportError:  # pragma: no cover - unit tests without ROS2 messages
    Detection3DArray = object  # type: ignore[assignment]

from detection_3d.place_targets import (
    PLACE_TARGET_NAMES,
    get_place_target,
    validate_place_targets,
)
from detection_3d.protocol import (
    ARM_STATE_REACHED,
    FUNC_ARM_FEEDBACK,
    TARGET_TYPE_GRASP,
    TARGET_TYPE_PLACE,
    ArmFeedback,
    pack_arm_target,
    pack_pump_control,
    parse_arm_feedback,
)
from detection_3d.target_filter import EMAFilter, StabilityFilter, distance
from detection_3d.vision_transform import (
    VisionTransformConfig,
    transform_camera_to_arm_base,
)

class BridgeState(Enum):
    """Host-side grasp/place state machine states."""

    WAIT_DETECTION = 'WAIT_DETECTION'
    SEND_GRASP = 'SEND_GRASP'
    GRASP_DELAY = 'GRASP_DELAY'
    SEND_PLACE = 'SEND_PLACE'
    PLACE_DELAY = 'PLACE_DELAY'


# ---------------------------------------------------------------------------
# Serial transport
# ---------------------------------------------------------------------------

class CdcSerial:
    """Small Linux termios wrapper for STM32 USB CDC serial transport."""

    def __init__(self, port: str, baud_rate: int):
        self.requested_port = port
        self.baud_rate = baud_rate
        self.fd = None
        self.port = None

    @property
    def is_open(self) -> bool:
        """Return whether the serial file descriptor is currently open."""
        return self.fd is not None

    def open(self):
        """Open and configure the STM32 CDC serial port."""
        if termios is None:
            raise OSError('termios is unavailable; CDC serial requires Linux')

        self.close()
        self.port = self._resolve_port()
        self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0
        baud = self._termios_baud(self.baud_rate)
        attrs[4] = baud
        attrs[5] = baud
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def close(self):
        """Close the serial file descriptor if it is open."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def write(self, data: bytes) -> bool:
        """Write a complete frame to the serial port."""
        if self.fd is None:
            raise OSError('serial port is not open')
        total = len(data)
        written = 0
        while written < total:
            try:
                n = os.write(self.fd, data[written:])
                if n > 0:
                    written += n
                else:
                    return False
            except BlockingIOError:
                _, w_ready, _ = select.select([], [self.fd], [], 0.1)
                if not w_ready:
                    return False
        return True

    def read_available(self, timeout_sec: float = 0.0, max_bytes: int = 512) -> bytes:
        """Read currently available serial bytes without blocking the ROS timer."""
        if self.fd is None:
            return b''
        readable, _, _ = select.select([self.fd], [], [], timeout_sec)
        if not readable:
            return b''
        try:
            return os.read(self.fd, max_bytes)
        except BlockingIOError:
            return b''

    def _resolve_port(self) -> str:
        if self.requested_port and self.requested_port != 'auto':
            return self.requested_port

        by_id = sorted(glob.glob('/dev/serial/by-id/*STM32*')) + sorted(
            glob.glob('/dev/serial/by-id/*STMicroelectronics*')
        )
        if by_id:
            return os.path.realpath(by_id[0])

        acm = sorted(glob.glob('/dev/ttyACM*'))
        if acm:
            return acm[0]

        raise FileNotFoundError('no STM32 USB CDC device found')

    @staticmethod
    def _termios_baud(baud_rate: int):
        if termios is None:
            raise OSError('termios is unavailable; CDC serial requires Linux')
        baud_name = f'B{baud_rate}'
        if not hasattr(termios, baud_name):
            raise ValueError(f'unsupported baud rate: {baud_rate}')
        return getattr(termios, baud_name)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ArmSerialBridgeNode(Node):
    """Bridge stable vision targets to MCU arm commands over USB CDC."""

    def __init__(self):
        if rclpy is None:
            raise RuntimeError('ArmSerialBridgeNode requires ROS2 rclpy')

        super().__init__('arm_serial_bridge')

        # ---- serial ----
        self.declare_parameter('serial_port', 'auto')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('send_rate', 10.0)

        # ---- detection ----
        self.declare_parameter('detection_topic', '/detection/detections_3d')
        self.declare_parameter('target_class', '')

        # ---- filtering ----
        self.declare_parameter('ema_alpha', 0.25)
        self.declare_parameter('stable_radius_m', 0.02)
        self.declare_parameter('stable_frames', 3)
        self.declare_parameter('max_send_rate', 3.0)
        self.declare_parameter('read_feedback', True)

        # ---- host state-machine timing ----
        self.declare_parameter('reach_tolerance_m', 0.02)
        self.declare_parameter('reach_stable_frames', 2)
        self.declare_parameter('arrival_delay_sec', 1.0)
        self.declare_parameter('feedback_timeout_sec', 0.5)
        self.declare_parameter('feedback_loss_abort_sec', 3.0)
        self.declare_parameter('detection_timeout_sec', 0.5)
        self.declare_parameter('grasp_occlusion_hold_enabled', True)
        self.declare_parameter('grasp_command_filter_window', 5)
        self.declare_parameter('grasp_occlusion_timeout_sec', 8.0)
        self.declare_parameter('arrival_stall_enabled', True)
        self.declare_parameter('arrival_stall_epsilon_m', 0.015)
        self.declare_parameter('arrival_stall_frames', 5)
        self.declare_parameter('arrival_stall_max_distance_m', 0.08)
        self.declare_parameter('mcu_reached_enabled', True)
        self.declare_parameter('mcu_reached_stable_frames', 2)
        self.declare_parameter('mcu_reached_max_distance_m', 0.10)
        self.declare_parameter('mcu_reached_min_motion_m', 0.01)

        # ---- vision camera -> arm base transform ----
        self.declare_parameter('camera_to_arm_transform_enabled', True)
        self.declare_parameter('camera_offset_x_m', 0.105)
        self.declare_parameter('camera_offset_y_m', 0.0)
        self.declare_parameter('camera_offset_z_m', -0.078)
        self.declare_parameter('camera_tilt_forward_deg', 45.0)
        self.declare_parameter('command_offset_x_m', 0.0)
        self.declare_parameter('command_offset_y_m', 0.0)
        self.declare_parameter('command_offset_z_m', 0.11)
        self.declare_parameter('command_abs_y_offset_m', 0.03)
        self.declare_parameter('serial_tx_log', True)
        self.declare_parameter('serial_tx_log_hex', False)
        self.declare_parameter('serial_rx_log', True)
        self.declare_parameter('serial_rx_log_hex', True)

        # ---- place targets ----
        self.declare_parameter('place_targets_m', [0.0, 0.0, 0.0])
        self.declare_parameter('place_target_index', 0)

        # === resolve parameters ===

        port = self.get_parameter('serial_port').value
        baud = int(self.get_parameter('baud_rate').value)
        self.send_rate = float(self.get_parameter('send_rate').value)
        det_topic = self.get_parameter('detection_topic').value
        self.target_class = self.get_parameter('target_class').value
        ema_alpha = float(self.get_parameter('ema_alpha').value)
        stable_radius = float(self.get_parameter('stable_radius_m').value)
        stable_frames = int(self.get_parameter('stable_frames').value)
        self.max_send_rate = float(self.get_parameter('max_send_rate').value)
        self.read_feedback = bool(self.get_parameter('read_feedback').value)
        self.reach_tolerance_m = float(
            self.get_parameter('reach_tolerance_m').value
        )
        self.reach_stable_frames = max(
            1, int(self.get_parameter('reach_stable_frames').value)
        )
        self.arrival_delay_sec = float(
            self.get_parameter('arrival_delay_sec').value
        )
        self.feedback_timeout_sec = float(
            self.get_parameter('feedback_timeout_sec').value
        )
        self.feedback_loss_abort_sec = float(
            self.get_parameter('feedback_loss_abort_sec').value
        )
        self.detection_timeout_sec = float(
            self.get_parameter('detection_timeout_sec').value
        )
        self.grasp_occlusion_hold_enabled = bool(
            self.get_parameter('grasp_occlusion_hold_enabled').value
        )
        self.grasp_command_filter_window = max(
            1, int(self.get_parameter('grasp_command_filter_window').value)
        )
        self.grasp_occlusion_timeout_sec = float(
            self.get_parameter('grasp_occlusion_timeout_sec').value
        )
        self.arrival_stall_enabled = bool(
            self.get_parameter('arrival_stall_enabled').value
        )
        self.arrival_stall_epsilon_m = float(
            self.get_parameter('arrival_stall_epsilon_m').value
        )
        self.arrival_stall_frames = max(
            1, int(self.get_parameter('arrival_stall_frames').value)
        )
        self.arrival_stall_max_distance_m = float(
            self.get_parameter('arrival_stall_max_distance_m').value
        )
        self.mcu_reached_enabled = bool(
            self.get_parameter('mcu_reached_enabled').value
        )
        self.mcu_reached_stable_frames = max(
            1, int(self.get_parameter('mcu_reached_stable_frames').value)
        )
        self.mcu_reached_max_distance_m = float(
            self.get_parameter('mcu_reached_max_distance_m').value
        )
        self.mcu_reached_min_motion_m = float(
            self.get_parameter('mcu_reached_min_motion_m').value
        )
        self.camera_to_arm_transform_enabled = bool(
            self.get_parameter('camera_to_arm_transform_enabled').value
        )
        if not self.camera_to_arm_transform_enabled:
            self.get_logger().warn(
                'camera_to_arm_transform_enabled=false is ignored; '
                '0x12 targets must be sent in arm_base coordinates'
            )
            self.camera_to_arm_transform_enabled = True
        self.vision_transform = VisionTransformConfig(
            camera_offset_x_m=float(self.get_parameter('camera_offset_x_m').value),
            camera_offset_y_m=float(self.get_parameter('camera_offset_y_m').value),
            camera_offset_z_m=float(self.get_parameter('camera_offset_z_m').value),
            camera_tilt_forward_deg=float(
                self.get_parameter('camera_tilt_forward_deg').value
            ),
        )
        self.command_offset_m = (
            float(self.get_parameter('command_offset_x_m').value),
            float(self.get_parameter('command_offset_y_m').value),
            float(self.get_parameter('command_offset_z_m').value),
        )
        self.command_abs_y_offset_m = float(
            self.get_parameter('command_abs_y_offset_m').value
        )
        self.serial_tx_log = bool(self.get_parameter('serial_tx_log').value)
        self.serial_tx_log_hex = bool(self.get_parameter('serial_tx_log_hex').value)
        self.serial_rx_log = bool(self.get_parameter('serial_rx_log').value)
        self.serial_rx_log_hex = bool(self.get_parameter('serial_rx_log_hex').value)

        place_targets_m_raw = self.get_parameter('place_targets_m').value
        place_target_index = int(self.get_parameter('place_target_index').value)

        # === serial ===
        self.serial = CdcSerial(port, baud)
        self._open_serial()

        # === filters ===
        self.ema = EMAFilter(alpha=ema_alpha)
        self.stability = StabilityFilter(
            radius_m=stable_radius, required_frames=stable_frames
        )

        # === place targets ===
        self._place_targets = validate_place_targets(place_targets_m_raw)
        self._place_target_index = place_target_index
        self._place_target = get_place_target(
            self._place_targets, self._place_target_index
        )
        self._place_target_command = self._place_target
        self._locked_place_target: tuple[float, float, float] | None = None
        self._locked_place_index: int | None = None

        # === host-driven state ===
        self._bridge_state = BridgeState.WAIT_DETECTION
        self._state_enter_time = time.monotonic()
        self._latest_stable_camera_target: tuple[float, float, float] | None = None
        self._last_detection_time: float | None = None
        self._latest_feedback: ArmFeedback | None = None
        self._last_feedback_time: float | None = None
        self._active_target: tuple[float, float, float] | None = None
        self._active_target_sent_time: float | None = None
        self._active_target_start_end_xyz: tuple[float, float, float] | None = None
        self._hold_target: tuple[float, float, float] | None = None
        self._grasp_command_history: list[tuple[float, float, float]] = []
        self._grasp_occlusion_start_time: float | None = None
        self._reach_count = 0
        self._mcu_reached_count = 0
        self._last_mcu_reached_feedback_time: float | None = None
        self._stall_count = 0
        self._last_arrival_end_xyz: tuple[float, float, float] | None = None
        self._delay_start_time: float | None = None
        self._pump_command_sent = False
        self._last_detection_warn_time = 0.0
        self._last_feedback_warn_time = 0.0
        self._last_serial_reopen_time = 0.0

        # === rate limiting / tx state ===
        self.last_sent_target = None
        self.last_send_time = 0.0

        # === rx buffer ===
        self._rx_buffer = bytearray()

        # === subscribers ===
        self.sub_det = self.create_subscription(
            Detection3DArray, det_topic, self.detection_callback, 10
        )

        # === timer ===
        period = 1.0 / max(self.send_rate, 0.1)
        self.timer = self.create_timer(period, self.send_timer_callback)

        self.get_logger().info(
            f'Serial bridge ready. '
            f'rate={self.send_rate}Hz, '
            f'max_send_rate={self.max_send_rate}Hz, '
            f'ema_alpha={ema_alpha:.2f}, stable={stable_frames} frames, '
            f'target_class="{self.target_class or "any"}", '
            f'reach_tolerance={self.reach_tolerance_m:.3f}m, '
            f'detection_timeout={self.detection_timeout_sec:.2f}s, '
            f'feedback_timeout={self.feedback_timeout_sec:.2f}s, '
            f'feedback_loss_abort={self.feedback_loss_abort_sec:.2f}s, '
            f'grasp_occlusion_hold={self.grasp_occlusion_hold_enabled}, '
            f'grasp_occlusion_timeout={self.grasp_occlusion_timeout_sec:.2f}s, '
            f'grasp_filter_window={self.grasp_command_filter_window}, '
            f'arrival_stall={self.arrival_stall_enabled}, '
            f'stall_epsilon={self.arrival_stall_epsilon_m:.3f}m, '
            f'stall_frames={self.arrival_stall_frames}, '
            f'stall_max_dist={self.arrival_stall_max_distance_m:.3f}m, '
            f'mcu_reached={self.mcu_reached_enabled}, '
            f'mcu_reached_frames={self.mcu_reached_stable_frames}, '
            f'mcu_reached_max_dist={self.mcu_reached_max_distance_m:.3f}m, '
            f'transform_enabled={self.camera_to_arm_transform_enabled}, '
            f'camera_tilt_forward={self.vision_transform.camera_tilt_forward_deg:.1f}deg, '
            f'command_offset=({self.command_offset_m[0]:.3f}, '
            f'{self.command_offset_m[1]:.3f}, '
            f'{self.command_offset_m[2]:.3f})m, '
            f'command_abs_y_offset={self.command_abs_y_offset_m:.3f}m, '
            f'place_target_index={self._place_target_index}'
        )
        for i, t in enumerate(self._place_targets):
            name = PLACE_TARGET_NAMES.get(i, f'idx_{i}')
            self.get_logger().info(
                f'  place[{i}] {name}: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})m'
            )

    # ------------------------------------------------------------------
    # Serial helpers
    # ------------------------------------------------------------------

    def _open_serial(self) -> bool:
        try:
            self.serial.open()
            self.get_logger().info(
                f'Serial opened: {self.serial.port} @ {self.serial.baud_rate}'
            )
            return True
        except OSError as exc:
            self.get_logger().error(f'Failed to open STM32 CDC: {exc}')
            return False

    def _write_frame(self, frame: bytes) -> bool:
        if not self.serial.is_open and not self._open_serial():
            return False
        try:
            return self.serial.write(frame)
        except OSError as exc:
            self.get_logger().error(
                f'Serial write error: {exc}; will reopen next tick'
            )
            self.serial.close()
            return False

    # ------------------------------------------------------------------
    # Incoming frame parser
    # ------------------------------------------------------------------

    def _read_and_parse_feedback(self):
        if not self.read_feedback:
            return

        if not self.serial.is_open:
            self._try_reopen_serial_for_feedback(time.monotonic())
            return

        raw = self.serial.read_available(timeout_sec=0.0, max_bytes=512)
        if raw:
            self._rx_buffer.extend(raw)

        while True:
            frame = self._try_parse_frame()
            if frame is None:
                break
            func_id, payload, frame_data = frame
            if func_id == FUNC_ARM_FEEDBACK:
                self._handle_arm_feedback(payload, frame_data)
            else:
                self._log_mcu_frame_rx(func_id, payload, frame_data)

    def _try_reopen_serial_for_feedback(self, now: float):
        if now - self._last_serial_reopen_time < 1.0:
            return
        self._last_serial_reopen_time = now
        self.get_logger().warn('Serial is closed; trying to reopen for feedback')
        self._open_serial()

    def _try_parse_frame(self) -> tuple[int, bytes, bytes] | None:
        while len(self._rx_buffer) >= 5:
            if self._rx_buffer[0] == 0x55 and self._rx_buffer[1] == 0xAA:
                func_id = self._rx_buffer[2]
                payload_len = self._rx_buffer[3]
                frame_len = 5 + payload_len
                if len(self._rx_buffer) < frame_len:
                    return None
                frame_data = bytes(self._rx_buffer[:frame_len])
                expected_cs = sum(frame_data[:-1]) & 0xFF
                if frame_data[-1] != expected_cs:
                    self.get_logger().warn(
                        f'Bad checksum: got 0x{frame_data[-1]:02X}, '
                        f'expected 0x{expected_cs:02X}'
                    )
                    del self._rx_buffer[0]
                    continue
                payload = bytes(frame_data[4:-1])
                del self._rx_buffer[:frame_len]
                return (func_id, payload, frame_data)
            else:
                del self._rx_buffer[0]
        return None

    # ------------------------------------------------------------------
    # MCU -> Host frame handlers
    # ------------------------------------------------------------------

    def _handle_arm_feedback(self, payload: bytes, frame: bytes | None = None):
        try:
            feedback = parse_arm_feedback(payload)
        except ValueError as exc:
            self.get_logger().warn(f'Bad 0x21 feedback frame, dropping: {exc}')
            return

        self._latest_feedback = feedback
        self._last_feedback_time = time.monotonic()
        self._log_arm_feedback_rx(feedback, frame)

    # ------------------------------------------------------------------
    # Detection filtering
    # ------------------------------------------------------------------

    def detection_callback(self, msg: Detection3DArray):
        """Cache only stable camera-frame targets from the vision pipeline."""
        if self._bridge_state not in (
            BridgeState.WAIT_DETECTION,
            BridgeState.SEND_GRASP,
        ):
            return
        if not msg.detections:
            self._handle_missing_grasp_detection('empty detection message')
            return

        best = None
        best_score = -1.0
        for det in msg.detections:
            if not det.results:
                continue
            cls_name = det.results[0].hypothesis.class_id
            score = det.results[0].hypothesis.score
            if self.target_class and cls_name != self.target_class:
                continue
            if score > best_score:
                best_score = score
                best = det

        if best is None:
            self._handle_missing_grasp_detection('no matching detection')
            return

        pos = best.results[0].pose.pose.position
        raw_camera = (float(pos.x), float(pos.y), float(pos.z))

        filtered = self.ema.update(raw_camera)
        stable = self.stability.update(filtered)
        if stable is not None:
            self._latest_stable_camera_target = stable
            self._last_detection_time = time.monotonic()
            self._grasp_occlusion_start_time = None
            if self._bridge_state == BridgeState.WAIT_DETECTION:
                self._active_target = None
                self._hold_target = None
                self._active_target_sent_time = None
                self._active_target_start_end_xyz = None
                self._reset_arrival_tracking()
                self._set_state(BridgeState.SEND_GRASP)

    def _handle_missing_grasp_detection(self, reason: str):
        if self._bridge_state == BridgeState.SEND_GRASP and (
            self.grasp_occlusion_hold_enabled and self._grasp_command_history
        ):
            return
        self._drop_grasp_detection(reason)

    # ------------------------------------------------------------------
    # Timer state machine
    # ------------------------------------------------------------------

    def send_timer_callback(self):
        """Drive the host-side grasp/place state machine on each timer tick."""
        self._read_and_parse_feedback()

        now = time.monotonic()
        if self._bridge_state == BridgeState.WAIT_DETECTION:
            return
        if self._abort_on_feedback_loss(now):
            return
        if self._bridge_state == BridgeState.SEND_GRASP:
            self._maybe_send_grasp(now)
        elif self._bridge_state == BridgeState.GRASP_DELAY:
            self._send_pump_for_delay(TARGET_TYPE_GRASP)
            self._send_delay_hold(
                now,
                target_type=TARGET_TYPE_GRASP,
                tag='grasp_hold',
                next_state=BridgeState.SEND_PLACE,
            )
        elif self._bridge_state == BridgeState.SEND_PLACE:
            self._maybe_send_place(now)
        elif self._bridge_state == BridgeState.PLACE_DELAY:
            self._send_place_delay_then_pump_off(now)

    def _abort_on_feedback_loss(self, now: float) -> bool:
        """Fail safe when MCU feedback is absent long enough to make state unsafe."""
        if self._feedback_loss_age(now) <= self.feedback_loss_abort_sec:
            return False

        self.get_logger().warn(
            'MCU feedback lost; pump off requested and state reset to WAIT_DETECTION'
        )
        self._send_pump_command(False)
        self._reset_detection_state()
        self._active_target = None
        self._active_target_sent_time = None
        self._active_target_start_end_xyz = None
        self._hold_target = None
        self._reset_grasp_command_history()
        self._reset_arrival_tracking()
        self._delay_start_time = None
        self._pump_command_sent = False
        self._set_state(BridgeState.WAIT_DETECTION)
        return True

    def _feedback_loss_age(self, now: float) -> float:
        if self._last_feedback_time is not None:
            return now - self._last_feedback_time
        return now - self._state_enter_time

    def _maybe_send_grasp(self, now: float):
        if self._latest_stable_camera_target is None:
            return
        if not self._has_fresh_detection(now):
            self._warn_detection_timeout(now)
            self._maybe_hold_occluded_grasp(now)
            return
        if not self._has_fresh_feedback(now):
            self._warn_feedback_timeout(now)
            return
        if self._latest_feedback is None:
            return

        target = transform_camera_to_arm_base(
            self._latest_stable_camera_target,
            self.vision_transform,
            theta1_rad=self._latest_feedback.theta1_rad,
            current_end_xyz_m=self._latest_feedback.end_xyz_m,
        )
        target = self._apply_command_offset(target)

        sent = self._send_target_immediate(
            target, target_type=TARGET_TYPE_GRASP, tag='grasp', now=now
        )
        if sent:
            self._activate_target(target, now)
            self._record_grasp_command(target)
        if self._active_target is not None:
            self._update_arrival(
                now,
                self._active_target,
                reached_state=BridgeState.GRASP_DELAY,
            )

    def _maybe_hold_occluded_grasp(self, now: float):
        if (
            not self.grasp_occlusion_hold_enabled
            or not self._grasp_command_history
        ):
            self._drop_grasp_detection('vision target timeout')
            return
        if not self._has_fresh_feedback(now):
            self._warn_feedback_timeout(now)
            return

        if self._grasp_occlusion_start_time is None:
            self._grasp_occlusion_start_time = now
            self.get_logger().warn(
                'Vision target occluded during grasp; holding filtered target'
            )
        elif now - self._grasp_occlusion_start_time > self.grasp_occlusion_timeout_sec:
            self._drop_grasp_detection('vision occlusion timeout')
            return

        target = self._filtered_grasp_command()
        sent = self._send_target_immediate(
            target, target_type=TARGET_TYPE_GRASP, tag='grasp_occluded', now=now
        )
        if sent:
            self._activate_target(target, now)
        if self._active_target is not None:
            self._update_arrival(
                now,
                self._active_target,
                reached_state=BridgeState.GRASP_DELAY,
            )

    def _maybe_send_place(self, now: float):
        target = self._active_place_target
        sent = self._send_target_immediate(
            target, target_type=TARGET_TYPE_PLACE, tag='place', now=now
        )
        if sent:
            self._activate_target(target, now)
        if self._active_target is not None:
            self._update_arrival(
                now,
                self._active_target,
                reached_state=BridgeState.PLACE_DELAY,
            )

    def _send_delay_hold(
        self,
        now: float,
        target_type: int,
        tag: str,
        next_state: BridgeState,
    ):
        target = self._hold_target
        if target is None:
            target = (
                self._active_place_target
                if target_type == TARGET_TYPE_PLACE
                else self._active_target
            )
        if target is not None:
            self._send_target_immediate(
                target,
                target_type=target_type,
                tag=tag,
                now=now,
            )

        if not self._pump_command_sent:
            return
        if self._delay_start_time is None:
            self._delay_start_time = now
        if now - self._delay_start_time < self.arrival_delay_sec:
            return

        if next_state == BridgeState.WAIT_DETECTION:
            self._reset_detection_state()
        self._set_state(next_state)
        self._reset_arrival_tracking()
        self._delay_start_time = None
        self._active_target = None
        self._active_target_sent_time = None
        self._active_target_start_end_xyz = None
        self._hold_target = None
        if next_state == BridgeState.WAIT_DETECTION:
            self._reset_grasp_command_history()
        self._pump_command_sent = False

    def _send_place_delay_then_pump_off(self, now: float) -> None:
        """Keep the placed block stable before releasing the pump."""
        target = self._hold_target or self._active_place_target
        sent = self._send_target_immediate(
            target,
            target_type=TARGET_TYPE_PLACE,
            tag='place_hold',
            now=now,
        )

        if self._delay_start_time is None:
            if not sent:
                return
            self._delay_start_time = now
        if now - self._delay_start_time < self.arrival_delay_sec:
            return

        if not self._pump_command_sent:
            if not self._send_pump_command(False):
                return
            self._pump_command_sent = True

        self._reset_detection_state()
        self._set_state(BridgeState.WAIT_DETECTION)
        self._reset_arrival_tracking()
        self._delay_start_time = None
        self._active_target = None
        self._active_target_sent_time = None
        self._active_target_start_end_xyz = None
        self._hold_target = None
        self._reset_grasp_command_history()
        self._pump_command_sent = False

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    def _activate_target(
        self, target: tuple[float, float, float], sent_time: float
    ) -> None:
        """Track the command target that Host will use for arrival decisions."""
        if self._is_new_active_target(target):
            self._reset_arrival_tracking()
            self._active_target_start_end_xyz = (
                self._latest_feedback.end_xyz_m if self._latest_feedback else None
            )
            self._active_target_sent_time = sent_time
        self._active_target = target
        self._hold_target = target

    def _is_new_active_target(self, target: tuple[float, float, float]) -> bool:
        if self._active_target is None:
            return True
        return distance(self._active_target, target) > self.reach_tolerance_m

    def _send_target_immediate(
        self,
        target: tuple[float, float, float],
        target_type: int,
        tag: str,
        now: float | None = None,
    ) -> bool:
        now = time.monotonic() if now is None else now
        min_interval = 1.0 / max(self.max_send_rate, 0.1)
        if now - self.last_send_time < min_interval:
            self.get_logger().debug(
                f'{tag} target skipped: rate limit ({self.max_send_rate} Hz)'
            )
            return False

        frame = pack_arm_target(target_type, target[0], target[1], target[2])
        if not self._write_frame(frame):
            return False

        self.last_sent_target = target
        self.last_send_time = now
        self._log_target_tx(frame, target, target_type, tag)
        return True

    def _record_grasp_command(self, target: tuple[float, float, float]) -> None:
        """Keep recent successful grasp commands for camera-occluded approach."""
        self._grasp_command_history.append(target)
        if len(self._grasp_command_history) > self.grasp_command_filter_window:
            del self._grasp_command_history[0]

    def _filtered_grasp_command(self) -> tuple[float, float, float]:
        """Return the per-axis median of recent grasp commands."""
        return tuple(
            float(median(target[axis] for target in self._grasp_command_history))
            for axis in range(3)
        )

    def _has_fresh_detection(self, now: float) -> bool:
        if self._latest_stable_camera_target is None:
            return False
        if self._last_detection_time is None:
            return False
        return now - self._last_detection_time <= self.detection_timeout_sec

    def _warn_detection_timeout(self, now: float):
        if now - self._last_detection_warn_time < 1.0:
            return
        self._last_detection_warn_time = now
        self.get_logger().warn(
            'No fresh vision target; grasp target is not sent'
        )

    def _has_fresh_feedback(self, now: float) -> bool:
        if self._latest_feedback is None or self._last_feedback_time is None:
            return False
        return now - self._last_feedback_time <= self.feedback_timeout_sec

    def _warn_feedback_timeout(self, now: float):
        if now - self._last_feedback_warn_time < 1.0:
            return
        self._last_feedback_warn_time = now
        self.get_logger().warn(
            'No fresh MCU feedback; grasp target is not sent'
        )

    def _apply_command_offset(
        self, target: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Apply final mechanical command compensation before sending to MCU."""
        y_with_offset = target[1] + self.command_offset_m[1]
        y_command = self._apply_signed_magnitude_offset(
            y_with_offset, self.command_abs_y_offset_m
        )
        return (
            target[0] + self.command_offset_m[0],
            y_command,
            target[2] + self.command_offset_m[2],
        )

    @staticmethod
    def _apply_signed_magnitude_offset(value: float, delta: float) -> float:
        """Offset y magnitude without flipping direction when delta is negative."""
        sign = -1.0 if value < 0.0 else 1.0
        magnitude = max(0.0, abs(value) + delta)
        return sign * magnitude

    def _update_arrival(
        self,
        now: float,
        target: tuple[float, float, float],
        reached_state: BridgeState,
    ):
        if not self._has_fresh_feedback(now) or self._latest_feedback is None:
            self._reset_arrival_tracking()
            return

        feedback = self._latest_feedback
        dist_m = distance(feedback.end_xyz_m, target)
        reason = None

        if dist_m <= self.reach_tolerance_m:
            self._reach_count += 1
        else:
            self._reach_count = 0

        self._update_mcu_reached_count(feedback, dist_m)

        if self._is_arrival_stalled(feedback.end_xyz_m, dist_m):
            self._stall_count += 1
        else:
            self._stall_count = 0

        if self._reach_count >= self.reach_stable_frames:
            reason = 'distance'
        elif self._mcu_reached_count >= self.mcu_reached_stable_frames:
            reason = 'mcu_reached'
        elif self._stall_count >= self.arrival_stall_frames:
            reason = 'stall'

        if reason is not None:
            self.get_logger().info(f'Arrival detected by {reason}')
            self._delay_start_time = None
            self._hold_target = target
            self._reset_arrival_tracking()
            self._pump_command_sent = False
            self._set_state(reached_state)

    def _is_arrival_stalled(
        self,
        end_xyz_m: tuple[float, float, float],
        dist_to_target_m: float,
    ) -> bool:
        """Treat a near-target stopped end-effector as contact with the object."""
        if not self.arrival_stall_enabled:
            self._last_arrival_end_xyz = end_xyz_m
            return False
        if dist_to_target_m > self.arrival_stall_max_distance_m:
            self._last_arrival_end_xyz = end_xyz_m
            return False
        if self._last_arrival_end_xyz is None:
            self._last_arrival_end_xyz = end_xyz_m
            return False

        moved_m = distance(end_xyz_m, self._last_arrival_end_xyz)
        self._last_arrival_end_xyz = end_xyz_m
        return moved_m <= self.arrival_stall_epsilon_m

    def _is_mcu_reached_arrival(
        self, feedback: ArmFeedback, dist_to_target_m: float
    ) -> bool:
        """Accept MCU reached only when it is plausibly for the active target."""
        if not self.mcu_reached_enabled:
            return False
        if feedback.arm_state != ARM_STATE_REACHED:
            return False
        if not self._is_feedback_after_active_target():
            return False
        if dist_to_target_m > self.mcu_reached_max_distance_m:
            return False
        if dist_to_target_m <= self.reach_tolerance_m:
            return True
        return self._moved_since_active_target(feedback.end_xyz_m)

    def _moved_since_active_target(
        self, end_xyz_m: tuple[float, float, float]
    ) -> bool:
        if self._active_target_start_end_xyz is None:
            return False
        return (
            distance(end_xyz_m, self._active_target_start_end_xyz)
            >= self.mcu_reached_min_motion_m
        )

    def _update_mcu_reached_count(
        self, feedback: ArmFeedback, dist_to_target_m: float
    ) -> None:
        """Count MCU reached frames once per feedback sample, not once per timer."""
        if self._last_feedback_time == self._last_mcu_reached_feedback_time:
            return
        self._last_mcu_reached_feedback_time = self._last_feedback_time
        if self._is_mcu_reached_arrival(feedback, dist_to_target_m):
            self._mcu_reached_count += 1
        else:
            self._mcu_reached_count = 0

    def _reset_arrival_tracking(self):
        self._reach_count = 0
        self._mcu_reached_count = 0
        self._last_mcu_reached_feedback_time = None
        self._stall_count = 0
        self._last_arrival_end_xyz = None

    def _send_pump_for_delay(self, target_type: int):
        if self._pump_command_sent:
            return
        pump_on = target_type == TARGET_TYPE_GRASP
        if self._send_pump_command(pump_on):
            self._pump_command_sent = True

    def _send_pump_command(self, pump_on: bool) -> bool:
        frame = pack_pump_control(pump_on)
        if not self._write_frame(frame):
            return False
        self._log_pump_tx(frame, pump_on)
        return True

    def _log_target_tx(
        self,
        frame: bytes,
        target: tuple[float, float, float],
        target_type: int,
        tag: str,
    ) -> None:
        """Print the exact target command values written to the CDC serial port."""
        if not self.serial_tx_log:
            return
        msg = (
            f'SERIAL_TX 0x12 ARM_TARGET target={tag} type={target_type} '
            f'x={target[0]:.6f} y={target[1]:.6f} z={target[2]:.6f}'
        )
        if target_type == TARGET_TYPE_PLACE and self._locked_place_index is not None:
            name = PLACE_TARGET_NAMES.get(
                self._locked_place_index, f'idx_{self._locked_place_index}'
            )
            msg = f'{msg} place=[{self._locked_place_index}] {name}'
        if self.serial_tx_log_hex:
            msg = f'{msg} frame={frame.hex(" ")}'
        self.get_logger().info(msg)

    def _log_pump_tx(self, frame: bytes, pump_on: bool) -> None:
        """Print the exact pump command state written to the CDC serial port."""
        if not self.serial_tx_log:
            return
        msg = f'SERIAL_TX 0x13 PUMP state={1 if pump_on else 0}'
        if self.serial_tx_log_hex:
            msg = f'{msg} frame={frame.hex(" ")}'
        self.get_logger().info(msg)

    def _log_arm_feedback_rx(
        self, feedback: ArmFeedback, frame: bytes | None
    ) -> None:
        """Print parsed MCU feedback only when a complete frame was received."""
        if not self.serial_rx_log:
            return
        x, y, z = feedback.end_xyz_m
        msg = (
            'SERIAL_RX 0x21 ARM_FEEDBACK '
            f'state={feedback.arm_state} '
            f'end_x={x:.6f} end_y={y:.6f} end_z={z:.6f} '
            f'theta1={feedback.theta1_rad:.6f}'
        )
        if self.serial_rx_log_hex and frame is not None:
            msg = f'{msg} frame={frame.hex(" ")}'
        self.get_logger().info(msg)

    def _log_mcu_frame_rx(self, func_id: int, payload: bytes, frame: bytes) -> None:
        """Print non-feedback MCU frames that passed framing and checksum."""
        if not self.serial_rx_log:
            return
        msg = f'SERIAL_RX 0x{func_id:02X} payload={payload.hex(" ")}'
        if self.serial_rx_log_hex:
            msg = f'{msg} frame={frame.hex(" ")}'
        self.get_logger().info(msg)

    @property
    def _active_place_target(self) -> tuple[float, float, float]:
        """Return the currently active place target (locked or fallback)."""
        if self._locked_place_target is not None:
            return self._locked_place_target
        return self._place_target_command

    def _lock_place_target(self) -> None:
        """Lock place target based on end_y at grasp arrival moment.

        end_y < 0  → right_front  (index 3)
        end_y > 0  → left_rear   (index 1)
        end_y == 0 → use configured place_target_index as fallback
        """
        if self._latest_feedback is None:
            self.get_logger().warn(
                'Cannot lock place target: no MCU feedback available; '
                'falling back to place_target_index'
            )
            return

        end_y = self._latest_feedback.end_xyz_m[1]

        if len(self._place_targets) >= 4:
            if end_y < 0:
                idx = 3  # right_front
            elif end_y > 0:
                idx = 1  # left_rear
            else:
                idx = self._place_target_index
        else:
            idx = self._place_target_index
            self.get_logger().warn(
                f'place_targets_m has only {len(self._place_targets)} '
                f'target(s); dynamic selection disabled, '
                f'using place_target_index={idx}'
            )

        name = PLACE_TARGET_NAMES.get(idx, f'idx_{idx}')
        self._locked_place_target = self._place_targets[idx]
        self._locked_place_index = idx
        self.get_logger().info(
            f'Locked place target [{idx}] {name}: '
            f'({self._locked_place_target[0]:.4f}, '
            f'{self._locked_place_target[1]:.4f}, '
            f'{self._locked_place_target[2]:.4f}) '
            f'(end_y={end_y:.4f})'
        )

    def _set_state(self, state: BridgeState):
        if self._bridge_state == state:
            return
        prev = self._bridge_state
        self._bridge_state = state
        self._state_enter_time = time.monotonic()

        if state == BridgeState.GRASP_DELAY:
            self._lock_place_target()
        elif state == BridgeState.WAIT_DETECTION:
            self._locked_place_target = None
            self._locked_place_index = None

        self.get_logger().info(f'Arm bridge state: {prev.value} -> {state.value}')

    def _drop_grasp_detection(self, reason: str):
        if self._bridge_state != BridgeState.SEND_GRASP:
            return
        self.get_logger().warn(f'Vision target lost during grasp: {reason}')
        self._reset_detection_state()
        self._active_target = None
        self._active_target_sent_time = None
        self._active_target_start_end_xyz = None
        self._hold_target = None
        self._reset_grasp_command_history()
        self._reset_arrival_tracking()
        self._set_state(BridgeState.WAIT_DETECTION)

    def _reset_detection_state(self):
        self.ema.reset()
        self.stability.reset()
        self._latest_stable_camera_target = None
        self._last_detection_time = None

    def _reset_grasp_command_history(self):
        self._grasp_command_history = []
        self._grasp_occlusion_start_time = None

    def _is_feedback_after_active_target(self) -> bool:
        """Avoid treating a stale MCU reached state as arrival for a new target."""
        if self._active_target_sent_time is None or self._last_feedback_time is None:
            return False
        return self._last_feedback_time > self._active_target_sent_time

    # ------------------------------------------------------------------

    def destroy_node(self):
        """Close CDC serial before shutting down the ROS node."""
        self.serial.close()
        self.get_logger().info('Serial closed')
        super().destroy_node()


def main(args=None):
    """Run the arm serial bridge ROS2 node."""
    if rclpy is None:
        raise RuntimeError('arm_serial_bridge_node requires ROS2 rclpy')

    rclpy.init(args=args)
    node = ArmSerialBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
