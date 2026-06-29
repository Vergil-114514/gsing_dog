"""
ROS2 node: USB CDC serial bridge to STM32 MCU.

Host -> MCU:  func 0x12, target_type + arm_base target xyz.
MCU -> Host:  func 0x21, arm state + current end xyz + theta1.
"""

from enum import Enum
import glob
import os
import select
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

from detection_3d.place_targets import validate_place_targets, get_place_target
from detection_3d.protocol import (
    ARM_STATE_ERROR,
    FUNC_ARM_FEEDBACK,
    TARGET_TYPE_GRASP,
    TARGET_TYPE_PLACE,
    ArmFeedback,
    pack_arm_target,
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
    ERROR = 'ERROR'


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
        self.declare_parameter('reach_tolerance_m', 0.015)
        self.declare_parameter('reach_stable_frames', 3)
        self.declare_parameter('arrival_delay_sec', 1.0)
        self.declare_parameter('feedback_timeout_sec', 0.5)

        # ---- vision camera -> arm base transform ----
        self.declare_parameter('camera_to_arm_transform_enabled', True)
        self.declare_parameter('camera_offset_x_m', 0.105)
        self.declare_parameter('camera_offset_y_m', 0.0)
        self.declare_parameter('camera_offset_z_m', -0.078)

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
        )

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
        place_targets = validate_place_targets(place_targets_m_raw)
        self._place_target = get_place_target(place_targets, place_target_index)

        # === host-driven state ===
        self._bridge_state = BridgeState.WAIT_DETECTION
        self._latest_stable_camera_target: tuple[float, float, float] | None = None
        self._latest_feedback: ArmFeedback | None = None
        self._last_feedback_time: float | None = None
        self._active_target: tuple[float, float, float] | None = None
        self._hold_target: tuple[float, float, float] | None = None
        self._reach_count = 0
        self._delay_start_time: float | None = None
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
            f'feedback_timeout={self.feedback_timeout_sec:.2f}s, '
            f'transform_enabled={self.camera_to_arm_transform_enabled}, '
            f'place=({self._place_target[0]:.3f}, '
            f'{self._place_target[1]:.3f}, '
            f'{self._place_target[2]:.3f})m'
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
            func_id, payload = frame
            if func_id == FUNC_ARM_FEEDBACK:
                self._handle_arm_feedback(payload)
            else:
                self.get_logger().debug(
                    f'MCU frame func=0x{func_id:02X} '
                    f'payload={payload.hex(" ")} ({len(payload)}B)'
                )

    def _try_reopen_serial_for_feedback(self, now: float):
        if now - self._last_serial_reopen_time < 1.0:
            return
        self._last_serial_reopen_time = now
        self.get_logger().warn('Serial is closed; trying to reopen for feedback')
        self._open_serial()

    def _try_parse_frame(self) -> tuple[int, bytes] | None:
        while len(self._rx_buffer) >= 5:
            if self._rx_buffer[0] == 0x55 and self._rx_buffer[1] == 0xAA:
                func_id = self._rx_buffer[2]
                payload_len = self._rx_buffer[3]
                frame_len = 5 + payload_len
                if len(self._rx_buffer) < frame_len:
                    return None
                frame_data = self._rx_buffer[:frame_len]
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
                return (func_id, payload)
            else:
                del self._rx_buffer[0]
        return None

    # ------------------------------------------------------------------
    # MCU -> Host frame handlers
    # ------------------------------------------------------------------

    def _handle_arm_feedback(self, payload: bytes):
        try:
            feedback = parse_arm_feedback(payload)
        except ValueError as exc:
            self.get_logger().warn(f'Bad 0x21 feedback frame, dropping: {exc}')
            return

        self._latest_feedback = feedback
        self._last_feedback_time = time.monotonic()

        if feedback.arm_state == ARM_STATE_ERROR:
            self._set_state(BridgeState.ERROR)
            self.get_logger().error('MCU arm_state=error; stop sending targets')

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
            return

        pos = best.results[0].pose.pose.position
        raw_camera = (float(pos.x), float(pos.y), float(pos.z))

        filtered = self.ema.update(raw_camera)
        stable = self.stability.update(filtered)
        if stable is not None:
            self._latest_stable_camera_target = stable
            if self._bridge_state == BridgeState.WAIT_DETECTION:
                self._active_target = None
                self._hold_target = None
                self._reach_count = 0
                self._set_state(BridgeState.SEND_GRASP)

    # ------------------------------------------------------------------
    # Timer state machine
    # ------------------------------------------------------------------

    def send_timer_callback(self):
        """Drive the host-side grasp/place state machine on each timer tick."""
        self._read_and_parse_feedback()

        now = time.monotonic()
        if self._bridge_state == BridgeState.ERROR:
            return
        if self._bridge_state == BridgeState.WAIT_DETECTION:
            return
        if self._bridge_state == BridgeState.SEND_GRASP:
            self._maybe_send_grasp(now)
        elif self._bridge_state == BridgeState.GRASP_DELAY:
            self._send_delay_hold(
                now,
                target_type=TARGET_TYPE_GRASP,
                tag='grasp_hold',
                next_state=BridgeState.SEND_PLACE,
            )
        elif self._bridge_state == BridgeState.SEND_PLACE:
            self._maybe_send_place(now)
        elif self._bridge_state == BridgeState.PLACE_DELAY:
            self._send_delay_hold(
                now,
                target_type=TARGET_TYPE_PLACE,
                tag='place_hold',
                next_state=BridgeState.WAIT_DETECTION,
            )

    def _maybe_send_grasp(self, now: float):
        if self._latest_stable_camera_target is None:
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

        if self._send_target_immediate(
            target, target_type=TARGET_TYPE_GRASP, tag='grasp', now=now
        ):
            self._active_target = target
            self._hold_target = target
            self._update_arrival(
                now,
                target,
                reached_state=BridgeState.GRASP_DELAY,
            )

    def _maybe_send_place(self, now: float):
        target = self._place_target
        if self._send_target_immediate(
            target, target_type=TARGET_TYPE_PLACE, tag='place', now=now
        ):
            self._active_target = target
            self._hold_target = target
            self._update_arrival(
                now,
                target,
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
                self._place_target
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

        if self._delay_start_time is None:
            self._delay_start_time = now
        if now - self._delay_start_time < self.arrival_delay_sec:
            return

        if next_state == BridgeState.WAIT_DETECTION:
            self._reset_detection_state()
        self._set_state(next_state)
        self._reach_count = 0
        self._delay_start_time = None
        self._active_target = None
        self._hold_target = None

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

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
        self.get_logger().info(
            f'Sent {tag} type={target_type} '
            f'({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})m'
        )
        return True

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

    def _update_arrival(
        self,
        now: float,
        target: tuple[float, float, float],
        reached_state: BridgeState,
    ):
        if not self._has_fresh_feedback(now) or self._latest_feedback is None:
            self._reach_count = 0
            return

        dist_m = distance(self._latest_feedback.end_xyz_m, target)
        if dist_m <= self.reach_tolerance_m:
            self._reach_count += 1
        else:
            self._reach_count = 0

        if self._reach_count >= self.reach_stable_frames:
            self._delay_start_time = now
            self._hold_target = target
            self._reach_count = 0
            self._set_state(reached_state)

    def _set_state(self, state: BridgeState):
        if self._bridge_state == state:
            return
        prev = self._bridge_state
        self._bridge_state = state
        self.get_logger().info(f'Arm bridge state: {prev.value} -> {state.value}')

    def _reset_detection_state(self):
        self.ema.reset()
        self.stability.reset()
        self._latest_stable_camera_target = None

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
