"""
ROS2 node: USB CDC serial bridge to STM32 MCU.

Host -> MCU:  func 0x12 — camera-frame arm target XYZ
MCU -> Host:  func 0x21 — 1-byte grasp/place flag (0=grasp, 1=place)
"""

import glob
import os
import select
import termios
import time

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray

from detection_3d.protocol import (
    pack_arm_target_xyz,
    parse_arm_flag,
    FUNC_ARM_FLAG,
)
from detection_3d.target_filter import EMAFilter, StabilityFilter
from detection_3d.place_targets import validate_place_targets, get_place_target


# ---------------------------------------------------------------------------
# Serial transport
# ---------------------------------------------------------------------------

class CdcSerial:
    def __init__(self, port: str, baud_rate: int):
        self.requested_port = port
        self.baud_rate = baud_rate
        self.fd = None
        self.port = None

    @property
    def is_open(self) -> bool:
        return self.fd is not None

    def open(self):
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
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def write(self, data: bytes) -> bool:
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
        baud_name = f'B{baud_rate}'
        if not hasattr(termios, baud_name):
            raise ValueError(f'unsupported baud rate: {baud_rate}')
        return getattr(termios, baud_name)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ArmSerialBridgeNode(Node):

    def __init__(self):
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

        # === MCU-driven state ===
        self._grasp_place_flag: int = 0       # 0 = grasp, 1 = place
        self._latest_stable_target: tuple[float, float, float] | None = None

        # === rate limiting / state ===
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
        if not self.read_feedback or not self.serial.is_open:
            return

        raw = self.serial.read_available(timeout_sec=0.0, max_bytes=512)
        if raw:
            self._rx_buffer.extend(raw)

        while True:
            frame = self._try_parse_frame()
            if frame is None:
                break
            func_id, payload = frame
            if func_id == FUNC_ARM_FLAG:
                self._handle_arm_flag(payload)
            else:
                self.get_logger().debug(
                    f'MCU frame func=0x{func_id:02X} '
                    f'payload={payload.hex(" ")} ({len(payload)}B)'
                )

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

    def _handle_arm_flag(self, payload: bytes):
        try:
            flag = parse_arm_flag(payload)
        except ValueError as exc:
            self.get_logger().warn(f'Bad 0x21 frame, dropping: {exc}')
            return

        if flag not in (0, 1):
            self.get_logger().warn(f'Unknown flag value {flag}, ignored')
            return

        prev_flag = self._grasp_place_flag
        if flag == prev_flag:
            return

        self._grasp_place_flag = flag
        self.get_logger().info(
            f'MCU flag: {prev_flag} -> {flag} '
            f'({"grasp" if flag == 0 else "place"})'
        )
        if flag == 0:
            self.ema.reset()
            self.stability.reset()
            self._latest_stable_target = None

    # ------------------------------------------------------------------
    # Detection filtering (camera-frame coords, no FK)
    # ------------------------------------------------------------------

    def detection_callback(self, msg: Detection3DArray):
        if self._grasp_place_flag != 0:
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
            self._latest_stable_target = stable

    # ------------------------------------------------------------------
    # Timer — send based on MCU flag
    # ------------------------------------------------------------------

    def send_timer_callback(self):
        self._read_and_parse_feedback()

        if self._grasp_place_flag == 0:
            self._maybe_send_grasp()
        elif self._grasp_place_flag == 1:
            self._maybe_send_place()

    def _maybe_send_grasp(self):
        if self._latest_stable_target is not None:
            self._send_target_immediate(self._latest_stable_target, 'grasp')

    def _maybe_send_place(self):
        self._send_target_immediate(self._place_target, 'place')

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    def _send_target_immediate(
        self, target: tuple[float, float, float], tag: str
    ) -> bool:
        now = time.monotonic()
        min_interval = 1.0 / max(self.max_send_rate, 0.1)
        if now - self.last_send_time < min_interval:
            self.get_logger().warn(
                f'{tag} target skipped: rate limit ({self.max_send_rate} Hz)'
            )
            return False

        frame = pack_arm_target_xyz(target[0], target[1], target[2])
        if not self._write_frame(frame):
            return False

        self.last_sent_target = target
        self.last_send_time = now
        self.get_logger().info(
            f'Sent {tag} ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})m'
        )
        return True

    # ------------------------------------------------------------------

    def destroy_node(self):
        self.serial.close()
        self.get_logger().info('Serial closed')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmSerialBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
