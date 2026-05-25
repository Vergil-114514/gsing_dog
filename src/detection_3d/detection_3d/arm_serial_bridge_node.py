"""
ROS2 node: subscribe to 3D detections and send camera-frame XYZ to STM32 USB CDC.

Protocol frame:
  [0x55][0xAA][0x12][0x0C][x_f32_le][y_f32_le][z_f32_le][checksum]

Coordinates are sent in meters using the STM32 camera convention:
  +X right, -Y forward, +Z down.
"""

import glob
import os
import select
import struct
import termios
import time
import math

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import Bool


FUNC_ARM_CONTROL = 0x12
FUNC_SUCTION_CONTROL = 0x13


def build_frame(func_id: int, payload: bytes) -> bytes:
    header = bytes([0x55, 0xAA, func_id, len(payload)])
    frame_no_checksum = header + payload
    checksum = sum(frame_no_checksum) & 0xFF
    return frame_no_checksum + bytes([checksum])


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

    def write(self, data: bytes):
        if self.fd is None:
            raise OSError('serial port is not open')
        os.write(self.fd, data)

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


class ArmSerialBridgeNode(Node):

    def __init__(self):
        super().__init__('arm_serial_bridge')

        self.declare_parameter('serial_port', 'auto')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('send_rate', 10.0)
        self.declare_parameter('detection_topic', '/detection/detections_3d')
        self.declare_parameter('suction_topic', '/arm/suction')
        self.declare_parameter('target_class', '')
        self.declare_parameter('min_send_delta_m', 0.03)
        self.declare_parameter('ema_alpha', 0.25)
        self.declare_parameter('stable_radius_m', 0.02)
        self.declare_parameter('stable_frames', 3)
        self.declare_parameter('max_send_rate', 3.0)
        self.declare_parameter('read_feedback', True)

        port = self.get_parameter('serial_port').value
        baud = int(self.get_parameter('baud_rate').value)
        self.send_rate = float(self.get_parameter('send_rate').value)
        det_topic = self.get_parameter('detection_topic').value
        suction_topic = self.get_parameter('suction_topic').value
        self.target_class = self.get_parameter('target_class').value
        self.min_send_delta_m = float(self.get_parameter('min_send_delta_m').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)
        self.stable_radius_m = float(self.get_parameter('stable_radius_m').value)
        self.stable_frames = int(self.get_parameter('stable_frames').value)
        self.max_send_rate = float(self.get_parameter('max_send_rate').value)
        self.read_feedback = bool(self.get_parameter('read_feedback').value)

        self.serial = CdcSerial(port, baud)
        self._open_serial()

        self.pending_target = None
        self.last_sent_target = None
        self.last_send_time = 0.0
        self.filtered_target = None
        self.stable_candidate = None
        self.stable_count = 0

        self.sub_det = self.create_subscription(
            Detection3DArray, det_topic, self.detection_callback, 10
        )
        self.sub_suction = self.create_subscription(
            Bool, suction_topic, self.suction_callback, 10
        )

        period = 1.0 / max(self.send_rate, 0.1)
        self.timer = self.create_timer(period, self.send_timer_callback)

        self.get_logger().info(
            f'ArmSerialBridge ready. rate={self.send_rate}Hz, '
            f'max_send_rate={self.max_send_rate}Hz, '
            f'min_delta={self.min_send_delta_m:.3f}m, '
            f'ema_alpha={self.ema_alpha:.2f}, stable={self.stable_frames} frames, '
            f'target_class="{self.target_class or "any"}", units=m, frame=stm32_camera'
        )

    def _open_serial(self) -> bool:
        try:
            self.serial.open()
            self.get_logger().info(f'Serial opened: {self.serial.port} @ {self.serial.baud_rate}')
            return True
        except OSError as exc:
            self.get_logger().error(f'Failed to open STM32 CDC: {exc}')
            return False

    def detection_callback(self, msg: Detection3DArray):
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
        raw_optical = (float(pos.x), float(pos.y), float(pos.z))
        raw_target = self._optical_to_stm32_camera(raw_optical)
        filtered_target = self._update_ema(raw_target)
        stable_target = self._update_stability(filtered_target)

        self.get_logger().debug(
            f'Optical XYZ=({raw_optical[0]:.3f}, {raw_optical[1]:.3f}, {raw_optical[2]:.3f})m, '
            f'stm32=({raw_target[0]:.3f}, {raw_target[1]:.3f}, {raw_target[2]:.3f})m, '
            f'filtered=({filtered_target[0]:.3f}, {filtered_target[1]:.3f}, {filtered_target[2]:.3f})m, '
            f'stable_count={self.stable_count}, conf={best_score:.2f}'
        )

        if stable_target is None:
            return

        if self.pending_target is not None and not self._target_changed(stable_target, self.pending_target):
            return
        if self.last_sent_target is not None and not self._target_changed(stable_target, self.last_sent_target):
            return

        self.pending_target = stable_target
        self.get_logger().info(
            f'Stable target ready XYZ=({stable_target[0]:.3f}, '
            f'{stable_target[1]:.3f}, {stable_target[2]:.3f})m'
        )

    def suction_callback(self, msg: Bool):
        mode = 1 if msg.data else 0
        frame = build_frame(FUNC_SUCTION_CONTROL, bytes([mode]))
        if self._write_frame(frame):
            self.get_logger().info(f'Suction {"ON" if mode else "OFF"}')

    def send_timer_callback(self):
        if self.pending_target is None:
            self._read_feedback_once()
            return

        now = time.monotonic()
        min_interval = 1.0 / max(self.max_send_rate, 0.1)
        if now - self.last_send_time < min_interval:
            self._read_feedback_once()
            return

        target = self.pending_target
        if self.last_sent_target is not None and not self._target_changed(target, self.last_sent_target):
            self.pending_target = None
            self._read_feedback_once()
            return

        payload = struct.pack('<fff', target[0], target[1], target[2])
        frame = build_frame(FUNC_ARM_CONTROL, payload)
        if self._write_frame(frame):
            self.last_sent_target = target
            self.last_send_time = now
            self.pending_target = None
            self.get_logger().info(
                f'Sent arm target STM32 XYZ=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})m'
            )

        self._read_feedback_once()

    def _write_frame(self, frame: bytes) -> bool:
        if not self.serial.is_open and not self._open_serial():
            return False
        try:
            self.serial.write(frame)
            return True
        except OSError as exc:
            self.get_logger().error(f'Serial write error: {exc}; will reopen next tick')
            self.serial.close()
            return False

    def _read_feedback_once(self):
        if not self.read_feedback or not self.serial.is_open:
            return
        data = self.serial.read_available(timeout_sec=0.0, max_bytes=512)
        if data:
            self.get_logger().debug(f'STM32 RX: {data.hex(" ")}')

    def _target_changed(self, a, b) -> bool:
        if a is None or b is None:
            return True
        return self._distance(a, b) >= self.min_send_delta_m

    def _update_ema(self, raw_target):
        alpha = min(max(self.ema_alpha, 0.0), 1.0)
        if self.filtered_target is None:
            self.filtered_target = raw_target
        else:
            self.filtered_target = tuple(
                alpha * raw_target[i] + (1.0 - alpha) * self.filtered_target[i]
                for i in range(3)
            )
        return self.filtered_target

    def _update_stability(self, target):
        if self.stable_candidate is None:
            self.stable_candidate = target
            self.stable_count = 1
            if self.stable_count >= max(self.stable_frames, 1):
                return self.stable_candidate
            return None

        if self._distance(target, self.stable_candidate) <= self.stable_radius_m:
            # Keep the accepted candidate smooth while it gathers stable frames.
            self.stable_candidate = tuple(
                0.5 * self.stable_candidate[i] + 0.5 * target[i]
                for i in range(3)
            )
            self.stable_count += 1
        else:
            self.stable_candidate = target
            self.stable_count = 1
            return None

        if self.stable_count >= max(self.stable_frames, 1):
            return self.stable_candidate
        return None

    @staticmethod
    def _distance(a, b) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    @staticmethod
    def _optical_to_stm32_camera(optical_target):
        # ROS optical camera frame: +X right, +Y down, +Z forward.
        # STM32 legacy camera frame: +X right, -Y forward, +Z down.
        return (
            optical_target[0],
            -optical_target[2],
            optical_target[1],
        )

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
