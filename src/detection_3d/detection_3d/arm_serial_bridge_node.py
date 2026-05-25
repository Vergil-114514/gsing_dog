"""
ROS2 node: Unified USB CDC serial bridge to STM32 MCU.

Handles ALL host <-> MCU communication over a single serial port.

Host -> MCU:
  func 0x10 — leg joint control
  func 0x11 — arm joint + pump control
  func 0x12 — arm cartesian target (arm_base, meters)
  func 0x13 — suction on/off

MCU -> Host:
  func 0x20 — robot state feedback (reserved)
  func 0x21 — 4-axis arm joint angles + grasp/place flag

Grasp/place is controlled by the MCU via the flag byte in 0x21 frames:
  flag = 0 → grasp: transform camera detections -> arm_base, send target
  flag = 1 → place: send pre-configured place coords

The arm joint angles are used to dynamically update the camera -> arm_base
TF so that the transform accounts for the arm's current pose.
"""

import glob
import math
import os
import select
import struct
import termios
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Duration
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PointStamped, TransformStamped
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped support

from detection_3d.protocol import (
    pack_arm_target_xyz,
    pack_arm_joint_pump_command,
    pack_leg_command,
    pack_suction,
    parse_arm_pose,
    FUNC_ARM_POSE,
    FUNC_ROBOT_STATE,
)
from detection_3d.target_filter import EMAFilter, StabilityFilter
from detection_3d.place_targets import validate_place_targets, get_place_target
from detection_3d.arm_kinematics import joints_to_camera_arm_transform

try:
    from quadruped_interfaces.msg import LegCommand, ArmPumpCommand, RobotState
    _HAS_QUADRUPED_INTERFACES = True
except ImportError:
    LegCommand = None       # type: ignore
    ArmPumpCommand = None   # type: ignore
    RobotState = None       # type: ignore
    _HAS_QUADRUPED_INTERFACES = False


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
        """Write all bytes, looping for non-blocking fd short writes."""
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

def _rpy_to_quaternion(roll: float, pitch: float, yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


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
        self.declare_parameter('min_send_delta_m', 0.03)
        self.declare_parameter('ema_alpha', 0.25)
        self.declare_parameter('stable_radius_m', 0.02)
        self.declare_parameter('stable_frames', 3)
        self.declare_parameter('max_send_rate', 3.0)
        self.declare_parameter('read_feedback', True)

        # ---- TF ----
        self.declare_parameter('target_frame', 'arm_base')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('camera_to_arm_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('camera_to_arm_rpy', [0.0, 0.0, 0.0])

        # ---- place targets ----
        self.declare_parameter('place_targets_m', [0.0, 0.0, 0.0])
        self.declare_parameter('place_target_index', 0)

        # ---- topics (vision) ----
        self.declare_parameter('suction_topic', '/arm/suction')

        # ---- topics (quadruped leg / arm) ----
        self.declare_parameter('leg_command_topic', 'leg_command')
        self.declare_parameter('arm_pump_command_topic', 'arm_pump_command')
        self.declare_parameter('robot_state_topic', 'robot_state')

        # === resolve parameters ===

        port = self.get_parameter('serial_port').value
        baud = int(self.get_parameter('baud_rate').value)
        self.send_rate = float(self.get_parameter('send_rate').value)
        det_topic = self.get_parameter('detection_topic').value
        self.target_class = self.get_parameter('target_class').value
        self.min_send_delta_m = float(self.get_parameter('min_send_delta_m').value)
        ema_alpha = float(self.get_parameter('ema_alpha').value)
        stable_radius = float(self.get_parameter('stable_radius_m').value)
        stable_frames = int(self.get_parameter('stable_frames').value)
        self.max_send_rate = float(self.get_parameter('max_send_rate').value)
        self.read_feedback = bool(self.get_parameter('read_feedback').value)

        self.target_frame = self.get_parameter('target_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.camera_to_arm_xyz = tuple(
            float(v) for v in self.get_parameter('camera_to_arm_xyz').value
        )
        self.camera_to_arm_rpy = tuple(
            float(v) for v in self.get_parameter('camera_to_arm_rpy').value
        )

        place_targets_m_raw = self.get_parameter('place_targets_m').value
        place_target_index = int(self.get_parameter('place_target_index').value)

        suction_topic = self.get_parameter('suction_topic').value
        leg_topic = self.get_parameter('leg_command_topic').value
        arm_pump_topic = self.get_parameter('arm_pump_command_topic').value
        robot_state_topic = self.get_parameter('robot_state_topic').value

        # === serial ===
        self.serial = CdcSerial(port, baud)
        self._open_serial()

        # === TF ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

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
        self._arm_joints: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._latest_stable_target: tuple[float, float, float] | None = None
        self._place_sent: bool = False
        self._prev_flag: int = 0

        # === rate limiting / state ===
        self.last_sent_target = None
        self.last_send_time = 0.0
        self._last_tf_warning_time = 0.0

        # === rx buffer ===
        self._rx_buffer = bytearray()

        # === publishers ===
        if _HAS_QUADRUPED_INTERFACES:
            self.robot_state_pub = self.create_publisher(
                RobotState, robot_state_topic, 10
            )
        else:
            self.robot_state_pub = None

        # === subscribers ===
        self.sub_det = self.create_subscription(
            Detection3DArray, det_topic, self.detection_callback, 10
        )
        self.sub_suction = self.create_subscription(
            Bool, suction_topic, self.suction_callback, 10
        )

        if _HAS_QUADRUPED_INTERFACES:
            self.sub_leg = self.create_subscription(
                LegCommand, leg_topic, self.leg_callback, 10
            )
            self.sub_arm_pump = self.create_subscription(
                ArmPumpCommand, arm_pump_topic, self.arm_pump_callback, 10
            )

        # === timer ===
        period = 1.0 / max(self.send_rate, 0.1)
        self.timer = self.create_timer(period, self.send_timer_callback)

        self.get_logger().info(
            f'Unified serial bridge ready. '
            f'rate={self.send_rate}Hz, '
            f'max_send_rate={self.max_send_rate}Hz, '
            f'min_delta={self.min_send_delta_m:.3f}m, '
            f'ema_alpha={ema_alpha:.2f}, stable={stable_frames} frames, '
            f'target_class="{self.target_class or "any"}", '
            f'target_frame={self.target_frame}, '
            f'camera_frame={self.camera_frame}, '
            f'place=({self._place_target[0]:.3f}, '
            f'{self._place_target[1]:.3f}, '
            f'{self._place_target[2]:.3f})m, '
            f'quadruped_ifaces={_HAS_QUADRUPED_INTERFACES}, '
            f'units=m'
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
        """Read incoming serial data, parse complete frames."""
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
            if func_id == FUNC_ARM_POSE:
                self._handle_arm_pose(payload)
            elif func_id == FUNC_ROBOT_STATE:
                self._handle_robot_state(payload)
            else:
                self.get_logger().debug(
                    f'MCU frame func=0x{func_id:02X} '
                    f'payload={payload.hex(" ")} ({len(payload)}B)'
                )

    def _try_parse_frame(self) -> tuple[int, bytes] | None:
        """Extract one complete frame from rx_buffer. Returns (func_id, payload) or None."""
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

    def _handle_arm_pose(self, payload: bytes):
        """Parse 0x21 arm-pose frame, update dynamic TF."""
        joints, flag = parse_arm_pose(payload)
        self._arm_joints = joints

        prev_flag = self._grasp_place_flag
        self._grasp_place_flag = flag

        if flag != prev_flag:
            self.get_logger().info(
                f'MCU flag: {prev_flag} -> {flag} '
                f'({"grasp" if flag == 0 else "place"})'
            )
            if flag == 0:
                # Switching to grasp: reset filters
                self.ema.reset()
                self.stability.reset()
                self._latest_stable_target = None
            else:
                # Switching to place: ready to send
                self._place_sent = False

        # Compute FK and publish dynamic TF
        self._publish_dynamic_camera_to_arm(joints)

    def _handle_robot_state(self, payload: bytes):
        """Log incoming RobotState frame (payload format TBD with MCU firmware)."""
        self.get_logger().debug(
            f'RobotState rx ({len(payload)}B) — parser not yet defined, '
            f'raw={payload.hex(" ")}'
        )

    # ------------------------------------------------------------------
    # Dynamic TF
    # ------------------------------------------------------------------

    def _publish_dynamic_camera_to_arm(self, joints: list[float]):
        """Publish camera_frame -> target_frame TF computed from arm joint angles."""
        tx, ty, tz, roll, pitch, yaw = joints_to_camera_arm_transform(
            joints[0], joints[1], joints[2], joints[3],
            self.camera_to_arm_xyz,
            self.camera_to_arm_rpy,
        )

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = self.target_frame
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = tz
        qx, qy, qz, qw = _rpy_to_quaternion(roll, pitch, yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(t)

    # ------------------------------------------------------------------
    # Detection filtering
    # ------------------------------------------------------------------

    def detection_callback(self, msg: Detection3DArray):
        # Always filter detections when in grasp mode (flag=0).
        # In place mode we skip filtering to save CPU.
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
        arm_base = self._transform_to_target(
            pos.x, pos.y, pos.z, msg.header.frame_id
        )
        if arm_base is None:
            return

        filtered = self.ema.update(arm_base)
        stable = self.stability.update(filtered)
        if stable is not None:
            self._latest_stable_target = stable

    def _transform_to_target(
        self, x: float, y: float, z: float, source_frame: str
    ) -> tuple[float, float, float] | None:
        """Transform a point from source_frame to target_frame. Returns (x,y,z) or None."""
        if not source_frame:
            return None
        try:
            ps = PointStamped()
            ps.header.frame_id = source_frame
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.point.x = float(x)
            ps.point.y = float(y)
            ps.point.z = float(z)
            tf_timeout = Duration(seconds=0.1)
            transformed = self.tf_buffer.transform(
                ps, self.target_frame, timeout=tf_timeout
            )
            return (
                transformed.point.x,
                transformed.point.y,
                transformed.point.z,
            )
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_tf_warning_time > 1.0:
                self.get_logger().warn(
                    f'TF transform [{source_frame} -> {self.target_frame}] '
                    f'failed: {exc}'
                )
                self._last_tf_warning_time = now
            return None

    # ------------------------------------------------------------------
    # Callbacks — suction / leg / arm forwarding
    # ------------------------------------------------------------------

    def suction_callback(self, msg: Bool):
        frame = pack_suction(msg.data)
        if self._write_frame(frame):
            self.get_logger().info(f'Suction {"ON" if msg.data else "OFF"}')

    def leg_callback(self, msg):
        frame = pack_leg_command(
            list(msg.joint_positions),
            list(msg.joint_velocities),
            int(msg.control_mode),
        )
        if self._write_frame(frame):
            self.get_logger().debug(
                f'Leg cmd: mode={msg.control_mode}, '
                f'pos[0]={msg.joint_positions[0]:.3f}'
            )

    def arm_pump_callback(self, msg):
        frame = pack_arm_joint_pump_command(
            list(msg.joint_positions),
            bool(msg.pump_on),
            float(msg.pump_pressure),
        )
        if self._write_frame(frame):
            self.get_logger().debug(
                f'Arm joint cmd: pump={msg.pump_on}, '
                f'pos[0]={msg.joint_positions[0]:.3f}'
            )

    # ------------------------------------------------------------------
    # Timer — send grasp / place based on MCU flag
    # ------------------------------------------------------------------

    def send_timer_callback(self):
        self._read_and_parse_feedback()

        if self._grasp_place_flag == 0:
            self._maybe_send_grasp()
        elif self._grasp_place_flag == 1:
            self._maybe_send_place()

    def _maybe_send_grasp(self):
        if self._latest_stable_target is None:
            return

        target = self._latest_stable_target
        self._latest_stable_target = None

        ok = self._send_target_immediate(target, 'grasp')
        if not ok:
            # Send failed — keep target for retry
            self._latest_stable_target = target

    def _maybe_send_place(self):
        if self._place_sent:
            return

        ok = self._send_target_immediate(self._place_target, 'place')
        if ok:
            self._place_sent = True
        # On failure, retry on next tick

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
            f'Sent {tag} arm_base=({target[0]:.3f}, '
            f'{target[1]:.3f}, {target[2]:.3f})m'
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
