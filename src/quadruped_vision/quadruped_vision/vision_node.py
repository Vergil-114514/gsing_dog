"""
vision_node — Object detection & localization for the quadruped platform.

Placeholder skeleton.  Replace with real inference pipeline (YOLO, RT-DETR,
or similar) that publishes detected objects and their 3D positions.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VisionNode(Node):
    """Vision recognition node (skeleton)."""

    def __init__(self):
        super().__init__('vision_node')

        # ---- Parameters ----
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('camera_topic', '/camera/image_raw')

        # ---- Publishers ----
        self.result_pub = self.create_publisher(String, 'vision_result', 10)

        # ---- Subscribers (placeholder — subscribe to actual camera/image topic) ----
        # self.image_sub = self.create_subscription(
        #     Image, self.get_parameter('camera_topic').value, self.image_callback, 10)

        # ---- Inference timer (placeholder) ----
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.get_logger().info('Vision node started (skeleton)')

    def timer_callback(self):
        """Placeholder: publish empty result at 1 Hz."""
        msg = String()
        msg.data = ''
        self.result_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
