"""ROS2 node that records stable 3D detection samples to CSV."""

import csv
from pathlib import Path

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray


class CoordinateSampleLoggerNode(Node):
    """
    Record Detection3DArray samples for offline stability/accuracy evaluation.

    The logger is intentionally separate from the detection pipeline so it can
    be enabled during calibration runs without changing runtime behavior.
    """

    def __init__(self):
        super().__init__('coordinate_sample_logger')

        self.declare_parameter('detection_topic', '/detection/detections_3d')
        self.declare_parameter('output_path', 'detection_samples.csv')
        self.declare_parameter('target_class', '')
        self.declare_parameter('append', False)

        detection_topic = self.get_parameter('detection_topic').value
        output_path = Path(self.get_parameter('output_path').value)
        self.target_class = self.get_parameter('target_class').value
        append = bool(self.get_parameter('append').value)

        if output_path.parent != Path('.'):
            output_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = output_path.exists()
        mode = 'a' if append else 'w'
        self._file = output_path.open(mode, newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._sample_count = 0

        if not append or not file_exists:
            self._writer.writerow([
                'stamp_sec',
                'stamp_nanosec',
                'detection_id',
                'class_id',
                'score',
                'x',
                'y',
                'z',
            ])
            self._file.flush()

        self.sub_det = self.create_subscription(
            Detection3DArray,
            detection_topic,
            self.detection_callback,
            10,
        )

        self.get_logger().info(
            f'Coordinate sample logger ready. topic={detection_topic}, '
            f'output={output_path}, target_class="{self.target_class or "any"}"'
        )

    def detection_callback(self, msg: Detection3DArray) -> None:
        """
        Write each matching 3D detection as one CSV row.

        Args:
            msg: Detection3DArray containing stable target coordinates.
        """
        for det in msg.detections:
            if not det.results:
                continue

            hyp = det.results[0].hypothesis
            class_id = hyp.class_id
            if self.target_class and class_id != self.target_class:
                continue

            pos = det.results[0].pose.pose.position
            self._writer.writerow([
                msg.header.stamp.sec,
                msg.header.stamp.nanosec,
                det.id,
                class_id,
                float(hyp.score),
                float(pos.x),
                float(pos.y),
                float(pos.z),
            ])
            self._sample_count += 1

        self._file.flush()

    def destroy_node(self):
        self._file.close()
        self.get_logger().info(f'Wrote {self._sample_count} coordinate samples')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CoordinateSampleLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
