"""ROS2 node that records same-frame depth estimator A/B samples to CSV."""

import csv
from pathlib import Path

import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray

from detection_3d.depth_processor import (
    compute_roi_size,
    estimate_target_point_from_roi,
    extract_roi,
    filter_depth_roi,
)
from detection_3d.geometry import map_source_pixel_to_depth_pixel, project_pixel_to_xyz


class EstimatorComparisonLoggerNode(Node):
    """
    Record baseline and optimized 3D estimates from the same input frames.

    The node is for calibration/evidence runs only. It does not publish control
    outputs and does not affect the normal detection or grasping pipeline.
    """

    def __init__(self):
        super().__init__('estimator_comparison_logger')

        self.declare_parameter('output_path', 'estimator_comparison_samples.csv')
        self.declare_parameter('append', False)
        self.declare_parameter('target_class', '')
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('sync_slop', 0.15)
        self.declare_parameter('depth_roi_size', 5)
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('detection_topic', '/detection/detections_2d')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('source_image_width', 640)
        self.declare_parameter('source_image_height', 480)
        self.declare_parameter('depth_pixel_offset_x_px', 0.0)
        self.declare_parameter('depth_pixel_offset_y_px', 0.0)
        self.declare_parameter('max_depth_m', 10.0)
        self.declare_parameter('min_depth_valid_ratio', 0.3)
        self.declare_parameter('depth_roi_ratio', 0.3)
        self.declare_parameter('depth_outlier_sigma', 2.0)
        self.declare_parameter('depth_cluster_tolerance_m', 0.03)
        self.declare_parameter('depth_min_cluster_ratio', 0.15)

        output_path = Path(self.get_parameter('output_path').value)
        append = bool(self.get_parameter('append').value)
        self.target_class = str(self.get_parameter('target_class').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        sync_slop = float(self.get_parameter('sync_slop').value)
        self.min_roi = int(self.get_parameter('depth_roi_size').value)
        depth_topic = self.get_parameter('depth_topic').value
        detection_topic = self.get_parameter('detection_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.source_w = int(self.get_parameter('source_image_width').value)
        self.source_h = int(self.get_parameter('source_image_height').value)
        self.depth_pixel_offset_x_px = float(
            self.get_parameter('depth_pixel_offset_x_px').value
        )
        self.depth_pixel_offset_y_px = float(
            self.get_parameter('depth_pixel_offset_y_px').value
        )
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.min_depth_valid_ratio = float(self.get_parameter('min_depth_valid_ratio').value)
        self.depth_roi_ratio = float(self.get_parameter('depth_roi_ratio').value)
        self.outlier_sigma = float(self.get_parameter('depth_outlier_sigma').value)
        self.depth_cluster_tolerance_m = float(
            self.get_parameter('depth_cluster_tolerance_m').value
        )
        self.depth_min_cluster_ratio = float(
            self.get_parameter('depth_min_cluster_ratio').value
        )

        self.fx: float | None = None
        self.fy: float | None = None
        self.cx: float | None = None
        self.cy: float | None = None

        self.bridge = CvBridge()
        self._sample_count = 0

        if output_path.parent != Path('.'):
            output_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = output_path.exists()
        mode = 'a' if append else 'w'
        self._file = output_path.open(mode, newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        if not append or not file_exists:
            self._writer.writerow([
                'stamp_sec',
                'stamp_nanosec',
                'detection_index',
                'class_id',
                'score',
                'center_u_px',
                'center_v_px',
                'roi_size',
                'baseline_x',
                'baseline_y',
                'baseline_z',
                'baseline_quality',
                'optimized_x',
                'optimized_y',
                'optimized_z',
                'optimized_quality',
                'optimized_offset_x_px',
                'optimized_offset_y_px',
                'optimized_valid_ratio',
                'optimized_cluster_ratio',
            ])
            self._file.flush()

        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10,
        )
        self.sub_depth = Subscriber(self, Image, depth_topic)
        self.sub_detections = Subscriber(self, Detection2DArray, detection_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_detections],
            queue_size=30,
            slop=sync_slop,
        )
        self.sync.registerCallback(self.sync_callback)

        self.get_logger().info(
            f'Estimator comparison logger ready. output={output_path}, '
            f'target_class="{self.target_class or "any"}"'
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """
        Cache camera intrinsics needed for pixel-to-XYZ projection.

        Args:
            msg: CameraInfo message from the aligned depth camera stream.
        """
        if self.fx is not None:
            return
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def sync_callback(self, depth_msg: Image, det2d_msg: Detection2DArray) -> None:
        """
        Write comparable baseline/optimized estimates for each matching detection.

        Args:
            depth_msg: Depth image synchronized with 2D detections.
            det2d_msg: YOLO 2D detections for the same scene time.
        """
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_h, depth_w = depth_image.shape[:2]
        scale_x = depth_w / self.source_w
        scale_y = depth_h / self.source_h

        for index, det2d in enumerate(det2d_msg.detections):
            if not det2d.results:
                continue

            hyp = det2d.results[0].hypothesis
            class_id = hyp.class_id
            if self.target_class and class_id != self.target_class:
                continue

            box_w = det2d.bbox.size_x * scale_x
            box_h = det2d.bbox.size_y * scale_y
            roi_size = compute_roi_size(box_w, box_h, self.depth_roi_ratio, self.min_roi)

            half = roi_size // 2
            u, v = map_source_pixel_to_depth_pixel(
                det2d.bbox.center.position.x,
                det2d.bbox.center.position.y,
                self.source_w,
                self.source_h,
                depth_w,
                depth_h,
                self.depth_pixel_offset_x_px,
                self.depth_pixel_offset_y_px,
                clamp_half_size=half,
            )

            roi = extract_roi(depth_image, u, v, roi_size)
            if roi is None:
                continue

            baseline_depth_m, baseline_quality = filter_depth_roi(
                roi,
                depth_scale=self.depth_scale,
                min_valid_ratio=self.min_depth_valid_ratio,
                outlier_sigma=self.outlier_sigma,
            )
            optimized = estimate_target_point_from_roi(
                roi,
                depth_scale=self.depth_scale,
                min_valid_ratio=self.min_depth_valid_ratio,
                outlier_sigma=self.outlier_sigma,
                cluster_tolerance_m=self.depth_cluster_tolerance_m,
                min_cluster_ratio=self.depth_min_cluster_ratio,
            )
            if optimized is None:
                continue
            if baseline_depth_m <= 0.0 or baseline_quality <= 0.0:
                continue
            if baseline_depth_m > self.max_depth_m or optimized.depth_m > self.max_depth_m:
                continue

            baseline_xyz = project_pixel_to_xyz(
                u,
                v,
                baseline_depth_m,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
            )
            optimized_xyz = project_pixel_to_xyz(
                u + optimized.offset_x_px,
                v + optimized.offset_y_px,
                optimized.depth_m,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
            )

            self._writer.writerow([
                depth_msg.header.stamp.sec,
                depth_msg.header.stamp.nanosec,
                index,
                class_id,
                float(hyp.score),
                u,
                v,
                roi_size,
                baseline_xyz[0],
                baseline_xyz[1],
                baseline_xyz[2],
                baseline_quality,
                optimized_xyz[0],
                optimized_xyz[1],
                optimized_xyz[2],
                optimized.quality,
                optimized.offset_x_px,
                optimized.offset_y_px,
                optimized.valid_ratio,
                optimized.cluster_ratio,
            ])
            self._sample_count += 1

        self._file.flush()

    def destroy_node(self) -> None:
        """
        Close the CSV file before shutting down the ROS node.
        """
        self._file.close()
        self.get_logger().info(
            f'Wrote {self._sample_count} paired estimator comparison samples'
        )
        super().destroy_node()


def main(args=None) -> None:
    """
    Start the same-frame estimator comparison logger node.

    Args:
        args: Optional ROS arguments.
    """
    rclpy.init(args=args)
    node = EstimatorComparisonLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
