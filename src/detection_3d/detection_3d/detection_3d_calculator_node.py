"""
ROS2 node: 2D detection → 3D coordinate calculator.

Improvements over v1:
  - Adaptive ROI (scaled to detection box size) instead of fixed 5x5.
  - Outlier-filtered depth with quality scoring.
  - Target selection: pick best candidate by confidence × depth quality.
  - Coordinate jump detection.
  - Sliding-window coordinate stabilizer (only publish stable results).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import TransformBroadcaster

from detection_3d.geometry import (
    map_source_pixel_to_depth_pixel,
    project_pixel_to_xyz,
)
from detection_3d.depth_processor import (
    compute_roi_size,
    extract_roi,
    estimate_target_point_from_roi,
    filter_depth_roi,
)
from detection_3d.target_selector import (
    TargetCandidate,
    compute_composite_score,
    select_best_target,
    detect_coordinate_jump,
)
from detection_3d.coordinate_stabilizer import CoordinateStabilizer


class Detection3DCalculatorNode(Node):
    """2D detections → 3D coordinates with depth quality filtering and stabilization."""

    def __init__(self):
        super().__init__('detection_3d_calculator')

        # ---- depth ----
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('sync_slop', 0.15)
        self.declare_parameter('depth_roi_size', 5)          # fallback min ROI
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('detection_topic', '/detection/detections_2d')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('publish_detections_3d_topic', '/detection/detections_3d')
        self.declare_parameter('publish_markers_topic', '/detection/markers')
        self.declare_parameter('source_image_width', 640)
        self.declare_parameter('source_image_height', 480)
        self.declare_parameter('depth_pixel_offset_x_px', 0.0)
        self.declare_parameter('depth_pixel_offset_y_px', 0.0)
        self.declare_parameter('debug_projection_log', False)
        self.declare_parameter('max_depth_m', 10.0)

        # ---- new: depth quality ----
        self.declare_parameter('min_depth_valid_ratio', 0.3)
        self.declare_parameter('depth_roi_ratio', 0.3)
        self.declare_parameter('depth_outlier_sigma', 2.0)
        self.declare_parameter('depth_cluster_tolerance_m', 0.03)
        self.declare_parameter('depth_min_cluster_ratio', 0.15)
        self.declare_parameter('depth_estimator_mode', 'cluster_centroid')

        # ---- new: stability ----
        self.declare_parameter('max_depth_variance_m2', 0.0002)
        self.declare_parameter('coordinate_jump_threshold_m', 0.03)
        self.declare_parameter('stable_window_size', 7)
        self.declare_parameter('coordinate_output_mode', 'median')

        # === resolve ===

        self.depth_scale = float(self.get_parameter('depth_scale').value)
        sync_slop = float(self.get_parameter('sync_slop').value)
        self.min_roi = int(self.get_parameter('depth_roi_size').value)
        self.camera_frame = self.get_parameter('camera_frame').value
        depth_topic = self.get_parameter('depth_topic').value
        detection_topic = self.get_parameter('detection_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        det3d_topic = self.get_parameter('publish_detections_3d_topic').value
        markers_topic = self.get_parameter('publish_markers_topic').value
        self.source_w = int(self.get_parameter('source_image_width').value)
        self.source_h = int(self.get_parameter('source_image_height').value)
        self.depth_pixel_offset_x_px = float(
            self.get_parameter('depth_pixel_offset_x_px').value
        )
        self.depth_pixel_offset_y_px = float(
            self.get_parameter('depth_pixel_offset_y_px').value
        )
        self.debug_projection_log = bool(
            self.get_parameter('debug_projection_log').value
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
        self.depth_estimator_mode = str(
            self.get_parameter('depth_estimator_mode').value
        ).strip().lower()
        if self.depth_estimator_mode not in ('cluster_centroid', 'center_median'):
            self.get_logger().warning(
                f'Unknown depth_estimator_mode={self.depth_estimator_mode!r}; '
                'using cluster_centroid'
            )
            self.depth_estimator_mode = 'cluster_centroid'

        self.max_depth_variance_m2 = float(self.get_parameter('max_depth_variance_m2').value)
        self.jump_threshold_m = float(self.get_parameter('coordinate_jump_threshold_m').value)
        stable_window_size = int(self.get_parameter('stable_window_size').value)
        coordinate_output_mode = str(
            self.get_parameter('coordinate_output_mode').value
        ).strip().lower()

        # === camera intrinsics (populated from CameraInfo) ===
        self.fx: float | None = None
        self.fy: float | None = None
        self.cx: float | None = None
        self.cy: float | None = None
        self.depth_w: int | None = None
        self.depth_h: int | None = None

        # === state ===
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self._stabilizer = CoordinateStabilizer(
            window_size=stable_window_size,
            max_variance_m2=self.max_depth_variance_m2,
            output_mode=coordinate_output_mode,
        )
        self._prev_best_pos: tuple[float, float, float] | None = None
        self._prev_best_class: str = ""

        # === subscribers ===
        self.sub_camera_info = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10
        )

        self.sub_depth = Subscriber(self, Image, depth_topic)
        self.sub_detections = Subscriber(self, Detection2DArray, detection_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_detections],
            queue_size=30,
            slop=sync_slop,
        )
        self.sync.registerCallback(self.sync_callback)

        # === publishers ===
        self.pub_detections_3d = self.create_publisher(
            Detection3DArray, det3d_topic, 10
        )
        self.pub_markers = self.create_publisher(
            MarkerArray, markers_topic, 10
        )

        self.get_logger().info(
            f'Detection3DCalculator ready. depth_scale={self.depth_scale}, '
            f'roi_ratio={self.depth_roi_ratio}, min_roi={self.min_roi}, '
            f'outlier_sigma={self.outlier_sigma}, '
            f'estimator={self.depth_estimator_mode}, '
            f'cluster_tol={self.depth_cluster_tolerance_m}m, '
            f'min_cluster_ratio={self.depth_min_cluster_ratio}, '
            f'valid_ratio>={self.min_depth_valid_ratio}, '
            f'stable_window={stable_window_size}, '
            f'coord_output={coordinate_output_mode}, '
            f'jump_thresh={self.jump_threshold_m}m, '
            f'source_res={self.source_w}x{self.source_h}, '
            f'depth_pixel_offset=({self.depth_pixel_offset_x_px:.2f}, '
            f'{self.depth_pixel_offset_y_px:.2f})px'
        )

    # ------------------------------------------------------------------
    # Camera intrinsics
    # ------------------------------------------------------------------

    def camera_info_callback(self, msg: CameraInfo):
        if self.fx is not None:
            return
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.depth_w = msg.width
        self.depth_h = msg.height
        self.get_logger().info(
            f'Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, '
            f'cx={self.cx:.1f}, cy={self.cy:.1f}, '
            f'depth_res={self.depth_w}x{self.depth_h}'
        )

    # ------------------------------------------------------------------
    # Sync callback — main pipeline
    # ------------------------------------------------------------------

    def sync_callback(self, depth_msg: Image, det2d_msg: Detection2DArray):
        if self.fx is None:
            return

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_h, depth_w = depth_image.shape[:2]

        scale_x = depth_w / self.source_w
        scale_y = depth_h / self.source_h

        # ---- Phase 1: Build scored candidates ----
        candidates: list[TargetCandidate] = []

        for det2d in det2d_msg.detections:
            if not det2d.results:
                continue

            cls_name = det2d.results[0].hypothesis.class_id
            score = float(det2d.results[0].hypothesis.score)

            # Adaptive ROI based on detection box size
            box_w = det2d.bbox.size_x * scale_x
            box_h = det2d.bbox.size_y * scale_y
            roi_size = compute_roi_size(box_w, box_h, self.depth_roi_ratio, self.min_roi)

            # Center in depth image coordinates. The extra offset is a field
            # calibration knob for residual RGB/depth alignment error.
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

            # Extract and filter depth ROI
            roi = extract_roi(depth_image, u, v, roi_size)
            if roi is None:
                continue

            target_offset_x = 0.0
            target_offset_y = 0.0
            if self.depth_estimator_mode == 'center_median':
                depth_m, quality = filter_depth_roi(
                    roi,
                    depth_scale=self.depth_scale,
                    min_valid_ratio=self.min_depth_valid_ratio,
                    outlier_sigma=self.outlier_sigma,
                )
            else:
                estimate = estimate_target_point_from_roi(
                    roi,
                    depth_scale=self.depth_scale,
                    min_valid_ratio=self.min_depth_valid_ratio,
                    outlier_sigma=self.outlier_sigma,
                    cluster_tolerance_m=self.depth_cluster_tolerance_m,
                    min_cluster_ratio=self.depth_min_cluster_ratio,
                )

                if estimate is None:
                    continue

                depth_m = estimate.depth_m
                quality = estimate.quality
                target_offset_x = estimate.offset_x_px
                target_offset_y = estimate.offset_y_px

            if depth_m <= 0.0 or depth_m > self.max_depth_m or quality <= 0.0:
                continue

            target_u = u + target_offset_x
            target_v = v + target_offset_y

            # Project depth-cluster centroid to 3D
            x, y, z = project_pixel_to_xyz(
                target_u, target_v, depth_m, self.fx, self.fy, self.cx, self.cy
            )
            if self.debug_projection_log:
                self.get_logger().info(
                    f'PROJECTION cls={cls_name} '
                    f'src=({det2d.bbox.center.position.x:.1f}, '
                    f'{det2d.bbox.center.position.y:.1f}) '
                    f'depth_center=({u}, {v}) '
                    f'target=({target_u:.1f}, {target_v:.1f}) '
                    f'depth={depth_m:.3f}m xyz=({x:.3f}, {y:.3f}, {z:.3f})m'
                )

            composite = compute_composite_score(score, quality)

            candidates.append(TargetCandidate(
                x=x, y=y, z=z,
                class_name=cls_name,
                confidence=score,
                depth_quality=quality,
                composite_score=composite,
                source=det2d,
            ))

        # ---- Phase 2: Select best target ----
        best = select_best_target(candidates)
        if best is None:
            return

        # ---- Phase 3: Coordinate jump check ----
        new_pos = (best.x, best.y, best.z)

        # Reset stabilizer if target class changed
        if best.class_name != self._prev_best_class:
            self._stabilizer.reset()
            self._prev_best_pos = None
            self._prev_best_class = best.class_name

        if detect_coordinate_jump(new_pos, self._prev_best_pos, self.jump_threshold_m):
            self.get_logger().debug(
                f'Jump detected: {best.class_name} {new_pos}, '
                f'prev={self._prev_best_pos}, resetting tracker'
            )
            self._prev_best_pos = new_pos
            self._stabilizer.reset()
            return

        self._prev_best_pos = new_pos

        # ---- Phase 4: Coordinate stabilizer ----
        stable = self._stabilizer.update(new_pos)
        if stable is None:
            return

        sx, sy, sz = stable

        # ---- Phase 5: Publish ----
        det3d_array = Detection3DArray()
        det3d_array.header.stamp = depth_msg.header.stamp
        det3d_array.header.frame_id = depth_msg.header.frame_id or self.camera_frame

        det3d = Detection3D()
        det3d.header = det3d_array.header
        det3d.id = f'{best.class_name}_stable'
        det3d.bbox.center.position.x = sx
        det3d.bbox.center.position.y = sy
        det3d.bbox.center.position.z = sz
        det3d.bbox.center.orientation.w = 1.0
        det3d.bbox.size.x = 0.1
        det3d.bbox.size.y = 0.1
        det3d.bbox.size.z = 0.1

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = best.class_name
        hyp.hypothesis.score = best.confidence
        hyp.pose.pose.position.x = sx
        hyp.pose.pose.position.y = sy
        hyp.pose.pose.position.z = sz
        hyp.pose.pose.orientation.w = 1.0
        det3d.results.append(hyp)
        det3d_array.detections.append(det3d)

        self.pub_detections_3d.publish(det3d_array)

        # ---- TF + markers ----
        t = TransformStamped()
        t.header = det3d_array.header
        t.child_frame_id = f'detected_{best.class_name}'
        t.transform.translation.x = sx
        t.transform.translation.y = sy
        t.transform.translation.z = sz
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform([t])

        marker_array = MarkerArray()

        sphere = Marker()
        sphere.header = det3d_array.header
        sphere.ns = 'detections'
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = sx
        sphere.pose.position.y = sy
        sphere.pose.position.z = sz
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 0.05
        sphere.scale.y = 0.05
        sphere.scale.z = 0.05
        sphere.color.g = 1.0
        sphere.color.a = 0.8
        sphere.lifetime.nanosec = 500_000_000
        marker_array.markers.append(sphere)

        text = Marker()
        text.header = det3d_array.header
        text.ns = 'labels'
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = sx
        text.pose.position.y = sy - 0.05
        text.pose.position.z = sz
        text.pose.orientation.w = 1.0
        text.scale.z = 0.03
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = f'{best.class_name} ({sz:.2f}m)'
        text.lifetime.nanosec = 500_000_000
        marker_array.markers.append(text)

        self.pub_markers.publish(marker_array)

        self.get_logger().info(
            f'{best.class_name} stable at ({sx:.3f}, {sy:.3f}, {sz:.3f})m '
            f'conf={best.confidence:.2f} q={best.depth_quality:.2f} '
            f'score={best.composite_score:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = Detection3DCalculatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
