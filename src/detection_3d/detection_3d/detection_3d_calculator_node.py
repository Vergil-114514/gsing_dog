import numpy as np
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

from detection_3d.geometry import project_pixel_to_xyz


class Detection3DCalculatorNode(Node):

    def __init__(self):
        super().__init__('detection_3d_calculator')

        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('sync_slop', 0.15)
        self.declare_parameter('depth_roi_size', 5)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('detection_topic', '/detection/detections_2d')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('publish_detections_3d_topic', '/detection/detections_3d')
        self.declare_parameter('publish_markers_topic', '/detection/markers')
        self.declare_parameter('source_image_width', 640)
        self.declare_parameter('source_image_height', 480)
        self.declare_parameter('max_depth_m', 10.0)

        self.depth_scale = self.get_parameter('depth_scale').value
        sync_slop = self.get_parameter('sync_slop').value
        self.roi_size = self.get_parameter('depth_roi_size').value
        self.camera_frame = self.get_parameter('camera_frame').value
        depth_topic = self.get_parameter('depth_topic').value
        detection_topic = self.get_parameter('detection_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        det3d_topic = self.get_parameter('publish_detections_3d_topic').value
        markers_topic = self.get_parameter('publish_markers_topic').value
        self.source_w = self.get_parameter('source_image_width').value
        self.source_h = self.get_parameter('source_image_height').value
        self.max_depth_m = self.get_parameter('max_depth_m').value

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.depth_w = None
        self.depth_h = None

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

        self.pub_detections_3d = self.create_publisher(
            Detection3DArray, det3d_topic, 10
        )
        self.pub_markers = self.create_publisher(
            MarkerArray, markers_topic, 10
        )

        self.get_logger().info(
            f'Detection3DCalculator ready. depth_scale={self.depth_scale}, '
            f'roi={self.roi_size}x{self.roi_size}, frame={self.camera_frame}, '
            f'source_res={self.source_w}x{self.source_h}'
        )

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

    def sync_callback(self, depth_msg: Image, det2d_msg: Detection2DArray):
        if self.fx is None:
            return

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_h, depth_w = depth_image.shape[:2]
        half = self.roi_size // 2

        scale_x = depth_w / self.source_w
        scale_y = depth_h / self.source_h

        det3d_array = Detection3DArray()
        det3d_array.header.stamp = depth_msg.header.stamp
        det3d_array.header.frame_id = depth_msg.header.frame_id or self.camera_frame

        marker_array = MarkerArray()
        marker_id = 0
        tf_list = []

        for det2d in det2d_msg.detections:
            if not det2d.results:
                continue

            cls_name = det2d.results[0].hypothesis.class_id
            score = det2d.results[0].hypothesis.score

            u = int(det2d.bbox.center.position.x * scale_x)
            v = int(det2d.bbox.center.position.y * scale_y)
            u = max(half, min(u, depth_w - 1 - half))
            v = max(half, min(v, depth_h - 1 - half))

            roi = depth_image[v - half:v + half + 1, u - half:u + half + 1]
            valid = roi[roi > 0]
            if len(valid) == 0:
                continue

            depth_val = float(np.median(valid))
            z = depth_val * self.depth_scale
            if z <= 0.0 or z > self.max_depth_m:
                continue

            x, y, z = project_pixel_to_xyz(u, v, z, self.fx, self.fy, self.cx, self.cy)

            det3d = Detection3D()
            det3d.header = det3d_array.header
            det3d.id = det2d.id
            det3d.bbox.center.position.x = x
            det3d.bbox.center.position.y = y
            det3d.bbox.center.position.z = z
            det3d.bbox.center.orientation.w = 1.0
            det3d.bbox.size.x = 0.1
            det3d.bbox.size.y = 0.1
            det3d.bbox.size.z = 0.1

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls_name
            hyp.hypothesis.score = score
            hyp.pose.pose.position.x = x
            hyp.pose.pose.position.y = y
            hyp.pose.pose.position.z = z
            hyp.pose.pose.orientation.w = 1.0
            det3d.results.append(hyp)
            det3d_array.detections.append(det3d)

            t = TransformStamped()
            t.header = det3d_array.header
            t.child_frame_id = f'detected_{cls_name}_{det2d.id}'
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.w = 1.0
            tf_list.append(t)

            sphere = Marker()
            sphere.header = det3d_array.header
            sphere.ns = 'detections'
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.05
            sphere.scale.y = 0.05
            sphere.scale.z = 0.05
            sphere.color.g = 1.0
            sphere.color.a = 0.8
            sphere.lifetime.nanosec = 500_000_000
            marker_array.markers.append(sphere)
            marker_id += 1

            text = Marker()
            text.header = det3d_array.header
            text.ns = 'labels'
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y - 0.05
            text.pose.position.z = z
            text.pose.orientation.w = 1.0
            text.scale.z = 0.03
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f'{cls_name} ({z:.2f}m)'
            text.lifetime.nanosec = 500_000_000
            marker_array.markers.append(text)
            marker_id += 1

            self.get_logger().info(
                f'{cls_name} at ({x:.3f}, {y:.3f}, {z:.3f})m conf={score:.2f}'
            )

        self.pub_detections_3d.publish(det3d_array)
        if tf_list:
            self.tf_broadcaster.sendTransform(tf_list)
        if marker_array.markers:
            self.pub_markers.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = Detection3DCalculatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
