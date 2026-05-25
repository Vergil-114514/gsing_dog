import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        self.declare_parameter('model_path', '/home/leon/code/arm_camra/src/detection_3d/models/best.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('infer_size', 640)
        self.declare_parameter('skip_frames', 0)

        model_path = self.get_parameter('model_path').value
        self.confidence = self.get_parameter('confidence_threshold').value
        input_topic = self.get_parameter('input_topic').value
        self.publish_annotated = self.get_parameter('publish_annotated_image').value
        self.infer_size = self.get_parameter('infer_size').value
        self.skip_frames = self.get_parameter('skip_frames').value

        self.get_logger().info(f'Loading YOLO model from {model_path} (CPU)...')
        self.model = YOLO(model_path)
        self.model.to('cpu')
        self.class_names = self.model.names
        self.get_logger().info(f'Model loaded. Classes: {self.class_names}')

        self.bridge = CvBridge()
        self.frame_count = 0

        self.sub_image = self.create_subscription(
            Image, input_topic, self.image_callback, 10
        )
        self.pub_detections = self.create_publisher(
            Detection2DArray, '/detection/detections_2d', 10
        )
        if self.publish_annotated:
            self.pub_annotated = self.create_publisher(
                Image, '/detection/annotated_image', 10
            )

        self.get_logger().info(
            f'YoloDetector ready (CPU). infer_size={self.infer_size}, '
            f'conf={self.confidence}, skip_frames={self.skip_frames}'
        )

    def image_callback(self, msg: Image):
        self.frame_count += 1
        if self.skip_frames > 0 and (self.frame_count % (self.skip_frames + 1)) != 1:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(
            cv_image,
            conf=self.confidence,
            imgsz=self.infer_size,
            device='cpu',
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        det_array = Detection2DArray()
        det_array.header = msg.header

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = self.class_names[cls_id]

            det = Detection2D()
            det.header = msg.header
            det.id = f'{cls_name}_{i}'

            det.bbox.center.position.x = float((x1 + x2) / 2.0)
            det.bbox.center.position.y = float((y1 + y2) / 2.0)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls_name
            hyp.hypothesis.score = conf
            det.results.append(hyp)

            det_array.detections.append(det)

        self.pub_detections.publish(det_array)

        if self.publish_annotated:
            annotated = result.plot()
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ann_msg.header = msg.header
            self.pub_annotated.publish(ann_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
