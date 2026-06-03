"""
ROS2 node: OpenVINO INT8 YOLOv8 detector.
Replaces ultralytics.YOLO for CPU-optimized inference on Intel N100.

Subscribes to a color Image topic, publishes Detection2DArray + annotated Image.
"""

import os
import time

import cv2
import numpy as np
import openvino as ov
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

# ---------------------------------------------------------------------------
# YOLOv8 constants
# ---------------------------------------------------------------------------
IMGSZ = 640
CLASS_NAMES = ["Cube_food", "Cube_ins", "Cube_medicine", "Cube_tool"]
COLORS = [
    (0, 255, 0),     # Cube_food: green
    (255, 0, 0),     # Cube_ins: blue
    (0, 0, 255),     # Cube_medicine: red
    (255, 255, 0),   # Cube_tool: cyan
]


# ---------------------------------------------------------------------------
# Preprocessing — matches YOLOv8 training pipeline
# ---------------------------------------------------------------------------

def letterbox(img: np.ndarray, new_shape: int = IMGSZ):
    """Resize with aspect-ratio preservation + padding."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw = new_shape - new_w
    dh = new_shape - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (r, left, top)


def preprocess(img: np.ndarray):
    """BGR image -> (1, 3, 640, 640) float32 NCHW tensor."""
    img_padded, pad_info = letterbox(img, IMGSZ)
    img_rgb = img_padded[..., ::-1]                     # BGR -> RGB
    img_norm = img_rgb.astype(np.float32) / 255.0       # [0, 1]
    img_chw = np.transpose(img_norm, (2, 0, 1))         # HWC -> CHW
    img_batch = np.expand_dims(img_chw, axis=0)         # add batch dim
    return np.ascontiguousarray(img_batch), pad_info


# ---------------------------------------------------------------------------
# Postprocessing — parse (1,8,8400) output, apply NMS
# ---------------------------------------------------------------------------

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """cxcywh -> xyxy."""
    out = np.copy(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list:
    """Non-maximum suppression. Returns list of kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def postprocess(
    output: np.ndarray,
    pad_info: tuple,
    orig_shape: tuple,
    conf_thres: float = 0.8,
    iou_thres: float = 0.20,
) -> np.ndarray:
    """
    Parse YOLOv8 OpenVINO output -> detection array (N, 6) [x1,y1,x2,y2,conf,cls].

    output: (1, 8, 8400), channels = [cx, cy, w, h, cls0..cls3]
    """
    output = np.squeeze(output).T   # (8400, 8)
    boxes_raw = output[:, :4]        # cx, cy, w, h
    scores = output[:, 4:]           # cls0..cls3

    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)

    mask = max_scores > conf_thres
    if not mask.any():
        return np.empty((0, 6))

    boxes_raw, max_scores, class_ids = boxes_raw[mask], max_scores[mask], class_ids[mask]
    boxes = xywh2xyxy(boxes_raw)

    # Scale from model input space back to original image
    r, left, top = pad_info
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r

    h0, w0 = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

    # Per-class NMS
    results = []
    for cls_id in range(len(CLASS_NAMES)):
        idx = np.where(class_ids == cls_id)[0]
        keep = nms(boxes[idx], max_scores[idx], iou_thres)
        for k in keep:
            results.append(np.concatenate([boxes[idx][k], [max_scores[idx][k], cls_id]]))
    if not results:
        return np.empty((0, 6))
    return np.array(results)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detections(img: np.ndarray, detections: np.ndarray, fps: float) -> np.ndarray:
    """Draw bounding boxes, labels, and FPS on a copy of the image."""
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return img


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class YoloDetectorNode(Node):
    """OpenVINO INT8 YOLOv8 detection node for Intel N100 CPU."""

    def __init__(self):
        super().__init__('yolo_detector')

        # Model path — defaults to best_int8.xml in package share
        default_model = os.path.join(
            get_package_share_directory('detection_3d'), 'models', 'best_int8.xml'
        )

        self.declare_parameter('model_path', default_model)
        self.declare_parameter('confidence_threshold', 0.8)
        self.declare_parameter('iou_threshold', 0.20)
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('publish_detections_topic', '/detection/detections_2d')
        self.declare_parameter('publish_annotated_topic', '/detection/annotated_image')
        self.declare_parameter('infer_size', 640)
        self.declare_parameter('skip_frames', 0)

        model_path = self.get_parameter('model_path').value
        self.confidence = float(self.get_parameter('confidence_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        input_topic = self.get_parameter('input_topic').value
        self.publish_annotated = bool(self.get_parameter('publish_annotated_image').value)
        detections_topic = self.get_parameter('publish_detections_topic').value
        annotated_topic = self.get_parameter('publish_annotated_topic').value
        self.infer_size = int(self.get_parameter('infer_size').value)
        self.skip_frames = int(self.get_parameter('skip_frames').value)

        # ---- Load OpenVINO model ----
        self.get_logger().info(f'Loading OpenVINO model from {model_path} ...')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        core = ov.Core()
        self.get_logger().info(f'OpenVINO available devices: {core.available_devices}')
        model = core.read_model(str(model_path))
        self._compiled = core.compile_model(model, "CPU")
        # Pre-create InferRequest to avoid per-frame allocation
        self._infer_request = self._compiled.create_infer_request()
        self.get_logger().info(
            f'Model loaded. Classes: {CLASS_NAMES}, '
            f'conf={self.confidence}, iou={self.iou_threshold}'
        )

        # ---- ROS2 interfaces ----
        self.bridge = CvBridge()
        self.frame_count = 0
        self._fps_history: list[float] = []

        self.sub_image = self.create_subscription(
            Image, input_topic, self.image_callback, 10
        )
        self.pub_detections = self.create_publisher(
            Detection2DArray, detections_topic, 10
        )
        if self.publish_annotated:
            self.pub_annotated = self.create_publisher(
                Image, annotated_topic, 10
            )

        self.get_logger().info(
            f'YoloDetector ready (OpenVINO CPU). '
            f'infer_size={self.infer_size}, conf={self.confidence}, '
            f'iou={self.iou_threshold}, skip_frames={self.skip_frames}'
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        """Run single OpenVINO inference, reuse pre-created InferRequest."""
        self._infer_request.set_input_tensor(ov.Tensor(tensor))
        self._infer_request.infer()
        return self._infer_request.get_output_tensor().data

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def image_callback(self, msg: Image):
        self.frame_count += 1
        if self.skip_frames > 0 and (self.frame_count % (self.skip_frames + 1)) != 1:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        t0 = time.perf_counter()
        tensor, pad_info = preprocess(cv_image)
        output = self._infer(tensor)
        detections = postprocess(
            output, pad_info, cv_image.shape,
            conf_thres=self.confidence, iou_thres=self.iou_threshold,
        )
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000

        instant_fps = 1000.0 / ms if ms > 0 else 0.0
        self._fps_history.append(instant_fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        avg_fps = np.mean(self._fps_history)

        # ---- Publish Detection2DArray ----
        det_array = Detection2DArray()
        det_array.header = msg.header

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            cls_name = CLASS_NAMES[cls_id]

            d2 = Detection2D()
            d2.header = msg.header
            d2.id = f'{cls_name}_{i}'
            d2.bbox.center.position.x = float((x1 + x2) / 2.0)
            d2.bbox.center.position.y = float((y1 + y2) / 2.0)
            d2.bbox.size_x = float(x2 - x1)
            d2.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls_name
            hyp.hypothesis.score = float(conf)
            d2.results.append(hyp)

            det_array.detections.append(d2)

        self.pub_detections.publish(det_array)

        # ---- Publish annotated image ----
        if self.publish_annotated:
            annotated = draw_detections(cv_image.copy(), detections, avg_fps)
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
