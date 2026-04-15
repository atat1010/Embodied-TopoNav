import threading
import time
from typing import Set

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

class YoloMaskNode(Node):
    def __init__(self) -> None:
        super().__init__('yolo_mask_node')

        self.declare_parameter('input_topic', '/camera/rgb/image_color')
        self.declare_parameter('combined_mask_topic', '/semantic/combined_mask')
        self.declare_parameter('overlay_topic', '/semantic/overlay')
        self.declare_parameter('model_path', 'yolov8n-seg.pt')
        self.declare_parameter('conf', 0.35)
        self.declare_parameter('iou', 0.5)
        self.declare_parameter('device', 'auto')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('half', True)
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('target_classes', [0])

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.combined_mask_topic = self.get_parameter('combined_mask_topic').get_parameter_value().string_value
        self.overlay_topic = self.get_parameter('overlay_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf = self.get_parameter('conf').get_parameter_value().double_value
        self.iou = self.get_parameter('iou').get_parameter_value().double_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value
        self.half = self.get_parameter('half').get_parameter_value().bool_value
        self.publish_overlay = self.get_parameter('publish_overlay').get_parameter_value().bool_value
        self.target_classes: Set[int] = set(
            self.get_parameter('target_classes').get_parameter_value().integer_array_value
        )

        self.combined_mask_pub = self.create_publisher(Image, self.combined_mask_topic, 10)
        self.overlay_pub = self.create_publisher(Image, self.overlay_topic, 10)
        self.bridge = CvBridge()

        self._busy = False
        self._lock = threading.Lock()
        self._frames = 0
        self._dropped_busy = 0
        self._processed_window = 0
        self._dropped_window = 0
        self._latency_sum_ms = 0.0
        self._combined_convert_sum_ms = 0.0
        self._overlay_convert_sum_ms = 0.0
        self._window_start = time.monotonic()

        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.get_logger().error(
                'Failed to import ultralytics. Install it first: pip install ultralytics. '
                f'Import error: {exc}'
            )
            raise

        if self.device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but unavailable, falling back to CPU.')
            self.device = 'cpu'
        if self.half and not self.device.startswith('cuda'):
            self.get_logger().warn('half=True only benefits CUDA, disabling half precision on CPU.')
            self.half = False

        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.get_logger().info(
            f'YOLO model loaded: {self.model_path}, targets={sorted(self.target_classes)}'
        )
        self.get_logger().info(
            'combined_mask协议: 0~254=类别ID, 255=背景; 动态类剔除下沉到C++节点执行'
        )
        self.get_logger().info(
            f'YOLO inference config: device={self.device}, imgsz={self.imgsz}, half={self.half}'
        )

        self.sub = self.create_subscription(Image, self.input_topic, self.image_cb, 10)
        self.get_logger().info(
            f'Subscribed to {self.input_topic}, publishing combined mask to {self.combined_mask_topic}'
        )

    def image_cb(self, msg: Image) -> None:
        with self._lock:
            if self._busy:
                self._dropped_busy += 1
                self._dropped_window += 1
                return
            self._busy = True

        try:
            t0 = time.monotonic()
            bgr = self.image_msg_to_bgr(msg)
            combined_mask, overlay = self.infer_mask(bgr)

            t_conv = time.monotonic()
            combined_mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding='mono8')
            self._combined_convert_sum_ms += (time.monotonic() - t_conv) * 1000.0
            combined_mask_msg.header = msg.header
            self.combined_mask_pub.publish(combined_mask_msg)

            if self.publish_overlay:
                t_conv = time.monotonic()
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                self._overlay_convert_sum_ms += (time.monotonic() - t_conv) * 1000.0
                overlay_msg.header = msg.header
                self.overlay_pub.publish(overlay_msg)

            self._frames += 1
            self._processed_window += 1
            self._latency_sum_ms += (time.monotonic() - t0) * 1000.0
            # if self._frames % 30 == 0:
            #     self.get_logger().info(f'Processed frames: {self._frames}')
            self._maybe_log_stats()
        except Exception as exc:
            self.get_logger().error(f'YOLO inference failed: {exc}')
        finally:
            with self._lock:
                self._busy = False

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        elapsed = now - self._window_start
        if elapsed < 1.0:
            return

        yolo_fps = self._processed_window / elapsed
        dropped_fps = self._dropped_window / elapsed
        avg_latency_ms = self._latency_sum_ms / self._processed_window if self._processed_window > 0 else 0.0
        avg_combined_convert_ms = self._combined_convert_sum_ms / self._processed_window if self._processed_window > 0 else 0.0
        avg_overlay_convert_ms = self._overlay_convert_sum_ms / self._processed_window if self._processed_window > 0 else 0.0
        self.get_logger().info(
            f'YOLO FPS({elapsed:.1f}s): {yolo_fps:.1f} | '
            f'avg infer: {avg_latency_ms:.1f} ms | '
            f'convert(combined/overlay): '
            f'{avg_combined_convert_ms:.2f}/{avg_overlay_convert_ms:.2f} ms | '
            f'dropped(busy): {self._dropped_window} ({dropped_fps:.1f} fps)'
        )

        self._window_start = now
        self._processed_window = 0
        self._dropped_window = 0
        self._latency_sum_ms = 0.0
        self._combined_convert_sum_ms = 0.0
        self._overlay_convert_sum_ms = 0.0

    def infer_mask(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = bgr.shape[:2]
        combined_mask = np.full((h, w), 255, dtype=np.uint8)
        overlay = bgr.copy()

        results = self.model.predict(
            bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
        )
        if not results:
            return combined_mask, overlay

        result = results[0]
        if result.masks is None or result.boxes is None:
            return combined_mask, overlay

        cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32)
        masks = result.masks.data.cpu().numpy()

        for i, cls_id in enumerate(cls_ids):
            dyn = masks[i] > 0.5    # mask[i] 依旧是全景图(原始尺寸)
            if dyn.shape[0] != h or dyn.shape[1] != w:
                dyn = cv2.resize(dyn.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

            # 协议: 0~254为类别ID, 255为背景。
            combined_mask[dyn] = np.uint8(cls_id)
            if self.publish_overlay and cls_id in self.target_classes:
                overlay[dyn] = (0, 0, 255)

        if self.publish_overlay:
            overlay = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)

        return combined_mask, overlay

    def image_msg_to_bgr(self, msg: Image) -> np.ndarray:
        if msg.encoding not in ('rgb8', 'bgr8'):
            raise ValueError(f'Unsupported input encoding: {msg.encoding}, expected rgb8/bgr8')

        row = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step))
        img = row[:, :msg.width * 3].reshape((msg.height, msg.width, 3))
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

def main() -> None:
    rclpy.init()
    node = YoloMaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
