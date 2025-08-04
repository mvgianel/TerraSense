#!/usr/bin/env python3
'''
Heavily inspired by https://github.com/amd/Kria-RoboticsAI/blob/main/files/ROSAI/camera_input/rosai_camera/rosai_camera/rosai_camera_demo.py
'''

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
import numpy as np
import sys
import os
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
import tensorflow as tf
import time
from collections import deque

# import CV BRIDGE
from cv_bridge import CvBridge, CvBridgeError
import cv2

sys.path.append('/usr/lib/python3.10/site-packages')
sys.path.append('/usr/local/share/pynq-venv/lib/python3.10/site-packages')
# the above path is needed by pynq_dpu

ml_model = 'quantized_resnet18_int8.tflite'
class_names = ['cobblestonebrick', 'dirtground', 'grass', 'pavement', 'sand', 'stairs']

class MLPublisher(Node):
    def __init__(self):
        super().__init__('ml_publisher')
        self.bridge = CvBridge()
        self.subscriber_ = self.create_subscription(Image, '/camera/camera/color/image_raw', self.listener_callback, 10)
        self.get_logger().info('[INFO] __init__, Create Subscription to rgb image...')
        self.subscriber_  # prevent unused variable warning
        self.publisher_ = self.create_publisher(String, 'terrain_class', 10)
        # Add terrain distance 
        # self.publisher_ = self.create_publisher(Image, 'terrain_dist', 10)

        # Load quantized model
        self.model_path = os.path.join(get_package_share_directory('terra_sense'), 'config', ml_model)
        self.get_logger().info("MODEL="+self.model_path)
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.in_dtype = self.input_details[0]['dtype']
        # self.get_logger().info('dtype='+str(self.dtype))

        # --- metrics state ---
        self.start_time = time.perf_counter()
        self.first_msg_time = None           # perf_counter at first frame
        self.last_msg_time  = None           # perf_counter at last frame
        self.total_frames   = 0
        self.model_ms = []                   # or: deque(maxlen=10000)
        self.end2end_ms = []
        
        
        _, self.inH, self.inW, _ = self.input_details[0]['shape']
        self.in_scale,  self.in_zero  = self.input_details[0]['quantization']   # for input
        self.out_scale, self.out_zero = self.output_details[0]['quantization']
        self.get_logger().info(f"Input dtype={self.in_dtype}, q=(scale={self.in_scale}, zp={self.in_zero})")
        self.get_logger().info(f"Output q=(scale={self.out_scale}, zp={self.out_zero})")
        self.get_logger().info(f"Model expects (H,W)=({self.inH},{self.inW})")
        self.get_logger().info('[INFO] __init__ exiting...')
        self.get_logger().info('========== Starting classification ==========')

    def normalize(self, image):
        image=image/255.0
        image=image-0.5
        image=image*2
        return image
    
    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def listener_callback(self, msg):
        #self.get_logger().info("Starting of listener callback...")
        t_cb_start = time.perf_counter()
        if self.first_msg_time is None:
            self.first_msg_time = t_cb_start

        # Preprocess image
        cv2_image_org = self.bridge.imgmsg_to_cv2(msg,desired_encoding="rgb8")
        cv2_image = cv2.resize(cv2_image_org, (self.inW, self.inH), interpolation=cv2.INTER_LINEAR)
        cv2_image = np.asarray(cv2_image, dtype=np.float32)
        cv2_image = self.normalize(cv2_image)

        # Quantized model: map to quantized domain
        cv2_image_q = np.round(cv2_image / self.in_scale + self.in_zero).astype(self.in_dtype)
        cv2_image_q = np.expand_dims(cv2_image_q, axis=0)

        # Inference
        t_inf_start = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], cv2_image_q)
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        t_inf_end = time.perf_counter()
        model_ms = (t_inf_end - t_inf_start) * 1e3
        self.model_ms.append(model_ms)
        # dtype = raw_output.dtype
        # self.get_logger().info('dtype='+str(dtype))

        # Dequantize Output
        dequantized = self.out_scale * (raw_output.astype(np.float32) - self.out_zero)
        # self.get_logger().info("y.dim="+str(dequantized.ndim))
        # self.get_logger().info("y.shape[-1]="+str(dequantized.shape[-1]))

        # Prediction
        probs = self.softmax(dequantized)
        # self.get_logger().info(str(probs))
        predicted_index = np.argmax(probs)
        # print(f"Dequantized prediction vector: {probabilities}")
        # print(f"Predicted class index: {predicted_index}")
        # print(f"Predicted class name: {class_names[predicted_index]}")

        #self.get_logger().info("prediction="+str(prediction))
        msg = String()
        msg.data = class_names[predicted_index]
        self.publisher_.publish(msg)

        t_cb_end = time.perf_counter()
        end2end_ms = (t_cb_end - t_cb_start) * 1e3
        self.end2end_ms.append(end2end_ms)

        self.total_frames += 1
        self.last_msg_time = t_cb_end

    def _report_metrics(self):
        if self.first_msg_time is None or self.last_msg_time is None or self.total_frames == 0:
            self.get_logger().info("[metrics] No frames processed.")
            return

        elapsed_s = self.last_msg_time - self.first_msg_time
        fps = self.total_frames / elapsed_s if elapsed_s > 0 else float('nan')

        m = np.array(self.model_ms, dtype=np.float32)
        e = np.array(self.end2end_ms, dtype=np.float32)

        def pct(a, p): return float(np.percentile(a, p)) if a.size else float('nan')

        self.get_logger().info(
            "\n====== ML Metrics (on shutdown) ======\n"
            f"Frames: {self.total_frames}\n"
            f"Elapsed: {elapsed_s:.3f} s\n"
            f"Throughput (FPS): {fps:.2f}\n"
            f"Model latency  ms  -> mean {m.mean():.2f} | p50 {pct(m,50):.2f} | p95 {pct(m,95):.2f} | max {m.max():.2f}\n"
            f"End-to-end ms     -> mean {e.mean():.2f} | p50 {pct(e,50):.2f} | p95 {pct(e,95):.2f} | max {e.max():.2f}\n"
            "======================================"
        )

    def destroy_node(self):
        try:
            self._report_metrics()   # print FPS/latency once at shutdown
        finally:
            return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MLPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # This triggers your overridden destroy_node(), then shuts down rclpy.
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
