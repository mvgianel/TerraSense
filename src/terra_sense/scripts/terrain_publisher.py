## OOOOOLDDD

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
import time

from cv_bridge import CvBridge, CvBridgeError
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer


from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time

sys.path.append('/usr/lib/python3.10/site-packages')
sys.path.append('/usr/local/share/pynq-venv/lib/python3.10/site-packages')
# the above path is needed by pynq_dpu
from pynq_dpu import DpuOverlay

ml_model = 'zcu102_q_train2_2_resnet18_terraset6_91acc_19jul.h5.xmodel'
class_names = ['cobblestonebrick', 'dirtground', 'grass', 'pavement', 'sand', 'stairs']

CLASS_COST = {
    'pavement': 10,
    'cobblestonebrick': 80,
    'dirtground': 100,
    'grass': 150,
    'sand': 200,
    'stairs': 220,   # keep < 100 here; we'll map 100 -> lethal in the layer if desired
}

class MLPublisher(Node):
    def __init__(self):
        super().__init__('ml_publisher')

        # ROS I/O
        self.bridge = CvBridge()
        self.subscriber_ = self.create_subscription(Image, '/camera/camera/color/image_raw', self.listener_callback, 10)
        self.get_logger().info('[INFO] __init__, Create Subscription to rgb image...')
        self.subscriber_  # prevent unused variable warning
        self.publisher_ = self.create_publisher(String, 'terrain_class', 10)
        # Add terrain distance 
        # self.publisher_ = self.create_publisher(Image, 'terrain_dist', 10)

        # Overlay the DPU and Vitis-AI .xmodel file
        self.overlay = DpuOverlay("dpu.bit")
        self.model_path = os.path.join(get_package_share_directory('terra_sense'), 'config', ml_model)
        self.get_logger().info("MODEL="+self.model_path)
        self.overlay.load_model(self.model_path)    

        # Create DPU runner
        self.dpu = self.overlay.runner

        # IO tensor info
        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()

        self.shapeIn = tuple(inputTensors[0].dims)
        self.shapeOut = tuple(outputTensors[0].dims)
        self.batch    = self.shapeIn[0]
        self.inH, self.inW = self.shapeIn[1], self.shapeIn[2]
        self.outputSize = int(outputTensors[0].get_data_size() / self.shapeIn[0])

        self.output_data = [np.empty(self.shapeOut, dtype=np.float32, order="C")]
        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]

        # --- metrics state ---
        # self.start_time = time.perf_counter()
        # self.first_msg_time = None           # perf_counter at first frame
        # self.last_msg_time  = None           # perf_counter at last frame
        # self.total_frames   = 0
        # self.model_ms = []                   # or: deque(maxlen=10000)
        # self.end2end_ms = []
        
        self.get_logger().info('[INFO] __init__ exiting...')
        self.get_logger().info(f"[INFO] Input shape: {self.shapeIn}, Output shape: {self.shapeOut}")
        self.get_logger().info('========== Starting classification ==========')

    def calculate_softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def normalize(self, image):
        image=image/255.0
        image=image-0.5
        image=image*2
        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # image = (image - mean) / std
        return image

    def listener_callback(self, msg):
        #self.get_logger().info("Starting of listener callback...")
        # t_cb_start = time.perf_counter()
        # if self.first_msg_time is None:
        #     self.first_msg_time = t_cb_start
        
        # ROS to RBG numboy
        cv2_image_org = self.bridge.imgmsg_to_cv2(msg,desired_encoding="rgb8")
        resized_image = cv2.resize(cv2_image_org, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Preprocess to match how model was trained
        preprocessed = self.normalize(resized_image)

        # Batch it into the input buffer
        self.input_data[0][0, ...] = preprocessed

        # Inference on DPU
         # --- model latency (DPU only) ---
        # t_inf_start = time.perf_counter()
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        # t_inf_end = time.perf_counter()
        # model_ms = (t_inf_end - t_inf_start) * 1e3
        # self.model_ms.append(model_ms)

        # Get top prediction
        temp = [j.reshape(1, self.outputSize) for j in self.output_data]
        probs = self.calculate_softmax(temp)
        predicted_index = np.argmax(probs)

        # Publish Data 
        # self.get_logger().info("prediction="+str(prediction))
        msg = String()
        # msg.data =' '
        msg.data = class_names[predicted_index]
        self.publisher_.publish(msg)

        # --- end-to-end latency (message arrival -> publish) ---
        # t_cb_end = time.perf_counter()
        # end2end_ms = (t_cb_end - t_cb_start) * 1e3
        # self.end2end_ms.append(end2end_ms)

        # self.total_frames += 1
        # self.last_msg_time = t_cb_end


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

    # def destroy_node(self):
    #     try:
    #         self._report_metrics()   # print FPS/latency once at shutdown
    #     finally:
    #         return super().destroy_node()
        
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
