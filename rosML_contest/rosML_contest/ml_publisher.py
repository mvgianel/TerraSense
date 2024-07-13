'''
Assuming Jupyter Notebook define ML inputs and outputs as:
from pynq import Overlay

overlay = Overlay("path_to_your_bitstream.bit")

input_buffer = overlay.allocate(shape=(1, 224, 224, 3), dtype=np.uint8)
output_buffer = overlay.allocate(shape=(1, 10), dtype=np.float32)

# Perform inference
overlay.model.run(input_buffer, output_buffer)

Heavily inspired by https://github.com/amd/Kria-RoboticsAI/blob/main/files/ROSAI/camera_input/rosai_camera/rosai_camera/rosai_camera_demo.py
'''

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import sys
sys.path.append('/usr/lib/python3.10/site-packages')
sys.path.append('/usr/local/share/pynq-venv/lib/python3.10/site-packages')
# the above path is needed by pynq_dpu
from pynq_dpu import DpuOverlay

class MLPublisher(Node):
    def __init__(self):
        super().__init__('ml_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'ml_results', 10)
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.overlay = Overlay("path_to_your_bitstream.bit")
        self.input_buffer = self.overlay.allocate(shape=(1, 224, 224, 3), dtype=np.uint8)
        self.output_buffer = self.overlay.allocate(shape=(1, 10), dtype=np.float32)

        # Overlay the DPU and Vitis-AI .xmodel file
        self.overlay = DpuOverlay("dpu.bit")
        self.model_path = '/home/root/jupyter_notebooks/pynq-dpu/dpu_mnist_classifier.xmodel'
        self.get_logger().info("MODEL="+self.model_path)
        self.overlay.load_model(self.model_path)

         # Create DPU runner
        self.dpu = self.overlay.runner

        self.get_logger().info('[INFO] __init__ exiting...')

    def timer_callback(self):
        # Perform inference
        self.overlay.model.run(self.input_buffer, self.output_buffer)
        msg = Float32MultiArray()
        msg.data = self.output_buffer.tolist()
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MLPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
