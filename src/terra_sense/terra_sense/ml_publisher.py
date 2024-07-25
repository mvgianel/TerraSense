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
from ament_index_python.packages import get_package_share_directory

# import CV BRIDGE
from cv_bridge import CvBridge, CvBridgeError
import cv2

sys.path.append('/usr/lib/python3.10/site-packages')
sys.path.append('/usr/local/share/pynq-venv/lib/python3.10/site-packages')
# the above path is needed by pynq_dpu
from pynq_dpu import DpuOverlay

ml_model = 'zcu102_q_train2_resnet18_terraset6.h5.xmodel'

class MLPublisher(Node):
    def __init__(self):
        super().__init__('ml_publisher')
        self.subscriber_ = self.create_subscription(Image, '/camera/camera/color/image_raw', self.listener_callback, 10)
        self.get_logger().info('[INFO] __init__, Create Subscription to rgb image...')
        self.subscriber_  # prevent unused variable warning
        self.publisher_ = self.create_publisher(Image, 'terrain_class', 10)
        # Add terrain distance 
        self.publisher_ = self.create_publisher(Image, 'terrain_dist', 10)

        # Overlay the DPU and Vitis-AI .xmodel file
        self.overlay = DpuOverlay("dpu.bit")
        self.model_path = os.path.join(get_package_share_directory('terra_sense'), 'config', ml_model)
        self.get_logger().info("MODEL="+self.model_path)
        self.overlay.load_model(self.model_path)

        # Create DPU runner
        self.dpu = self.overlay.runner
        self.get_logger().info('[INFO] __init__ exiting...')

    def calculate_softmax(self, data):
        result = np.exp(data)
        return result

    def listener_callback(self, msg):
        self.get_logger().info("Starting of listener callback...")
        bridge = CvBridge()
        cv2_image_org = bridge.imgmsg_to_cv2(msg,desired_encoding="rgb8")
        y1 = (128)
        y2 = (128+280)
        x1 = (208)
        x2 = (208+280)
        roi_img = cv2_image_org[ y1:y2, x1:x2, : ]
        resized_image = cv2.resize(roi_img, (224, 224), interpolation=cv2.INTER_LINEAR)
        roi_img_gray=cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        cv2_image_normal = np.asarray(roi_img_gray/255, dtype=np.float32)
        cv2_image = np.expand_dims(cv2_image_normal, axis=2)

        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()
        shapeIn = tuple(inputTensors[0].dims)
        shapeOut = tuple(outputTensors[0].dims)
        outputSize = int(outputTensors[0].get_data_size() / shapeIn[0])

        softmax = np.empty(outputSize)
        output_data = [np.empty(shapeOut, dtype=np.float32, order="C")]
        input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
        image = input_data[0]
        image[0,...] = cv2_image

        prediction = 0
        job_id = self.dpu.execute_async(input_data, output_data)
        self.dpu.wait(job_id)
        temp = [j.reshape(1, outputSize) for j in output_data]
        softmax = self.calculate_softmax(temp[0][0])
        prediction = softmax.argmax()

        self.get_logger().info("prediction="+str(prediction))

        # DISPLAY
        cv2_bgr_image = cv2.cvtColor(cv2_image_org, cv2.COLOR_RGB2BGR)
        cv2.imshow('rosai_demo',cv2_bgr_image)
        cv2.waitKey(1)

        # CONVERT BACK TO ROS & PUBLISH
        image_ros = bridge.cv2_to_imgmsg(cv2_image)
        self.publisher_.publish(image_ros)
        self.get_logger().info("published prediction="+str(prediction))

def main(args=None):
    rclpy.init(args=args)
    node = MLPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()