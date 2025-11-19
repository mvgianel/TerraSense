#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ImageTextOverlay(Node):
    def __init__(self):
        super().__init__('image_text_overlay')

        # bridge to convert ROS Image <-> OpenCV
        self.bridge = CvBridge()
        self.latest_text = ''

        # Subscribe to the camera image and string topics
        self.get_logger().info('[INFO] __init__, Create Subscription to rgb image...')
        self.sub1_ = self.create_subscription(String, '/terrain_class', self.text_cb, 10)
        self.sub2_ = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_cb, 10)
        self.pub = self.create_publisher(Image, '/camera/image_with_text', 10)

    def text_cb(self, msg):
        self.latest_text = msg.data

    def image_cb(self, img_msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        text = self.latest_text or "<no data>"
        org = (10, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        line_type = cv2.LINE_AA

        # 1) draw thicker black text for outline
        cv2.putText(cv_img,
                    text,
                    org,
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 6,
                    line_type)

        # 2) draw normal white text on top
        cv2.putText(cv_img,
                    text,
                    org,
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    line_type)

        out = self.bridge.cv2_to_imgmsg(cv_img, 'bgr8')
        out.header = img_msg.header
        self.pub.publish(out)



def main(args=None):
    rclpy.init(args=args)
    node = ImageTextOverlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()