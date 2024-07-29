#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class SpeedAdjustmentNode(Node):
    def __init__(self):
        super().__init__('speed_adjustment_node')
        self.declare_parameter('terrain_classification_topic', '/terrain_class')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.terrain_classification_topic = self.get_parameter('terrain_classification_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        self.current_speed = 0.2

        self.terrain_sub = self.create_subscription(
            String,
            self.terrain_classification_topic,
            self.terrain_callback,
            10)

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10)

        self.adjusted_cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

    def terrain_callback(self, msg):
        if msg.data == 'cobblestone/brick' or msg.data == 'dirtground':
            self.current_speed = 0.1
        elif msg.data == 'grass' or msg.data == 'sand':
            self.current_speed = 0.05
        elif msg.data == 'pavement':
            self.current_speed = 0.2
        elif msg.data == 'stairs':
            self.current_speed = 0.0

    def cmd_vel_callback(self, msg):
        adjusted_cmd_vel = Twist()
        adjusted_cmd_vel.linear.x = msg.linear.x * self.current_speed
        adjusted_cmd_vel.linear.y = msg.linear.y * self.current_speed
        adjusted_cmd_vel.linear.z = msg.linear.z * self.current_speed
        adjusted_cmd_vel.angular.x = msg.angular.x
        adjusted_cmd_vel.angular.y = msg.angular.y
        adjusted_cmd_vel.angular.z = msg.angular.z
        self.adjusted_cmd_vel_pub.publish(adjusted_cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = SpeedAdjustmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
