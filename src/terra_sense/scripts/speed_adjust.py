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

        self.current_terrain = '0'
        self.acceleration = 0
        self.stop = False

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
        if self.current_terrain == msg.data:
            self.acceleration = 0
        else:
            if msg.data == '2' or msg.data == '3' or msg.data == '5':
                self.acceleration = -0.5
            elif msg.data == '4' or msg.data == '1':
                self.acceleration = 0.0
            elif msg.data == '6':
                self.acceleration = 0.2
                self.stop = True

    def cmd_vel_callback(self, msg):
        adjusted_cmd_vel = Twist()
        if self.stop:
            adjusted_cmd_vel.linear.x = 0
            adjusted_cmd_vel.linear.y = 0
            adjusted_cmd_vel.linear.z = msg.linear.z 
            adjusted_cmd_vel.angular.x = msg.angular.x
            adjusted_cmd_vel.angular.y = msg.angular.y
            adjusted_cmd_vel.angular.z = msg.angular.z * self.acceleration
            self.adjusted_cmd_vel_pub.publish(adjusted_cmd_vel)
        else:
            adjusted_cmd_vel.linear.x = msg.linear.x * self.acceleration
            adjusted_cmd_vel.linear.y = msg.linear.y * self.acceleration
            adjusted_cmd_vel.linear.z = msg.linear.z * self.acceleration
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
