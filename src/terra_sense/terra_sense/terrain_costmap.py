import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid

class TerrainCostmapLayer(Node):
    def __init__(self):
        super().__init__('terrain_costmap_layer')
        self.subscription = self.create_subscription(String, 'terrain_class', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(OccupancyGrid, 'costmap', 10)
        self.costmap = OccupancyGrid()
        self.initialize_costmap()

    def initialize_costmap(self):
        self.costmap.header.frame_id = 'map'
        self.costmap.info.resolution = 0.05
        self.costmap.info.width = 100
        self.costmap.info.height = 100
        self.costmap.info.origin.position.x = 0.0
        self.costmap.info.origin.position.y = 0.0
        self.costmap.data = [0] * (self.costmap.info.width * self.costmap.info.height)

    def listener_callback(self, msg):
        terrain_type = msg.data
        if terrain_type == "smooth":
            self.update_costmap(0)
        elif terrain_type == "rough":
            self.update_costmap(100)
        elif terrain_type == "stop":
            self.update_costmap(-1)

    def update_costmap(self, cost):
        self.costmap.data = [cost] * (self.costmap.info.width * self.costmap.info.height)
        self.costmap.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.costmap)

def main(args=None):
    rclpy.init(args=args)
    node = TerrainCostmapLayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
