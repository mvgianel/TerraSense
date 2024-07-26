import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from nav2_costmap_2d.plugins.layer import Layer
from nav2_costmap_2d.costmap_2d import Costmap2D, FREE_SPACE, LETHAL_OBSTACLE, NO_INFORMATION

class CostmapLayer(Layer):
    

class CostmapNode(Node):
    def __init__(self):
        super().__init__()
        self.terrain_type = "4"  # Default terrain type

    def on_initialize(self):
        self.declare_parameter('topic', '/terrain_class')
        self.declare_parameter('default_cost', 0)
        self.topic = self.get_parameter('topic').value
        self.default_cost = self.get_parameter('default_cost').value
        self.subscription = self.create_subscription(
            String,
            self.topic,
            self.terrain_callback,
            10
        )

    def terrain_callback(self, msg):
        '''terrain types:
        1- Cobblestone/brick
        2- dirtground
        3- grass
        4- pavement
        5- sand
        6- stairs'''
        self.terrain_type = msg.data
        if self.terrain_type == "smooth":
            self.default_cost = FREE_SPACE
        elif self.terrain_type == "rough":
            self.default_cost = LETHAL_OBSTACLE
        elif self.terrain_type == "stop":
            self.default_cost = NO_INFORMATION
        self.update_costmap()

     def update_costmap(self):
        # Implement the logic to update the costmap based on the terrain type
        # This example sets the entire costmap to the default cost for simplicity
        for i in range(self.master_grid_.get_size_in_cells_x()):
            for j in range(self.master_grid_.get_size_in_cells_y()):
                self.master_grid_.set_cost(i, j, self.default_cost)

    def update_bounds(self, origin_x, origin_y, origin_yaw, bounds):
        bounds[0] = min(bounds[0], 0)
        bounds[1] = min(bounds[1], 0)
        bounds[2] = max(bounds[2], self.master_grid_.get_size_in_cells_x())
        bounds[3] = max(bounds[3], self.master_grid_.get_size_in_cells_y())

    def update_costs(self, master_grid, min_i, min_j, max_i, max_j):
        self.update_costmap()
        master_grid.update_with_true_costmap(min_i, min_j, max_i, max_j)

def main(args=None):
    rclpy.init(args=args)
    node = TerrainCostmapLayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
