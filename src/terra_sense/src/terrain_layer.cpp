// based on https://wiki.ros.org/costmap_2d/Tutorials/Creating%20a%20New%20Layer
#include <terra_sense/terrain_layer.hpp>
#include "nav2_costmap_2d/costmap_math.hpp"
#include "nav2_costmap_2d/footprint.hpp"

using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;

namespace terra_sense
{

TerrainLayer::TerrainLayer() : terrain_cost_(nav2_costmap_2d::FREE_SPACE)
{}

void TerrainLayer::onInitialize()
{
  declareParameter("topic", rclcpp::ParameterValue("/terrain_class"));
  declareParameter("default_cost", rclcpp::ParameterValue(0));

  auto node = node_.lock();
  if (!node) {
      RCLCPP_ERROR(rclcpp::get_logger("TerrainLayer"), "Failed to lock node");
      return;
  }

  node->get_parameter("topic", topic_);
  node->get_parameter("default_cost", default_cost_);

subscription_ = node->create_subscription<std_msgs::msg::String>(
      topic_, 10,
      std::bind(&TerrainLayer::terrainCallback, this, std::placeholders::_1));

  current_ = true;
}

void TerrainLayer::terrainCallback(const std_msgs::msg::String::SharedPtr msg)
{
  // 1-cobblestone/brick
  // 2-dirtground
  // 3-grass
  // 4-pavement
  // 5-sand
  // 6-stairs
  
  if (msg->data == "1" || msg->data == "2") {
            terrain_cost_ = 100;  // Slow
        } else if (msg->data == "3" || msg->data == "5") {
            terrain_cost_ = 150;  // Slower
        } else if (msg->data == "4") {
            terrain_cost_ = nav2_costmap_2d::FREE_SPACE;  // Normal
        } else if (msg->data == "6") {
            terrain_cost_ = nav2_costmap_2d::LETHAL_OBSTACLE;  // Stop
        } else {
            terrain_cost_ = default_cost_;  // Default behavior
        }
}

void TerrainLayer::updateBounds(double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/, double * min_x,
  double * min_y, double * max_x, double * max_y)
{
  *min_x = std::min(*min_x, 0.0);
  *min_y = std::min(*min_y, 0.0);
  *max_x = std::max(*max_x, static_cast<double>(layered_costmap_->getCostmap()->getSizeInCellsX()));
  *max_y = std::max(*max_y, static_cast<double>(layered_costmap_->getCostmap()->getSizeInCellsY()));
}

void TerrainLayer::updateCosts(nav2_costmap_2d::Costmap2D &master_grid, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }

  for (int i = min_i; i < max_i; i++) {
    for (int j = min_j; j < max_j; j++) {
        master_grid.setCost(i, j, terrain_cost_);
    }
  }
}

}

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(terra_sense::TerrainLayer, nav2_costmap_2d::Layer)
