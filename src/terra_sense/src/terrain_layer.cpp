#include <terra_sense/terrain_layer.hpp>
#include "nav2_costmap_2d/costmap_math.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "nav2_util/validate_messages.hpp"
#include "rclcpp/parameter_events_filter.hpp"

using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;
using nav2_costmap_2d::FREE_SPACE;

namespace terra_sense
{

TerrainLayer::TerrainLayer()
: Layer(),
  terrain_(),
  terrain_cost_(NO_INFORMATION),
  need_recalculation_(true),
  last_min_x_(-std::numeric_limits<float>::max()),
  last_min_y_(-std::numeric_limits<float>::max()),
  last_max_x_( std::numeric_limits<float>::max()),
  last_max_y_( std::numeric_limits<float>::max())
{}

void TerrainLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }

  node->declare_parameter(name_ + "." + "enabled", rclcpp::ParameterValue(true));
  node->get_parameter(name_ + "." + "enabled", enabled_);

  terrain_subscription_ = node->create_subscription<std_msgs::msg::String>(
    "/terrain_class", 10, std::bind(&TerrainLayer::terrainCallback, this, std::placeholders::_1));
  cost_publisher_ = node->create_publisher<std_msgs::msg::String>("/cost_changes",40);

  RCLCPP_INFO(node->get_logger(), "TerrainLayer initialized");
  
  // current_ = true;
  // need_recalculation_ = false;
}

void TerrainLayer::onFootprintChanged()
{
  need_recalculation_ = true;

  RCLCPP_DEBUG(rclcpp::get_logger("nav2_costmap_2d"), "TerrainLayer::onFootprintChanged(): num footprint points: %lu", layered_costmap_->getFootprint().size());
}

void TerrainLayer::updateBounds(double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
  double *min_x, double *min_y, double *max_x, double *max_y)
{
    if (need_recalculation_) {
    // get the real costmap extents
    auto cm = layered_costmap_->getCostmap();
    double map_ox = cm->getOriginX();
    double map_oy = cm->getOriginY();
    double map_ex = map_ox + cm->getSizeInMetersX();
    double map_ey = map_oy + cm->getSizeInMetersY();

    // merge (minimize/​maximize) rather than overwrite
    *min_x = std::min(*min_x, map_ox);
    *min_y = std::min(*min_y, map_oy);
    *max_x = std::max(*max_x, map_ex);
    *max_y = std::max(*max_y, map_ey);

    need_recalculation_ = false;

  }

}

void TerrainLayer::updateCosts(nav2_costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }
  
  auto* master_array = master_grid.getCharMap();
  auto size_x = master_grid.getSizeInCellsX();
  auto size_y = master_grid.getSizeInCellsY();

  min_i = std::max(0, min_i);
  min_j = std::max(0, min_j);
  max_i = std::min(max_i, static_cast<int>(master_grid.getSizeInCellsX()));
  max_j = std::min(max_j, static_cast<int>(master_grid.getSizeInCellsY()));

  for (int i = min_i; i < max_i; ++i) {
    for (int j = min_j; j < max_j; ++j) {
      unsigned char old_cost = master_grid.getCost(i, j);
      unsigned char new_cost = old_cost + terrain_cost_;

      if (terrain_cost_ == LETHAL_OBSTACLE) {
        // override to lethal
        new_cost = LETHAL_OBSTACLE;
      } else if (terrain_cost_ != NO_INFORMATION) {
        // additive, but clamp at lethal
        int sum = static_cast<int>(old_cost) + static_cast<int>(terrain_cost_);
        new_cost = static_cast<unsigned char>(std::min(sum, static_cast<int>(LETHAL_OBSTACLE)));
      }

      master_grid.setCost(i, j, new_cost);

      std_msgs::msg::String msg;
      std::stringstream ss;
      ss << "terrain: " << terrain_ 
         << ", prev cost: " << static_cast<int>(old_cost) 
         << ", new cost: " << static_cast<int>(new_cost);
      msg.data = ss.str();
      cost_publisher_->publish(msg);
      // auto index = master_grid.getIndex(i, j);
      // if (terrain_cost_ != NO_INFORMATION) {
      //   master_array[index] = getCost(mx, my) + terrain_cost_;
      // }
    }
  }
}

void TerrainLayer::terrainCallback(const std_msgs::msg::String::SharedPtr msg)
{
  // ignore if same terrain
  if(msg->data == terrain_)
    return;

  terrain_ = msg->data;

  if (terrain_ == "1" || terrain_ == "4") {
    terrain_cost_ = FREE_SPACE;
  } else if (terrain_ == "2" || terrain_ == "3" || terrain_ == "5") {
    terrain_cost_ = 5;
  } else if (terrain_ == "6") {
    terrain_cost_ = LETHAL_OBSTACLE;
  } else {
    terrain_cost_ = NO_INFORMATION;
  }

  // RCLCPP_INFO(rclcpp::get_logger("TerrainLayer"), "Terrain cost: '%d'", terrain_cost_);
  // force a full repaint
  need_recalculation_ = true;
}

}  // namespace terra_sense

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(terra_sense::TerrainLayer, nav2_costmap_2d::Layer)
