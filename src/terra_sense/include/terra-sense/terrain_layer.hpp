#ifndef TERRAIN_LAYER_H_
#define TERRAIN_LAYER_H_

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include <nav2_costmap_2d/costmap_layer.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
// #include <costmap_2d/GenericPluginConfig.h>
// #include <dynamic_reconfigure/server.h>
#include <std_msgs/msg/string.hpp>
#include <nav2_util/node_utils.hpp>

# based on https://github.com/ros-navigation/navigation2_tutorials/blob/master/nav2_gradient_costmap_plugin/include/nav2_gradient_costmap_plugin/gradient_layer.hpp

namespace terra-sense
{

class TerrainLayer : public nav2_costmap_2d::Layer
{

  public:
    TerrainLayer();

    virtual void onInitialize();
    virtual void updateBounds(
      double robot_x, double robot_y, double robot_yaw, double * min_x,
      double * min_y,
      double * max_x,
      double * max_y);
    virtual void updateCosts(
      nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j);

    virtual void reset()
    {
      return;
    }

    virtual void onFootprintChanged();

    virtual bool isClearable() {return false;}

private:
  double last_min_x_, last_min_y_, last_max_x_, last_max_y_;

  // Indicates that the entire gradient should be recalculated next time.
  bool need_recalculation_;

  // Size of gradient in cells
  int GRADIENT_SIZE = 20;
  // Step of increasing cost per one cell in gradient
  int GRADIENT_FACTOR = 10;
  
  std::string topic_;
  unsigned char default_cost_;
  unsigned char terrain_cost_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

}  // namespace nav2_gradient_costmap_plugin
#endif