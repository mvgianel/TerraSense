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

// based on https://github.com/ros-navigation/navigation2_tutorials/blob/master/nav2_gradient_costmap_plugin/include/nav2_gradient_costmap_plugin/gradient_layer.hpp

namespace terra_sense
{

class TerrainLayer : public nav2_costmap_2d::Layer
{

  public:
    TerrainLayer();

  virtual void reset()
  {
    return;
  }

    virtual void onInitialize() override;
    virtual void updateBounds(
      double robot_x, double robot_y, double robot_yaw, double * min_x,
      double * min_y,
      double * max_x,
      double * max_y) override; 
    virtual void updateCosts(
      nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
    
  virtual bool isClearable() {return false;}

private:
  void terrainCallback(const std_msgs::msg::String::SharedPtr msg);

  std::string topic_;
  unsigned char default_cost_;
  unsigned char terrain_cost_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

}  // namespace nav2_gradient_costmap_plugin
#endif