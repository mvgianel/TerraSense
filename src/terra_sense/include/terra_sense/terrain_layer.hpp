#ifndef TERRAIN_LAYER_H_
#define TERRAIN_LAYER_H_

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include <nav2_costmap_2d/costmap_layer.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/msg/string.hpp>
#include <nav2_util/node_utils.hpp>
#include "nav2_costmap_2d/layered_costmap.hpp"

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

  virtual void onInitialize();
  virtual void updateBounds(
    double robot_x, double robot_y, double robot_yaw, double * min_x,
    double * min_y,
    double * max_x,
    double * max_y); 
  virtual void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j);

  virtual void onFootprintChanged();

  virtual bool isClearable() {return false;}

private:
  std::string old_terrain_ = "0";
  double last_min_x_, last_min_y_, last_max_x_, last_max_y_;
  bool need_recalculation_;
  void terrainCallback(const std_msgs::msg::String::SharedPtr msg);
  std::string terrain_topic_; 
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr terrain_subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cost_publisher_;
  std::string terrain_;
  unsigned char terrain_cost_;
  unsigned char smooth_cost_;
  unsigned char rough_cost_;
  unsigned char obstacle_cost_;
};

}  // namespace terra_sense
#endif