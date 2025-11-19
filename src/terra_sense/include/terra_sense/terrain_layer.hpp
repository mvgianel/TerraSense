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


using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;
using nav2_costmap_2d::FREE_SPACE;

namespace terra_sense
{

class TerrainLayer : public nav2_costmap_2d::Layer
{
  public:
  
    TerrainLayer() = default;
    ~TerrainLayer() override = default;
    void onInitialize() override;
    void matchSize() override;
    void updateBounds(
    double robot_x, double robot_y, double robot_yaw, double * min_x,
    double * min_y,
    double * max_x,
    double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j) override;
  bool isClearable() override;
  void reset() override;
  void terrainCallback(const std_msgs::msg::String::SharedPtr msg);
  std::string stableterrain();

private:
  bool enabled_{true};
  double confidence_threshold_{0.6};   // param: terrain_layer.confidence_threshold
  double max_range_x{3.0};
  double max_range_y{3.0};
  double theta_max_{35.0};
  double r0_{1.0};
  double pow_p_{4.0};
  int history_len_{10}; 
  std::string terrain_class_;
  
  double robot_x_, robot_y_, robot_yaw_;
  
  int width_{0};
  int height_{0};
  std::vector<float> terrain_costs_;             // terrain-only map
  std::vector<std::string> history_;            // recent labels
  std::unordered_map<std::string, unsigned char> cost_dict;
  std::string last_label_;                      // last accepted label
  
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr terrain_subscription_;
};

}  // namespace terra_sense
#endif
