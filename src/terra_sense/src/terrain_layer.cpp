#include <terra_sense/terrain_layer.hpp>
#include "nav2_costmap_2d/costmap_math.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "nav2_util/validate_messages.hpp"
#include "rclcpp/parameter_events_filter.hpp"

namespace terra_sense
{

void TerrainLayer::onInitialize()
{
  CostmapLayer::onInitialize();
  auto node = node_.lock();

  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }

  matchSize();
  
   cost_dict = {
    {"pavement",      10},
    {"cobblestonebrick", 30},
    {"dirtground",     50},
    {"grass",          80},
    {"sand",           90},
    {"stairs",         100}
  };
  
  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".terrain_class", terrain_class_);
  node->get_parameter(name_ + ".theta_max", theta_max_);
  theta_max_ = theta_max_ * M_PI / 180.0;
  node->get_parameter(name_ + ".r0", r0_);
  node->get_parameter(name_ + ".pow_p", pow_p_);
  node->get_parameter(name_ + ".history_len", history_len_); 
  node->get_parameter(name_ + ".confidence_threshold", confidence_threshold_);
  node->get_parameter(name_ + ".max_range_x", max_range_x);
  node->get_parameter(name_ + ".max_range_y", max_range_y);
  

  auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
               .best_effort()
               .durability_volatile();
  terrain_subscription_ = node->create_subscription<std_msgs::msg::String>(
    "/terrain_class", qos, std::bind(&TerrainLayer::terrainCallback, this, std::placeholders::_1));
  
   // Initialize persistent mask to match master size
  auto * main_map = layered_costmap_->getCostmap();
  width_  = main_map->getSizeInCellsX();
  height_ = main_map->getSizeInCellsY();

    // Terrain-only cost memory. This persists between updates and
  // only changes when a new terrain label is accepted with confidence.
  // terrain_costs_.assign(width_ * height_, 0.0f);
  
  last_label_.clear();
  current_ = true;
}

bool TerrainLayer::isClearable() {
  // Prevent clearing so terrain overrides persist
  return false;
}

void TerrainLayer::updateBounds(double robot_x, double robot_y, double robot_yaw,
  double *min_x, double *min_y, double *max_x, double *max_y)
{

  if (!enabled_) {
    return;
  }
  
  
  // === STORE THE ROBOT POSE ===
  robot_x_   = robot_x;
  robot_y_   = robot_y;
  robot_yaw_ = robot_yaw;

  // Expand update area to include entire mask
  *min_x = std::min(*min_x, robot_x - max_range_x);
  *min_y = std::min(*min_y, robot_y - max_range_y);
  *max_x = std::max(*max_x, robot_x + max_range_x);
  *max_y = std::max(*max_y, robot_y + max_range_y);

  touch(robot_x, robot_y, min_x, min_y, max_x, max_y);

  current_ = true;
}

void TerrainLayer::updateCosts(nav2_costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j)
{
  // RCLCPP_INFO(rclcpp::get_logger("TerrainLayer"), "updateCosts() called");
  if (!enabled_) {
    return;
  }

  // Compute a stable terrain label with confidence gating
  const std::string lbl = stableterrain();
  const auto it = cost_dict.find(lbl);
  const unsigned char base =
    (it != cost_dict.end()) ? it->second : 0;

  const bool have_stable_label = !lbl.empty() && base > 0;
  if (!have_stable_label) {
    // Nothing to add for this cycle
    updateWithAddition(master_grid, min_i, min_j, max_i, max_j);
    return;
  }

  // If we have a confident label, overwrite terrain_costs_
  //    in the observed forward fan with the *new* terrain.
  //
  //    This is where we allow the terrain at a given place to change:
  //    - previously "sand" -> now "pavement" if lbl switches and passes
  //      confidence_threshold_.

    for (int j = min_j; j < max_j; ++j) {
      for (int i = min_i; i < max_i; ++i) {
        double wx, wy;
        master_grid.mapToWorld(i, j, wx, wy);

        const double dx = wx - robot_x_;
        const double dy = wy - robot_y_;
        const double r  = std::hypot(dx, dy);
        if (r <= 0.0 || r > max_range_x) {
          continue;
        }

        double th = std::atan2(dy, dx) - robot_yaw_;
        th = std::atan2(std::sin(th), std::cos(th)); // normalize [-pi, pi]
        if (std::abs(th) > theta_max_) {
          continue;
        }

        // Forward cone weighting
        const double w_r = std::exp(-r / r0_);             // distance falloff
        const double w_t = std::pow(std::cos(th), pow_p_); // angular falloff

        double cell_cost = 0.0;
        if (lbl == "stairs") {
          // Solid lethal wherever the cone covers
          cell_cost = nav2_costmap_2d::LETHAL_OBSTACLE;
        } else {
          cell_cost = static_cast<double>(base) * (w_r * w_t);
        }

        cell_cost = std::max(cell_cost, 1.0);
        this->setCost(i, j, cell_cost);

        // Read what other layers have already put into the master grid
        unsigned char master_cost = master_grid.getCost(i, j);
        // Treat unknown as 0 
        if (master_cost == nav2_costmap_2d::NO_INFORMATION) {
          master_cost = 0;  
        }


        // Add and saturate below lethal
        int sum = static_cast<int>(master_cost) + static_cast<int>(cell_cost);
        // if (sum >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
        //   sum = nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE - 1;
        // }
        // IMPORTANT:
        // We *overwrite* the stored terrain cost here.
        // That allows cheaper terrain to replace more expensive
        // terrain at the same place, once the label changes with confidence.
        // this->setCost(i,j, static_cast<unsigned char>(std::round(cell_cost)));
        master_grid.setCost(i, j, std::min(sum, 252));
      }
    }

    // RCLCPP_INFO(rclcpp::get_logger("TerrainLayer"), "terrain=%s base=%d ",
    //         lbl.c_str(), base);

  //  Add terrain cost onto whatever is already in master_grid
  // this->updateWithAddition(master_grid, min_i, min_j, max_i, max_j);
}

/**
 * Returns a "stable" terrain label using majority vote + confidence gating.
 *
 * - We track the last accepted label in last_label_.
 * - Compute majority label over history_.
 * - Only update last_label_ if the majority fraction
 *   exceeds confidence_threshold_.
 *
 * This enables
 *   - local costmap: low threshold -> reacts quickly to new terrain
 *   - global costmap: high threshold -> only changes with strong evidence
 */
std::string TerrainLayer::stableterrain()
{
  if (history_.empty()) {
    // No new messages; keep the last label (or empty at startup)
    return last_label_;
  }

  std::unordered_map<std::string, int> counts;
  for (const auto & s : history_) {
    counts[s]++;
  }

  auto best_it = std::max_element(
    counts.begin(), counts.end(),
    [](const auto & a, const auto & b) {
      return a.second < b.second;
    });

  const int best_count = best_it->second;
  const int total      = static_cast<int>(history_.size());
  const double frac =
    static_cast<double>(best_count) / std::max(1, total);


  if (frac >= confidence_threshold_) {
    // Accept this as the new terrain label
    last_label_ = best_it->first;
  }

//  RCLCPP_INFO(rclcpp::get_logger("TerrainLayer"), "label=%s confidence=%f ",
//             last_label_.c_str(), frac);

  // If frac < threshold, keep last_label_ (terrain does not change this cycle).
  return last_label_;
}

void TerrainLayer::matchSize()
{
  auto * main_map = layered_costmap_->getCostmap();
  width_  = main_map->getSizeInCellsX();
  height_ = main_map->getSizeInCellsY();
  double res         = main_map->getResolution();
  double origin_x    = main_map->getOriginX();
  double origin_y    = main_map->getOriginY();

  // Resize our internal terrain-only costmap
  resizeMap(width_, height_, res, origin_x, origin_y);

  // terrain_costs_.assign(width_ * height_, 0.0f);  // persistent terrain memory
  resetMaps();            
}

void TerrainLayer::terrainCallback(const std_msgs::msg::String::SharedPtr msg)
{
// RCLCPP_INFO(rclcpp::get_logger("TerrainLayer"), "Received terrain: '%s'", msg->data.c_str());

  //terrain_ = msg->data;
  
  //auto cost = cost_dict.find(terrain_);
  //terrain_cost_ = (cost != cost_dict.end() ? cost->second : NO_INFORMATION);
  
  history_.push_back(msg->data);
  if (history_.size() > static_cast<long unsigned int>(history_len_)) {
    history_.erase(history_.begin());
  }

  // RCLCPP_INFO(rclcpp::get_logger("TerrainLayer"), "Terrain cost: '%d'", terrain_cost_);
  }

void TerrainLayer::reset() {
  resetMaps();
  history_.clear();
  last_label_.clear();
}

}  // namespace terra_sense

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(terra_sense::TerrainLayer, nav2_costmap_2d::Layer)
