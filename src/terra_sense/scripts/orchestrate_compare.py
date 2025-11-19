#!/usr/bin/env python3
import os
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class OrchestrateCompare(Node):
    def __init__(self):
        super().__init__('orchestrate_compare')
        # Params
        self.goal_x = float(self.declare_parameter('goal_x', 1.0).value)
        self.goal_y = float(self.declare_parameter('goal_y', 0.0).value)
        self.goal_yaw = float(self.declare_parameter('goal_yaw', 0.0).value)
        self.global_frame = self.declare_parameter('global_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.save_dir = self.declare_parameter('save_dir', '/tmp/nav2_compare').value
        os.makedirs(self.save_dir, exist_ok=True)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Action clients (namespaced)
        self.nav_with = ActionClient(self, NavigateToPose, '/with_plugin/navigate_to_pose')
        self.nav_without = ActionClient(self, NavigateToPose, '/no_plugin/navigate_to_pose')

        # Subscriptions for costmaps & plan topics (namespaced)
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)

        # Costmaps
        self.global_with = None
        self.local_with = None
        self.global_without = None
        self.local_without = None

        self.create_subscription(OccupancyGrid, '/with_plugin/global_costmap/costmap',
                                 self._cb_global_with, qos)
        self.create_subscription(OccupancyGrid, '/with_plugin/local_costmap/costmap',
                                 self._cb_local_with, qos)
        self.create_subscription(OccupancyGrid, '/no_plugin/global_costmap/costmap',
                                 self._cb_global_without, qos)
        self.create_subscription(OccupancyGrid, '/no_plugin/local_costmap/costmap',
                                 self._cb_local_without, qos)

        # Planned path topics from planner_server (published for visualization) 
        # (topic name: "<ns>/plan")
        self.plan_with = None
        self.plan_without = None
        self.create_subscription(Path, '/with_plugin/plan', self._cb_plan_with, 10)
        self.create_subscription(Path, '/no_plugin/plan', self._cb_plan_without, 10)

        # Start orchestration once action servers are ready
        self.timer = self.create_timer(1.0, self._tick)
        self.state = 'WAIT_SERVERS'
        self.start_pose = None

    # --- Callbacks ---
    def _cb_global_with(self, msg): self.global_with = msg
    def _cb_local_with(self, msg): self.local_with = msg
    def _cb_global_without(self, msg): self.global_without = msg
    def _cb_local_without(self, msg): self.local_without = msg
    def _cb_plan_with(self, msg): self.plan_with = msg
    def _cb_plan_without(self, msg): self.plan_without = msg

    # --- Helpers ---
    def _now_tag(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def _pose_from_tf(self):
        trans = self.tf_buffer.lookup_transform(
            self.global_frame, self.base_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0))
        p = PoseStamped()
        p.header.frame_id = self.global_frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = trans.transform.translation.x
        p.pose.position.y = trans.transform.translation.y
        p.pose.orientation = trans.transform.rotation
        return p


    def yaw_to_quaternion(self, yaw):
        return (0.0, 0.0, math.sin(yaw/2.0), math.cos(yaw/2.0))
        
    def _goal_pose(self, x, y, yaw):
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        p = PoseStamped()
        p.header.frame_id = self.global_frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return p

    def _save_costmap_png(self, grid: OccupancyGrid, outpath: str):
        if grid is None: 
            self.get_logger().warn(f"Missing costmap for {outpath}")
            return
        w, h = grid.info.width, grid.info.height
        data = np.array(grid.data, dtype=np.int16).reshape(h, w)
        # map to [0,1]; unknown (-1) -> 0.5 gray, free(0)->1.0, occ(100)->0.0
        img = np.ones_like(data, dtype=np.float32)
        img[data == -1] = 0.5
        known = (data >= 0)
        img[known] = 1.0 - (data[known] / 100.0)
        plt.figure()
        plt.title(os.path.basename(outpath))
        plt.imshow(img, origin='lower')  # no explicit colormap to keep defaults
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    def _save_path_png(self, path: Path, outpath: str, bg: OccupancyGrid = None):
        plt.figure()
        plt.title(os.path.basename(outpath))
        if bg is not None:
            w, h = bg.info.width, bg.info.height
            data = np.array(bg.data, dtype=np.int16).reshape(h, w)
            img = np.ones_like(data, dtype=np.float32)
            img[data == -1] = 0.5
            known = (data >= 0)
            img[known] = 1.0 - (data[known] / 100.0)
            plt.imshow(img, origin='lower')

        if path and path.poses:
            xs = [p.pose.position.x for p in path.poses]
            ys = [p.pose.position.y for p in path.poses]
            plt.plot(xs, ys, linewidth=2.0)
        else:
            self.get_logger().warn(f"No path to plot at {outpath}")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    def _send_nav(self, client: ActionClient, goal_pose: PoseStamped):
        if not client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError("navigate_to_pose action server not available")
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        return client.send_goal_async(goal_msg)

    def _tick(self):
        try:
            if self.state == 'WAIT_SERVERS':
                if self.nav_with.wait_for_server(0.1) and self.nav_without.wait_for_server(0.1):
                    self.state = 'CAPTURE_START'
                    self.get_logger().info("Action servers ready.")
            elif self.state == 'CAPTURE_START':
                self.start_pose = self._pose_from_tf()
                tag = self._now_tag()
                # snapshot both stacks’ current costmaps
                self._save_costmap_png(self.global_with, f"{self.save_dir}/{tag}_with_global.png")
                self._save_costmap_png(self.local_with,  f"{self.save_dir}/{tag}_with_local.png")
                self._save_costmap_png(self.global_without, f"{self.save_dir}/{tag}_no_global.png")
                self._save_costmap_png(self.local_without,  f"{self.save_dir}/{tag}_no_local.png")
                self.state = 'NAV_WITH'
                self.get_logger().info("Captured initial costmaps. Navigating with plugin enabled.")
            elif self.state == 'NAV_WITH':
                goal = self._goal_pose(self.goal_x, self.goal_y, self.goal_yaw)
                self._future_with = self._send_nav(self.nav_with, goal)
                self.state = 'WAIT_WITH_DONE'
            elif self.state == 'WAIT_WITH_DONE':
                if self._future_with.done():
                    result = self._future_with.result()
                    self.get_logger().info(f"With-plugin nav done: {result.status}")
                    # Save the plan observed
                    tag = self._now_tag()
                    self._save_path_png(self.plan_with, f"{self.save_dir}/{tag}_with_plan.png", self.global_with)
                    # Return to start
                    self._future_with_back = self._send_nav(self.nav_with, self.start_pose)
                    self.state = 'WAIT_WITH_BACK'
            elif self.state == 'WAIT_WITH_BACK':
                if self._future_with_back.done():
                    self.get_logger().info("Returned to start (with plugin).")
                    self.state = 'NAV_NO'
            elif self.state == 'NAV_NO':
                goal = self._goal_pose(self.goal_x, self.goal_y, self.goal_yaw)
                self._future_no = self._send_nav(self.nav_without, goal)
                self.state = 'WAIT_NO_DONE'
            elif self.state == 'WAIT_NO_DONE':
                if self._future_no.done():
                    result = self._future_no.result()
                    self.get_logger().info(f"No-plugin nav done: {result.status}")
                    tag = self._now_tag()
                    self._save_path_png(self.plan_without, f"{self.save_dir}/{tag}_no_plan.png", self.global_without)
                    # Return to start
                    self._future_no_back = self._send_nav(self.nav_without, self.start_pose)
                    self.state = 'WAIT_NO_BACK'
            elif self.state == 'WAIT_NO_BACK':
                if self._future_no_back.done():
                    self.get_logger().info("Returned to start (no plugin). All done.")
                    self.state = 'DONE'
            elif self.state == 'DONE':
                self.get_logger().info(f"Artifacts saved under: {self.save_dir}")
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            rclpy.shutdown()

def main():
    rclpy.init()
    rclpy.spin(OrchestrateCompare())

if __name__ == '__main__':
    main()
