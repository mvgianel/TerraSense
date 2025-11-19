#!/usr/bin/env python3
import os
from datetime import datetime
import math
import re

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Output image size
TARGET_SIZE = 1024  # square canvas

class CostmapSaver(Node):
    def __init__(self):
        super().__init__('costmap_saver')
        self.get_logger().info('Starting CostmapSaver; subscribing to topics')
        # store latest messages
        self.prev_uint8 = None
        self.latest_terrain = ''
        self.latest_path = None
        self.latest_waypoints = None
        self.latest_odom = None
        # list of diffs: dicts with i,j,old,new
        self.latest_diffs = []

        # Subscribers
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.cb_costmap, 10)
        self.create_subscription(String, '/terrain_class', self.cb_terrain, 10)
        self.create_subscription(Path, '/local_plan', self.cb_path, 10)
        self.create_subscription(MarkerArray, '/waypoints', self.cb_waypoints, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.cb_odom, 10)
        # subscribe to diff topic; replace '/cost_changes' with your topic
        self.create_subscription(String, '/cost_changes', self.cb_diff, 10)

        # RViz costmap palette
        self.colors = np.zeros((256, 4), dtype=np.uint8)
        self.colors[0] = [0, 0, 0, 0]
        for k in range(1, 99):
            v = (255 * k) // 100
            self.colors[k] = [v, 0, 255 - v, 255]
        self.colors[99] = [0, 255, 255, 255]
        self.colors[100] = [255, 0, 255, 255]
        for k in range(101, 128):
            self.colors[k] = [0, 255, 0, 255]
        for k in range(128, 255):
            g = (255 * (k - 128)) // (254 - 128)
            self.colors[k] = [255, g, 0, 255]
        self.colors[255] = [0x70, 0x89, 0x86, 255]

    def cb_terrain(self, msg: String):
        self.latest_terrain = msg.data

    def cb_path(self, msg: Path):
        self.latest_path = msg

    def cb_waypoints(self, msg: MarkerArray):
        self.latest_waypoints = msg

    def cb_odom(self, msg: Odometry):
        self.latest_odom = msg

    def cb_diff(self, msg: String):
        # parse messages like "Cell (i,j)-terrain: T, prev cost: o, new cost: n"
        txt = msg.data
        m = re.search(r"Cell \((\d+),(\d+)\)-terrain: [^,]+, prev cost: (\d+), new cost: (\d+)", txt)
        if m:
            i, j, oldc, newc = map(int, m.groups())
            self.latest_diffs.append({'i': i, 'j': j, 'old': oldc, 'new': newc})

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def cb_costmap(self, msg: OccupancyGrid):
        # extract dimensions and origin
        h, w = msg.info.height, msg.info.width
        origin = msg.info.origin
        grid = np.array(msg.data, dtype=np.int8).reshape((h, w))
        grid_u = grid.astype(np.uint8)

        # skip if unchanged
        if self.prev_uint8 is not None and np.array_equal(grid_u, self.prev_uint8):
            return
        self.prev_uint8 = grid_u.copy()

        # render base costmap
        rgba = self.colors[grid_u]
        base = Image.fromarray(rgba, 'RGBA')
        img_rgba = base.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
        img = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (255, 255, 255))
        img.paste(img_rgba, mask=img_rgba.split()[3])
        draw = ImageDraw.Draw(img)

        # overlay diffs: half-cell previous/new
        cw = TARGET_SIZE / w
        ch = TARGET_SIZE / h
        for diff in self.latest_diffs:
            i, j = diff['i'], diff['j']
            oldc, newc = diff['old'], diff['new']
            # center of cell
            cx = (j + 0.5) * cw
            cy = (h - (i + 0.5)) * ch
            # half cell corners
            corners = [
                (cx - cw/2, cy - ch/2),  # top-left
                (cx + cw/2, cy - ch/2),  # top-right
                (cx + cw/2, cy + ch/2),  # bottom-right
                (cx - cw/2, cy + ch/2)   # bottom-left
            ]
            # prev cost triangle: top-left, bottom-left, center
            draw.polygon([corners[0], corners[3], (cx, cy)], fill=tuple(self.colors[oldc][:3]), outline=(0,0,0))
            # new cost triangle: top-right, bottom-right, center
            draw.polygon([corners[1], corners[2], (cx, cy)], fill=tuple(self.colors[newc][:3]), outline=(0,0,0))
        # clear diffs after overlay
        self.latest_diffs.clear()

        # font for terrain text
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
        except:
            font = ImageFont.load_default()
        # overlay terrain
        if self.latest_terrain:
            text = f"Terrain: {self.latest_terrain}"
            margin = 10
            max_w = TARGET_SIZE - 2*margin
            lines, line = [], ''
            for word in text.split():
                t = (f"{line} {word}").strip()
                if draw.textsize(t, font=font)[0] <= max_w:
                    line = t
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
            y = margin
            for ln in lines:
                draw.text((margin, y), ln, fill=(0,0,0), font=font)
                y += font.getsize(ln)[1] + 5

        # path, waypoints, odom remain as before (omitted for brevity)
        # ... (retain previous draw code for path, waypoints, odometry) ...

        # save image
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'local_costmap_{ts}.png'
        out = os.path.join(os.getcwd(), fname)
        img.save(out)
        self.get_logger().info(f'Saved new costmap image → {out}')


def main():
    rclpy.init()
    node = CostmapSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
