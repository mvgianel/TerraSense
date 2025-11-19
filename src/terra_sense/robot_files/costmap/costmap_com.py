#!/usr/bin/env python3
import os
from datetime import datetime
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from geometry_msgs.msg import PoseWithCovarianceStamped

# Output image size
TARGET_SIZE = 1024  # square canvas
# Padding around the ROI
ROI_PADDING = 400

def star_points(cx, cy, outer_r, inner_r, num_points=5, rotation=-math.pi/2):
    """Return a list of (x, y) vertices for a star polygon centered at (cx, cy)."""
    pts = []
    for i in range(num_points * 2):
        angle = rotation + i * math.pi / num_points
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts

class CostmapSaver(Node):
    def __init__(self, with_camera=False):
        super().__init__('costmap_saver')
        self.get_logger().info('Starting CostmapSaver; subscribing to topics')
        self.prev_uint8 = None
        self.latest_terrain = ''
        self.latest_path = None
        self.latest_goal = None
        self.latest_odom = None
        self.with_camera = with_camera

        # Subscribers
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.cb_costmap, 10)
        self.create_subscription(String, '/terrain_class', self.cb_terrain, 10)
        self.create_subscription(Path, '/plan', self.cb_path, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.cb_goal_pose, 10)
        # self.create_subscription(Odometry, '/rtabmap/localization_pose', self.cb_odom, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/rtabmap/localization_pose', self.cb_odom, 10)

        # RViz costmap palette
        self.colors = np.zeros((256, 4), dtype=np.uint8)
        self.colors[0] = [0, 0, 0, 0]
        for i in range(1, 99):
            v = (255 * i) // 100
            self.colors[i] = [v, 0, 255 - v, 255]
        self.colors[99] = [0, 255, 255, 255]
        self.colors[100] = [255, 0, 255, 255]
        for i in range(101, 128):
            self.colors[i] = [0, 255, 0, 255]
        for i in range(128, 255):
            g = (255 * (i - 128)) // (254 - 128)
            self.colors[i] = [255, g, 0, 255]
        self.colors[255] = [0x70, 0x89, 0x86, 255]

        self.bridge = CvBridge()
        self.latest_camera = None
        self.create_subscription(ROSImage, '/camera/camera/color/image_raw', self.cb_camera, 10)

    def cb_camera(self, msg: ROSImage):
        try:
            self.latest_camera = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert camera image: {e}")

    def cb_terrain(self, msg: String):
        self.latest_terrain = msg.data

    def cb_path(self, msg: Path):
        self.latest_path = msg

    def cb_goal_pose(self, msg: PoseStamped):
        self.latest_goal = msg

    def cb_odom(self, msg: PoseWithCovarianceStamped):
        self.latest_odom = msg
        self.get_logger().info('Received localization_pose')

    def quaternion_to_yaw(self, q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def cb_costmap(self, msg: OccupancyGrid):
        # Extract map info and grid
        h, w = msg.info.height, msg.info.width
        origin = msg.info.origin
        grid = np.array(msg.data, dtype=np.int8).reshape((h, w))
        grid_u = grid.astype(np.uint8)

        # Skip if unchanged
        if self.prev_uint8 is not None and np.array_equal(grid_u, self.prev_uint8):
            return
        self.prev_uint8 = grid_u.copy()

        # Render full map into an image
        rgba = self.colors[grid_u]
        base = Image.fromarray(rgba, 'RGBA').resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
        img = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (255, 255, 255))
        img.paste(base, mask=base.split()[3])
        draw = ImageDraw.Draw(img)

        # Collect ROI points
        roi_pts = []

        # Draw path and record points
        if self.latest_path and msg.info.resolution > 0:
            pts = []
            for p in self.latest_path.poses:
                x = (p.pose.position.x - origin.position.x) / msg.info.resolution * TARGET_SIZE / w
                y = (h - (p.pose.position.y - origin.position.y) / msg.info.resolution) * TARGET_SIZE / h
                pts.append((x, y))
            if len(pts) > 1:
                draw.line(pts, fill=(0, 255, 0), width=14)
                draw.line(pts, fill=(0, 0, 0), width=10)
            roi_pts.extend(pts)

        # Draw robot triangle and record center
        if self.latest_odom and msg.info.resolution > 0:
            # symmetric triangle
            r = TARGET_SIZE / 20
            tri = [(r, 0), (-r, -r), (-r, r)]
            # center
            op = self.latest_odom.pose.pose.position
            yaw = self.quaternion_to_yaw(self.latest_odom.pose.pose.orientation)
            gx = (op.x - origin.position.x) / msg.info.resolution * TARGET_SIZE / w
            gy = (h - (op.y - origin.position.y) / msg.info.resolution) * TARGET_SIZE / h
            # draw
            transformed = []
            for px, py in tri:
                xr = px * math.cos(yaw) - py * math.sin(yaw)
                yr = px * math.sin(yaw) + py * math.cos(yaw)
                transformed.append((gx + xr, gy - yr))
            draw.polygon(transformed, fill=(255, 255, 0))
            draw.line(transformed + [transformed[0]], fill=(0, 0, 0), width=int(TARGET_SIZE/100))
            roi_pts.append((gx, gy))

        # Draw a star for /goal_pose (PoseStamped)
        if self.latest_goal and msg.info.resolution > 0:
            gp = self.latest_goal.pose.position
            sx = (gp.x - origin.position.x) / msg.info.resolution * TARGET_SIZE / w
            sy = (h - (gp.y - origin.position.y) / msg.info.resolution) * TARGET_SIZE / h

            outer_r = TARGET_SIZE / 30.0
            inner_r = outer_r * 0.5
            border_w = max(2, int(TARGET_SIZE / 200))

            # You could rotate by goal yaw if desired:
            # yaw_goal = self.quaternion_to_yaw(self.latest_goal.pose.orientation)
            # pts = star_points(sx, sy, outer_r, inner_r, rotation=yaw_goal - math.pi/2)
            pts = star_points(sx, sy, outer_r, inner_r)

            draw.polygon(pts, fill=(255, 0, 0))
            draw.line(pts + [pts[0]], fill=(0, 0, 0), width=border_w)
            roi_pts.append((sx, sy))

        # If no ROI points, save full image
        if not roi_pts:
            crop = img
        else:
            # Compute bounding box around ROI with padding
            xs, ys = zip(*roi_pts)
            min_x = max(0, int(min(xs) - ROI_PADDING))
            max_x = min(TARGET_SIZE, int(max(xs) + ROI_PADDING))
            min_y = max(0, int(min(ys) - ROI_PADDING))
            max_y = min(TARGET_SIZE, int(max(ys) + ROI_PADDING))
            crop = img.crop((min_x, min_y, max_x, max_y))

        draw2 = ImageDraw.Draw(crop)
        # Font for text
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 50)
        except:
            font = ImageFont.load_default()

        # Overlay terrain text
        if self.latest_terrain:
            text = f"Terrain: {self.latest_terrain}"
            margin = 5
            max_w = TARGET_SIZE - 2 * margin
            lines, line = [], ''
            for word in text.split():
                t = f"{line} {word}".strip()
                if draw2.textsize(t, font=font)[0] <= max_w:
                    line = t
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
            y = margin
            for ln in lines:
                draw2.text((margin, y), ln, fill=(0, 0, 0), font=font)
                y += font.getsize(ln)[1] + 5

        # Save cropped image
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'local_costmap_{ts}.png'
        
        if self.with_camera and self.latest_camera is not None:
            try:
                # Resize camera image to match costmap crop height
                cam_h, cam_w = self.latest_camera.shape[:2]
                target_h = crop.height
                scale = target_h / cam_h
                resized_w = int(cam_w * scale)
                camera_resized = cv2.resize(self.latest_camera, (resized_w, target_h))

                # Convert PIL image to OpenCV format
                costmap_rgb = np.array(crop.convert('RGB'))
                costmap_bgr = cv2.cvtColor(costmap_rgb, cv2.COLOR_RGB2BGR)

                # Match widths: crop or pad the camera image to match costmap width
                costmap_w = costmap_bgr.shape[1]
                camera_w = camera_resized.shape[1]

                if camera_w > costmap_w:
                    # Crop the camera image
                    camera_resized = camera_resized[:, :costmap_w]
                elif camera_w < costmap_w:
                    # Pad the camera image to the right
                    pad_width = costmap_w - camera_w
                    camera_resized = cv2.copyMakeBorder(
                        camera_resized, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )

                # Combine side by side
                combined = np.hstack((costmap_bgr, camera_resized))

                # Save combined image
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                fname = f'combined_costmap_camera_{ts}.png'
                cv2.imwrite(fname, combined)
                self.get_logger().info(f'Saved combined image → {fname}')
            except Exception as e:
                self.get_logger().error(f'Error processing camera image: {e}')
        else:
            # Fallback if camera not available
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f'local_costmap_{ts}.png'
            crop.save(fname)
            self.get_logger().info(f'Saved cropped costmap image → {fname}')


def main():
    parser = argparse.ArgumentParser(description='Costmap Saver Node')
    parser.add_argument('--with-camera', action='store_true',
                        help='Save costmap image with camera image side-by-side')
    args = parser.parse_args()

    rclpy.init()
    node = CostmapSaver(with_camera=args.with_camera)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
