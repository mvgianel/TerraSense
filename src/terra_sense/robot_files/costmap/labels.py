from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- 1. Horizontal Gradient Legend with "FREE SPACE" and "LETHAL" ---
gradient_width = 800
bar_height = 80
margin_left = 60
margin_top = 40
spacing = 10
margin_right = 60
margin_bottom = 80  # space for labels under bar

# Define colors for gradient: cyan->blue->magenta->red
cyan = np.array([0, 255, 255], dtype=int)
blue = np.array([0, 0, 255], dtype=int)
magenta = np.array([255, 0, 255], dtype=int)
red = np.array([255, 0, 0], dtype=int)

# Fonts
try:
    font_label = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
except:
    font_label = ImageFont.load_default()

width = margin_left + gradient_width + margin_right
height = margin_top + bar_height + margin_bottom

legend = Image.new('RGB', (width, height), (255, 255, 255))
draw = ImageDraw.Draw(legend)

# Draw horizontal gradient
x_start = margin_left
y_start = margin_top

for x in range(gradient_width):
    frac = x / (gradient_width - 1)
    if frac <= 1/3:
        t = frac * 3
        color = cyan * (1 - t) + blue * t
    elif frac <= 2/3:
        t = (frac - 1/3) * 3
        color = blue * (1 - t) + magenta * t
    else:
        t = (frac - 2/3) * 3
        color = magenta * (1 - t) + red * t
    draw.line([(x_start + x, y_start), (x_start + x, y_start + bar_height)], fill=tuple(color.astype(int)))

draw.rectangle([x_start, y_start, x_start + gradient_width, y_start + bar_height], outline=(0,0,0), width=2)

# Draw labels: "FREE SPACE" and "LETHAL"
tick_y = y_start + bar_height
label_offset = 12
draw.text((x_start - 10, tick_y + label_offset), "FREE SPACE", fill=(0,0,0), font=font_label, anchor="lm")
draw.text((x_start + gradient_width - 10, tick_y + label_offset), "LETHAL", fill=(0,0,0), font=font_label, anchor="rm")

gradient_path = '/mnt/data/costmap_gradient_horizontal_labeled.png'
legend.save(gradient_path)

# --- 2. Symbol Legend: Star, Triangle, Line ---
icon_width, icon_height = 600, 120
icon_legend = Image.new('RGB', (icon_width, icon_height), (255, 255, 255))
draw2 = ImageDraw.Draw(icon_legend)

# Fonts
try:
    font_icon = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
except:
    font_icon = ImageFont.load_default()

# Line (Path)
line_start = (30, 20)
line_end = (180, 20)
draw2.line([line_start, line_end], fill=(0,255,0), width=12)
draw2.line([line_start, line_end], fill=(0,0,0), width=8)
draw2.text((200, 12), "Local Plan", fill=(0,0,0), font=font_icon)

# Triangle (Robot)
r = 10
tx, ty = 60, 60
triangle = [(tx + r, ty), (tx - r, ty - r), (tx - r, ty + r)]
draw2.polygon(triangle, fill=(255,255,0))
draw2.line(triangle + [triangle[0]], fill=(0,0,0), width=2)
draw2.text((90, ty - 10), "Robot Pose", fill=(0,0,0), font=font_icon)

# Star (Goal)
def star_points(cx, cy, outer_r, inner_r, num_points=5, rotation=-np.pi/2):
    pts = []
    for i in range(num_points * 2):
        angle = rotation + i * np.pi / num_points
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
    return pts

sx, sy = 300, 60
outer_r, inner_r = 12, 6
star = star_points(sx, sy, outer_r, inner_r)
draw2.polygon(star, fill=(255, 0, 0))
draw2.line(star + [star[0]], fill=(0, 0, 0), width=2)
draw2.text((sx + 20, sy - 10), "Goal Pose", fill=(0,0,0), font=font_icon)

icon_path = '/mnt/data/legend_icons_labeled.png'
icon_legend.save(icon_path)

gradient_path, icon_path
