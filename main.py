import cv2
import numpy as np
import json
from collections import defaultdict

# EDIT THESE:
IMAGE_PATH = "Examples/Source/Imperial.png"  # Edit this to change the image path
INACCURACY_VALUE = 0.0001  # RDP epsilon - lower = more polygons
NUM_COLORS = 256  # K-means colors
MIN_AREA = 0  # Min polygon area

image = cv2.imread(IMAGE_PATH, 1)
img_height, img_width = image.shape[:2]

# Convert BGR to RGB for correct colors
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# K-means segmentation
pixels = image_rgb.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
_, labels, palette = cv2.kmeans(
    pixels, NUM_COLORS, None, criteria, 10, cv2.KMEANS_PP_CENTERS
)
palette = np.uint8(palette)
labels = labels.flatten().reshape(img_height, img_width)

all_polygons = []

for color_idx in range(NUM_COLORS):
    mask = (labels == color_idx).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        # Get median color
        contour_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(image_rgb, mask=contour_mask)
        r, g, b = int(mean_color[0]), int(mean_color[1]), int(mean_color[2])
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        # Extract and simplify points
        points = [(pt[0][0], pt[0][1]) for pt in contour]

        def rdp_algorithm(points, epsilon):
            if len(points) <= 2:
                return points

            def perp_dist(point, line_start, line_end):
                x0, y0 = point
                x1, y1 = line_start
                x2, y2 = line_end
                if x2 == x1:
                    return abs(x0 - x1)
                if y2 == y1:
                    return abs(y0 - y1)
                num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
                den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                return num / den if den > 0 else 0

            dmax, index, end = 0, 0, len(points) - 1
            for i in range(1, end):
                d = perp_dist(points[i], points[0], points[end])
                if d > dmax:
                    index, dmax = i, d
            if dmax > epsilon:
                r1 = rdp_algorithm(points[: index + 1], epsilon)
                r2 = rdp_algorithm(points[index:], epsilon)
                return r1[:-1] + r2
            return [points[0], points[end]]

        simplified = rdp_algorithm(points, INACCURACY_VALUE)

        if len(simplified) < 3:
            continue

        fmt = "{:.4f}"
        desmos_points = []
        for x, y in simplified:
            dy = img_height - y
            desmos_points.append(f"({fmt.format(x)},{fmt.format(dy)})")

        latex = f"\\operatorname{{polygon}}({','.join(desmos_points)})"

        all_polygons.append({"area": area, "hex_color": hex_color, "latex": latex})

# Painter's Algorithm: Sort by area (LARGEST → SMALLEST)
all_polygons.sort(key=lambda p: p["area"], reverse=True)

# Generate JSON output
expressions = []
for poly in all_polygons:
    expressions.append(
        {
            "latex": poly["latex"],
            "color": poly["hex_color"],
            "fill": True,
            "fillOpacity": 1,
            "lineWidth": 0.6,
            "lineOpacity": 1,
            "lineColor": poly["hex_color"],
        }
    )

print(f"Generated {len(expressions)} polygons")
print(json.dumps(expressions))
