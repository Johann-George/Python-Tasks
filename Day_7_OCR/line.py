import cv2
import numpy as np

# --- Input / Output ---
image_path = "img.jpg"
output_path = "img_output_filtered.jpg"

# --- Load image and preprocess ---
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# --- Hough Line Transform ---
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=100,
    minLineLength=50,
    maxLineGap=10
)

vertical_lines = []
min_length = 100  # initial minimum for vertical filtering
x_merge_tolerance = 15  # tolerance for merging nearby lines

# --- Filter vertical lines ---
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # keep near-vertical and reasonably long lines
        if abs(angle) > 85 and length >= min_length:
            vertical_lines.append((x1, y1, x2, y2, length))

# --- Sort lines by x position ---
vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)

# --- Merge nearby vertical lines ---
merged_lines = []
if vertical_lines:
    current_group = [vertical_lines[0]]

    for line in vertical_lines[1:]:
        prev_x = np.mean([current_group[-1][0], current_group[-1][2]])
        curr_x = np.mean([line[0], line[2]])

        if abs(curr_x - prev_x) < x_merge_tolerance:
            current_group.append(line)
        else:
            merged_lines.append(max(current_group, key=lambda l: l[4]))
            current_group = [line]

    merged_lines.append(max(current_group, key=lambda l: l[4]))

# --- Draw only lines with length > 500 ---
display_lines = [line for line in merged_lines if line[4] > 500]

for x1, y1, x2, y2, length in display_lines:
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 4)
    cv2.putText(img, f"{int(length)}px", (x1 + 5, min(y1, y2) + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# --- Save output ---
cv2.imwrite(output_path, img)
print(f"✅ Output saved to: {output_path}")

# --- Print coordinates of displayed lines ---
print("\nVertical lines with length > 500 pixels:")
for coords in display_lines:
    print(coords[:4], f"Length = {int(coords[4])} px")
