import cv2
from rembg import remove
from PIL import Image, ImageOps
import json

print("OpenCV version: " + cv2.__version__)

original_image_path = 'input/girl.png'
resized_image_path = 'output/resized_image.png'
bg_removed_path = 'output/bg_removed.png'
bw_path = 'output/bw.png'
contours_path = 'output/contours.png'
largest_contour_json_path = 'output/contour.json'

# Open an image file
img = cv2.imread(original_image_path)
width, height = img.shape[:2]
print("image size: ", width, height)

max_size = 100
scaling_factor = max_size / max(width, height)

img_resized = cv2.resize(
    img,
    (int(height * scaling_factor), int(width * scaling_factor), )
)

cv2.imwrite(resized_image_path, img_resized)

with open(resized_image_path, 'rb') as i:
    with open(bg_removed_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)

bw_rem_img = Image.open(bg_removed_path)

alpha = bw_rem_img.split()[3]
inverted_alpha = ImageOps.invert(alpha)
bw = inverted_alpha.convert('L') 
bw.save(bw_path)

bw_img = cv2.imread(bw_path)
gray = cv2.cvtColor(bw_img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(bw_img, contours, -1, (0, 255, 0), 2)
cv2.imwrite(contours_path, bw_img)

print("Number of contours found = ", len(contours))
for contour in contours:
  print("Points in contour: ", len(contour))

largest_contour_index = -1
max_area = 0
for i in range(len(contours)):
  min_x = 1000000
  min_y = 1000000
  max_x = 0
  max_y = 0
  for point in contours[i]:
    x = point[0][0]
    y = point[0][1]
    if x < min_x:
      min_x = x 
    if y < min_y:
      min_y = y
    if x > max_x:
      max_x = x
    if y > max_y:
      max_y = y
  # Removing contours that are on the edge of the image.
  if min_x < max_size / 100 or min_y < max_size / 100 or max_x > width - max_size / 100 or max_y > height - max_size / 100:
    continue
  area = (max_x - min_x) * (max_y - min_y)
  if area > max_area:
    max_area = area
    largest_contour_index = i

if largest_contour_index == -1:
  raise Exception("No largest contour found")

largest_countour = contours[largest_contour_index]
data = {}
points = []
data["points"] = points
for point in largest_countour:
  points.append([float(point[0][0]), float(point[0][1])])

with open(largest_contour_json_path, 'w') as outfile:
  outfile.write(json.dumps(data))

cv2.imshow('Image with contours', bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()