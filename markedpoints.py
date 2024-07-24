import numpy as np
import cv2  # Assuming you use OpenCV for image processing

# Load your edge-detected image (replace with your image loading mechanism)
edge_image = cv2.imread('edgeimage.jpg', cv2.IMREAD_GRAYSCALE)

# Determine number of parts to divide the image into
n = 5  # Example: dividing into 5 parts

# Get image dimensions
height, width = edge_image.shape[:2]

# Calculate vertical line positions
line_positions = [int(k * width / n) for k in range(1, n)]

# Create a list to store intersection points
intersection_points = []

# Iterate through each vertical line position
for x in line_positions:
    for y in range(height):
        if edge_image[y, x] > 0:  # Check if pixel intensity indicates an edge
            intersection_points.append((x, y))
            # Optionally, draw a circle or mark the intersection point on the image
            cv2.circle(edge_image, (x, y), radius=3, color=(255, 0, 0), thickness=-1)  # Example: Red circle

cv2.imshow('Edges', edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save or display the image with marked intersection points
cv2.imwrite('edge_image_with_points.jpg', edge_image)
