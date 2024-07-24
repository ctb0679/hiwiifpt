import cv2
import numpy as np

# Read the RGB image
rgb_img = cv2.imread('image129.jpeg')

# Check if the image was loaded properly
if rgb_img is None:
    print("Error: Image not loaded properly.")
    exit()

# Convert to grayscale
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

# Create white padding and add to the top of the grayscale image
white_padding = np.ones((50, gray_img.shape[1]), dtype=np.uint8) * 255
gray_img_with_padding = np.vstack((white_padding, gray_img))

# Invert grayscale image
gray_img_with_padding = 255 - gray_img_with_padding

# Apply thresholding
gray_img_with_padding[gray_img_with_padding > 100] = 255
gray_img_with_padding[gray_img_with_padding <= 100] = 0

# Apply morphological closing
kernel = np.ones((30, 30), np.uint8)
closing = cv2.morphologyEx(gray_img_with_padding, cv2.MORPH_CLOSE, kernel)

# Ensure the closing result is 8-bit single-channel before applying Canny
closing = closing.astype(np.uint8)

# Detect edges using Canny
edges = cv2.Canny(closing, 100, 200)

# Resize the edges image for display
resized_edges = cv2.resize(edges, (int(edges.shape[1]/2), int(edges.shape[0]/2)))

# Display the resized edges
cv2.imshow('Edges', resized_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('edgeimage.jpg', resized_edges)