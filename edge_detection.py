import cv2

# Load the binary image
binary_image = cv2.imread('/home/junaidali/catkin_ws/src/hiwiifpt/CombinedBinaryImage_screenshot_28.05.2024.png', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(binary_image, 30, 150)  # Adjust threshold values as needed

# Display the edge image
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the edge image
cv2.imwrite('edge_image.jpg', edges)
