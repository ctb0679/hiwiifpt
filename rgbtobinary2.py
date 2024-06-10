import cv2

# Load the RGB image
image = cv2.imread('/home/junaidali/catkin_ws/src/hiwiifpt/image129.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding (Mean)
binary_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)

# Apply adaptive thresholding (Gaussian)
binary_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

# Combine the binary images
binary_combined = cv2.bitwise_or(binary_mean, binary_gaussian)

# Display the binary images
cv2.imshow('Mean Adaptive Binary Image', binary_mean)
cv2.imshow('Gaussian Adaptive Binary Image', binary_gaussian)
cv2.imshow('Combined Binary Image', binary_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the binary images
cv2.imwrite('binary_mean_image.jpg', binary_mean)
cv2.imwrite('binary_gaussian_image.jpg', binary_gaussian)
cv2.imwrite('combined_binary_image.jpg', binary_combined)
