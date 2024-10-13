import cv2
from matplotlib import pyplot as plt

# Load the image in grayscale
image_path = r'D:\My Files\HiWi IFPT\hiwiifpt\Images\40mm\Screenshot_test_40.png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Convert to binary image using thresholding
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply Gaussian blur to the binary image
blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

# Apply Canny edge detection on the blurred binary image
edges_blurred = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

# Display the original, binary, blurred, and edge-detected images
plt.figure(figsize=(24, 6))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Binary Image
plt.subplot(1, 4, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

# Blurred Image
plt.subplot(1, 4, 3)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

# Edge-detected Image on Blurred Image
plt.subplot(1, 4, 4)
plt.imshow(edges_blurred, cmap='gray')
plt.title('Edge Detection on Blurred Image')
plt.axis('off')

plt.show()
