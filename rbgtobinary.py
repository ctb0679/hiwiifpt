import cv2

# Load the RGB image
image = cv2.imread('/home/junaidali/catkin_ws/src/hiwiifpt/image129.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Display the binary image
cv2.imshow('Binary Image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the binary image
cv2.imwrite('binary_image.jpeg', binary)
