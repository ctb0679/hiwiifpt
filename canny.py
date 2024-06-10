import cv2
import numpy as np

# Load the image (replace with the correct path if needed)
image_path = '/home/junaidali/catkin_ws/src/hiwiifpt/Binary_Image_screenshot_28.05.2024.png'
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is None:
    print(f"Error: Could not read the image file {image_path}. Please check the file path and try again.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Perform morphological operations to clean up the edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and extract the window frame contour based on size and shape
    window_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]  # Adjust threshold as needed

    # Draw contours on the original image (optional)
    contour_image = image.copy()
    cv2.drawContours(contour_image, window_contours, -1, (0, 255, 0), 2)

    # Save or display the results
    cv2.imwrite('contour_image.jpg', contour_image)
    cv2.imshow('Contours', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Assuming there is one main window frame contour
    if window_contours:
        window_frame = window_contours[0]
        # Convert the contour to a format for the robot (e.g., list of coordinates)
        window_frame_coordinates = window_frame.reshape(-1, 2).tolist()

        # Output the coordinates
        print(window_frame_coordinates)
