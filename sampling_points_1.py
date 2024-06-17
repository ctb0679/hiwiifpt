import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the edge image
edge_image_path = 'Edge_image.png'
edge_image = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)

# Display the image and prompt user to select ROI
roi_top_left = None
roi_bottom_right = None
roi_defined = False

def draw_rectangle(event, x, y, flags, param):
    global roi_top_left, roi_bottom_right, roi_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        if not roi_defined:
            roi_top_left = (x, y)
            roi_defined = True
        else:
            roi_bottom_right = (x, y)

            # Draw ROI rectangle on the image
            cv2.rectangle(edge_image, roi_top_left, roi_bottom_right, (255, 255, 255), 1)
            cv2.imshow("Select ROI", edge_image)

            # Break the loop after selecting both points
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Display the image and set mouse callback
cv2.imshow("Select ROI", edge_image)
cv2.setMouseCallback("Select ROI", draw_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ensure ROI is properly defined
if roi_top_left and roi_bottom_right:
    # Detect contours in the edge image
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which is likely to be the flap
    contour = max(contours, key=cv2.contourArea)

    # Function to check if a point is within the ROI
    def is_point_in_roi(point, top_left, bottom_right):
        x, y = point
        return top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]

    # Filter contour points that are within the ROI
    filtered_contour = [point for point in contour if is_point_in_roi(point[0], roi_top_left, roi_bottom_right)]

    # Function to sample equally spaced points along a contour
    def sample_contour_points(contour, num_points):
        # Calculate the arc length of the contour
        arc_length = cv2.arcLength(contour, closed=False)
        
        # Calculate equally spaced intervals along the contour
        interval = arc_length / num_points
        
        # Initialize variables for sampling
        sampled_points = []
        distance = 0

        # Traverse the contour to sample points
        current_distance = 0
        for i in range(len(contour)):
            if len(sampled_points) >= num_points:
                break
            # Get the current point and the next point on the contour
            point = contour[i][0]
            if i < len(contour) - 1:
                next_point = contour[i + 1][0]
            else:
                next_point = contour[0][0]
            
            # Calculate the distance to the next point
            distance_to_next_point = np.linalg.norm(next_point - point)
            
            while current_distance + distance_to_next_point >= distance:
                # Calculate the exact position on the segment
                ratio = (distance - current_distance) / distance_to_next_point
                x = int(point[0] + ratio * (next_point[0] - point[0]))
                y = int(point[1] + ratio * (next_point[1] - point[1]))
                sampled_points.append((x, y))
                distance += interval

            current_distance += distance_to_next_point
        
        return np.array(sampled_points)

    # Sample points along the filtered contour
    filtered_contour_np = np.array(filtered_contour)
    sampled_points = sample_contour_points(filtered_contour_np, 10)

    # Create an image to draw the sampled points
    output_image = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)

    # Draw the sampled points on the image
    for point in sampled_points:
        cv2.circle(output_image, tuple(point), 5, (0, 0, 255), -1)

    # Draw the ROI rectangle
    cv2.rectangle(output_image, roi_top_left, roi_bottom_right, (0, 255, 0), 2)

    # Display the image with sampled points
    plt.imshow(output_image)
    plt.title('Sampled Points on Edge Image')
    plt.show()

    # Save the image with sampled points
    cv2.imwrite('sampled_points_image_roi.png', output_image)
else:
    print("ROI was not properly defined.")
