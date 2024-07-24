import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_vertical_lines_on_image(input_image_path, output_image_path, line_distance):
    # Load the input image
    image = cv2.imread(input_image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to binary image using thresholding
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    # Detect edges using Canny edge detection
    edges = cv2.Canny(binary_image, 100, 200)
    
    # Increase the thickness of the edges
    kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Convert the edge image to a color image
    color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Get the dimensions of the image
    image_height, image_width = edges.shape[:2]

    # Temporary image for drawing lines
    temp_image = np.copy(color_edges)

    # List to store coordinates of highlighted points
    highlighted_points = []

    # Draw vertical white lines and highlight intersections
    for x in range(0, image_width, line_distance):
        for y in range(image_height):
            if edges[y, x] == 255:  # Check if there is an edge at this point
                cv2.circle(color_edges, (x, y), dot_radius, (0, 0, 255), -1)  # Draw a red dot
                highlighted_points.append((x, y))  # Store the coordinates of the intersection
                break
            temp_image[y, x] = (255, 255, 255)  # Draw the vertical line in the temporary image

    # Remove vertical white lines by copying the original thick edges to the final image
    final_image = np.copy(color_edges)

    # Save the resulting image to a file using matplotlib
    plt.imsave(output_image_path, cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    
    # Display the resulting image using matplotlib
    plt.imshow(cv2.cvtColor(color_edges, cv2.COLOR_BGR2RGB))
    plt.title('Vertical Lines and Intersections Highlighted')
    plt.axis('off')  # Hide axis
    plt.show()

    # Save the coordinates to a text file
    with open(coordinates_output_path, 'w') as file:
        for point in highlighted_points:
            file.write(f"{point[0]}, {point[1]}\n")

    # Return the list of highlighted points
    return highlighted_points

# Parameters
input_image_path = '/home/junaidali/catkin_ws/src/hiwiifpt/image129.jpeg'  # Path to the input image
output_image_path = '/home/junaidali/catkin_ws/src/hiwiifpt/flapWheel_sampledPoints_full.jpg'  # Path to save the output image
coordinates_output_path = '/home/junaidali/catkin_ws/src/hiwiifpt/coordinates_full.txt'  # Path to save the coordinates of the highlighted points
line_distance = 50  # Distance between vertical lines
edge_thickness = 3  # Thickness of the edges of the edge image
dot_radius = 3  # Radius of the red dots at intersection

# Draw vertical lines on the input image
draw_vertical_lines_on_image(input_image_path, output_image_path, line_distance)

