import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # type: ignore
from scipy.interpolate import splprep, splev # type: ignore

def draw_dashed_line(image, start_point, end_point, color, thickness, dash_length):
    x1, y1 = start_point
    x2, y2 = end_point

    total_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    num_dashes = int(total_length // dash_length)

    for i in range(num_dashes):
        start = (int(x1 + (x2 - x1) * i / num_dashes), int(y1 + (y2 - y1) * i / num_dashes))
        end = (int(x1 + (x2 - x1) * (i + 0.5) / num_dashes), int(y1 + (y2 - y1) * (i + 0.5) / num_dashes))
        cv2.line(image, start, end, color, thickness)

def draw_vertical_lines_on_image(input_image_path, output_image_path, line_distance):
    # Load the input image
    image = cv2.imread(input_image_path)
        
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Convert to binary image using thresholding
    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    # Apply Gaussian blur to the binary image
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred_image, 100, 200)

    # Increase the thickness of the edges
    kernel = np.ones((thickness, thickness), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Get the coordinates of edge points
    edge_points = np.column_stack(np.where(edges > 0))

    # Extract x and y coordinates
    x_coords = edge_points[:, 1]
    y_coords = edge_points[:, 0]

    # Define the range and bins for the histogram
    x_min, x_max = x_coords.min(), x_coords.max()
    bins = np.arange(x_min, x_max + 1)

    # Calculate the histogram along the x-axis
    countsX, bin_edges = np.histogram(x_coords, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    '''
    # Limit the data for visualization
    countsX = countsX[:250]
    bin_centers = bin_centers[:250]
    '''

    # Find peaks in the histogram
    peaks, _ = find_peaks(countsX, height=75, distance=50)

    '''
    # Filter out peaks that are out of the truncated range
    valid_peaks = peaks[peaks < len(countsX)]
    '''

    '''
    # Get peak heights and sort them
    if valid_peaks.size > 0:
        peak_heights = countsX[valid_peaks]
        sorted_peaks = np.argsort(peak_heights)[::-1]
        top_two_peaks = valid_peaks[sorted_peaks[:2]]
    else:
        top_two_peaks = []
    '''

    # Check if there are at least two peaks found
    if len(peaks) >= 2:
        # Get peak heights and sort them
        peak_heights = countsX[peaks]
        sorted_peaks = np.argsort(peak_heights)[::-1]
        top_two_peaks = peaks[sorted_peaks[:2]]

        # Sort the top two peaks by x-coordinate (left to right)
        top_two_peaks = top_two_peaks[np.argsort(bin_centers[top_two_peaks])]
        
        # Extract x-coordinates of the peaks
        left_peak_x = bin_centers[top_two_peaks[0]]
        right_peak_x = bin_centers[top_two_peaks[1]]
        
        # Calculate pixel-to-mm conversion factor
        pixel_to_mm = 10.0 / (right_peak_x - left_peak_x)
        print(f"Conversion factor of 1 pixel is {pixel_to_mm} mm")

        # Round values to 2 decimal places
        left_peak_x = round(left_peak_x, 2)
        right_peak_x = round(right_peak_x, 2)

        print(f"Left peak X-coordinate: {round(left_peak_x * pixel_to_mm, 2)}")
        print(f"Right peak X-coordinate: {round(right_peak_x * pixel_to_mm, 2)}")
    else:
        print("Less than two peaks detected.")
        return

    '''
    # Consider histogram endpoints as potential peaks
    potential_peaks = np.concatenate(([0], valid_peaks, [len(countsX)-1]))
    potential_heights = countsX[potential_peaks]
    sorted_potential_peaks = np.argsort(potential_heights)[::-1]
    top_two_peaks_with_ends = potential_peaks[sorted_potential_peaks[:2]]
    '''

    # Consider histogram endpoints as potential peaks
    potential_peaks = np.concatenate(([0], peaks, [len(countsX)-1]))
    potential_heights = countsX[potential_peaks]
    sorted_potential_peaks = np.argsort(potential_heights)[::-1]
    top_two_peaks_with_ends = potential_peaks[sorted_potential_peaks[:2]]

    # Plotting
    plt.figure(figsize=(15, 6))

    # Original image with edges
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.contour(edges, colors='r', levels=[0.5])
    plt.title('Image with Edges')
    plt.axis('off')

    # Histogram and peaks
    plt.subplot(1, 2, 2)
    plt.bar(bin_centers, countsX, width=1, color='g', alpha=0.6, label='Histogram')
    plt.plot(bin_centers, countsX, label='Counts')

    if len(top_two_peaks) > 0:
        plt.scatter(bin_centers[top_two_peaks], countsX[top_two_peaks], color='r', s=50, zorder=5, label='Top Peaks')
        plt.axvline(bin_centers[top_two_peaks[0]], color='r', linestyle='--', label='Peak 1')
        if len(top_two_peaks) > 1:
            plt.axvline(bin_centers[top_two_peaks[1]], color='b', linestyle='--', label='Peak 2')

    plt.xlabel('X-coordinate')
    plt.ylabel('Number of Edge Points')
    plt.title('Histogram with Detected Peaks')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    ## For geting the sampled points within the required region (only Flap Wheel)

    # Increase the thickness of the edges
    kernel = np.ones((thickness, thickness), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Convert the edge image to a color image
    color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Get the dimensions of the image
    image_height, image_width = edges.shape[:2]

    # Temporary image for drawing lines
    temp_image = np.copy(color_edges)

    # List to store coordinates of highlighted points
    highlighted_points = []
###
    midpoints = []

    # Draw vertical white lines
    for x in range(0, image_width, line_distance):
        # Draw the vertical line on the copy image
        cv2.line(temp_image, (x, 0), (x, image_height - 1), (255, 255, 255), 1)

        # Highlight intersections with red dots
        intersections = []
        for y in range(image_height):
            if edges[y, x] == 255:  # Check if there is an edge at this point
                intersections.append(y)
                # cv2.circle(color_edges, (x, y), dot_radius, (0, 0, 255), -1)  # Draw a red dot

        if len(intersections) > 1:
            # Calculate the midpoint of the intersections
            midpoints.append(np.mean(intersections))

    # Select 10 midpoints from the middle of the list
    num_midpoints = len(midpoints)
    if num_midpoints > 10:
        start_idx = (num_midpoints - 10) // 2
        selected_midpoints = midpoints[start_idx:start_idx + 10]
    else:
        selected_midpoints = midpoints

    # Calculate the average of the selected midpoints
    if selected_midpoints:
        average_y = np.mean(selected_midpoints)
        
        # Draw a horizontal blue line at the average y-coordinate
        # cv2.line(color_edges, (0, int(average_y)), (image_width - 1, int(average_y)), (255, 0, 0), thickness)

    # Draw a horizontal dashed blue line at the average y-coordinate with increased thickness
    draw_dashed_line(color_edges, (0, int(average_y)), (image_width - 1, int(average_y)), (255, 0, 0), thickness, dash_length)

    # Draw vertical lines at the detected peaks
    for peak_x in [left_peak_x, right_peak_x]:
        cv2.line(color_edges, (int(peak_x), 0), (int(peak_x), image_height), (255, 0, 0), thickness)  # Blue vertical line


    # Remove the vertical lines from the final output
    final_image = color_edges.copy()
    for x in range(0, image_width, line_distance):
        cv2.line(final_image, (x, 0), (x, image_height - 1), (0, 0, 0), 1)
###


    # Draw vertical white lines and highlight intersections
    for x in range(int(left_peak_x), int(right_peak_x), line_distance):
        for y in range(image_height):
            if edges[y, x] == 255:  # Check if there is an edge at this point
                cv2.circle(color_edges, (x, y), dot_radius, (0, 0, 255), -1)  # Draw a red dot
                # Convert pixel coordinates to mm and adjust based on reference axes
                x_mm = (x - left_peak_x) * pixel_to_mm
                y_mm = (average_y - y) * pixel_to_mm  # Flip y-axis
                highlighted_points.append((x_mm, y_mm))  # Store the coordinates in mm
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

    # Save the coordinates in mm to a text file
    with open(coordinates_output_path, 'w') as file:
        for point in highlighted_points:
            file.write(f"{point[0]:.2f}, {point[1]:.2f}\n")

    # Return the list of highlighted points
    return highlighted_points

def plot_spline_from_coordinates(coordinates_file):
    # Read coordinates from the file
    with open(coordinates_output_path, 'r') as file:
        lines = file.readlines()

    # Parse the coordinates
    coordinates = []
    for line in lines:
        x, y = map(float, line.strip().split(', '))
        coordinates.append((x, y))

    # Convert to numpy array for easier manipulation
    coordinates = np.array(coordinates)

    # Sort coordinates by x-value
    sorted_coords = coordinates[np.argsort(coordinates[:, 0])]
    x_sorted = sorted_coords[:, 0]
    y_sorted = sorted_coords[:, 1]

    # Scale the coordinates
    # x_scaled = x_sorted * x_scale
    # y_scaled = y_sorted * y_scale


     # Fit a spline to the data points
    tck, u = splprep([x_sorted, y_sorted], s=0)
    unew = np.linspace(0, 1.0, num=len(x_sorted)*10)  # Generate points for smooth spline
    out = splev(unew, tck)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(out[0], out[1], 'b-', label='Spline')
    # plt.scatter(x_sorted, y_sorted, color='r', label='Sampled Points')
    plt.xlabel('X-coordinate (mm)')
    plt.ylabel('Y-coordinate (mm)')
    plt.title('Spline Through Sampled Points')
    # plt.gca().invert_yaxis()  # Flip the y-axis

    # Set axis limits for better scaling (optional)
    # plt.xlim(min(x_scaled) - 100, max(x_scaled) + 100)  # Extend range for X-axis
    # plt.ylim(min(y_scaled) - 100, max(y_scaled) + 100)  # Extend range for Y-axis

    # plt.gca().invert_yaxis()  # Flip the y-axis
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters
input_image_path = r'D:\My Files\HiWi IFPT\hiwiifpt\Images\30mm\2024_08_16-09_16_image_raw.jpeg'  # Path to the input image
output_image_path = r'D:\My Files\HiWi IFPT\hiwiifpt\outputs\test_30.jpg'  # Path to save the output image
coordinates_output_path = r'D:\My Files\HiWi IFPT\hiwiifpt\outputs\coordinates_test_30.txt'  # Path to save the coordinates of the highlighted points
line_distance = 5  # Distance between vertical lines
thickness = 2  # Thickness of the edges of the edge image
dot_radius = 3  # Radius of the red dots at intersection
x_scale = 1.5  # Scale factor for X-axis
y_scale = 1.2  # Scale factor for Y-axis
dash_length = 10  # Length of each dash in the dashed line

# Draw vertical lines on the input image
draw_vertical_lines_on_image(input_image_path, output_image_path, line_distance)
plot_spline_from_coordinates(coordinates_output_path)
