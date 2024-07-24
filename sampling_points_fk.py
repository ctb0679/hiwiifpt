import cv2
import numpy as np
import matplotlib.pyplot as plt

def select_roi(image):
    r = cv2.selectROI("Select Region", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region")
    return r

def main():
    # Load the image
    image_path = "image129.jpeg"
    image = cv2.imread(image_path)
    
    # Select ROI
    x, y, w, h = select_roi(image)
    
    # Crop the selected ROI
    roi = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to get the flap wheel
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    
    if not filtered_contours:
        print("No contours found. Try adjusting the area threshold or preprocessing steps.")
        return
    
    # Draw contours on the original image for visualization
    contour_image = roi.copy()
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
    
    # Find bounding box of the flap wheel
    x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(filtered_contours[0])
    
    # Sample points along the width of the flap wheel
    num_samples = 10
    interval = w_roi // (num_samples - 1)
    sample_points = [(x_roi + i * interval, y_roi + h_roi) for i in range(num_samples)]
    
    # Find intersection of sample lines with the flap profile
    profile_points = []
    for pt in sample_points:
        for dy in range(h_roi):
            if edges[pt[1] - dy, pt[0]] > 0:
                profile_points.append((pt[0], pt[1] - dy))
                break
    
    # Draw sampled points on the image
    for pt in profile_points:
        cv2.circle(contour_image, pt, 3, (255, 0, 0), -1)
    
    # Calculate pixel-to-mm conversion factor using the known diameter of the metal shaft (6 mm)
    shaft_diameter_mm = 6
    shaft_width_pixels = np.abs(sample_points[0][0] - sample_points[-1][0])
    pixel_to_mm = shaft_diameter_mm / shaft_width_pixels
    
    # Transform profile points from pixels to millimeters
    profile_points_mm = [(pt[0] * pixel_to_mm, pt[1] * pixel_to_mm) for pt in profile_points]
    
    # Save and display the results
    cv2.imwrite("output/contour_image.png", contour_image)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title("Sampled Flap Profile")
    plt.show()
    
    print("Profile points in mm:", profile_points_mm)

if __name__ == "__main__":
    main()
