import cv2
import numpy as np
import glob
import os

# Load the saved camera matrix and distortion coefficients from .txt files
mtx = np.loadtxt('camera_matrix.txt')  # Replace with the path to your camera matrix file
dist = np.loadtxt('dist_coefficients.txt')  # Replace with the path to your distortion coefficients file

# Path to the folder containing the images
image_folder = 'flapProfiles/6x5profiles/'  # Replace with the actual folder path

# Output folder to save undistorted images
output_folder = 'flapProfiles/undistorted_6x5profiles/'  # Replace with the actual output folder path
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# Get all images in the folder (assuming .jpg, .png, etc.)
image_paths = glob.glob(os.path.join(image_folder, '*.*'))  # Adjust the extension filter as needed

# Process each image in the folder
for image_path in image_paths:
    # Load the pre-taken image from disk
    img = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if img is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Get the width and height of the image
    h, w = img.shape[:2]

    # Get the optimal new camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image if necessary
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    # Save the undistorted image to the output folder
    filename = os.path.basename(image_path)  # Get the original filename
    undistorted_image_path = os.path.join(output_folder, 'undistorted_' + filename)  # Create output path
    cv2.imwrite(undistorted_image_path, dst)
    print(f"Undistorted image saved at: {undistorted_image_path}")

print("Processing complete!")
