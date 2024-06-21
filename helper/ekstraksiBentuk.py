import cv2
import numpy as np
from skimage import exposure
import os
from scipy.spatial import distance

def ekstrakBentuk(image):
    # Load the image in grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(imageGray, (512, 512))  # Resize the image to 512x512


    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Apply histogram equalization to improve contrast
    equalized_image = exposure.equalize_hist(blurred_image)
    equalized_image = (equalized_image * 255).astype(np.uint8)

    # Apply Sobel operator to detect edges
    sobel_x = cv2.Sobel(equalized_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(equalized_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobel_x, sobel_y)
    sobel = np.uint8(sobel / np.max(sobel) * 255)
    
    # Find contours in the Sobel image
    contours, _ = cv2.findContours(sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming we are interested in the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments to find the centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    # Calculate area
    area = M["m00"]
    
    # Calculate perimeter (length)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate the major axis length
    distances = [distance.euclidean((cx, cy), point[0]) for point in contour]
    major_axis_length = max(distances)
    
    # Calculate the minor axis length
    minor_axis_length = min(distances)
    
    # Calculate diameter
    diameter = (major_axis_length + minor_axis_length) / 2
    
    # Calculate shape factor
    shape_factor = (perimeter ** 2) / (4 * np.pi * area)
    
        
    return major_axis_length, perimeter, diameter, area, shape_factor


