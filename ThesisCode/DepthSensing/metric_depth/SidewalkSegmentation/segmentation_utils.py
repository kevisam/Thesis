from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

def points_to_calc_distance(image,largest_contour):
    # Find the moments of the largest contour
    M = cv.moments(largest_contour)

    # Calculate the centroid
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])

    # Draw a horizontal line across the centroid
    cv.line(image, (0, centroid_y), (image.shape[1], centroid_y), (0, 255, 0), 2)

    # Get contour points that intersect with the horizontal line
    # Define a threshold for the distance from the line
    threshold_distance = 5

    # Get contour points that intersect with the horizontal line
    intersecting_points = []
    for point in largest_contour:
        x, y = point[0]
        if abs(y - centroid_y) < threshold_distance:
            intersecting_points.append((x, y))
    return intersecting_points

def find_sidewalk_contours(image):
    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply thresholding to segment the white objects
    _, thresholded_image = cv.threshold(gray_image, 240, 255, cv.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv.findContours(thresholded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv.contourArea)
    
    return largest_contour

def make_mask(model,logits):
    # Get predicted classes
    predicted_classes = np.argmax(logits, axis=0)

    # Get class labels from the model
    class_labels = model.config.id2label

    # Define a colormap for visualization
    cmap = plt.cm.get_cmap('binary_r', len(class_labels))

    # Filter out pixels corresponding to flat sidewalk class
    flat_sidewalk_class = 2  # Assuming flat sidewalk class is labeled as 2
    flat_sidewalk_mask = (predicted_classes == flat_sidewalk_class)

    # Create a figure and axes for the plot
    plt.figure(figsize=(10, 5))

    # Display only the flat sidewalk class in the segmentation mask
    masked_predicted_classes = np.where(flat_sidewalk_mask, 1, 0)  # Invert mask
    plt.imshow(masked_predicted_classes, cmap='binary_r', interpolation='nearest', vmin=0, vmax=1)  # Use binary colormap for black and white
    plt.axis('off')

    plt.savefig('sidewalk_mask.png', bbox_inches='tight', pad_inches=0) 



def average_x_coordinate(points):
    if not points:
        return None
    
    sum_x = 0
    
    for point in points:
        sum_x += point[0]
    
    return sum_x/len(points)

def average_point(coordinates):
    if not coordinates:
        return None
    
    sum_x = 0
    sum_y = 0
    
    for x, y in coordinates:
        sum_x += x
        sum_y += y
    
    mean_x = round(sum_x / len(coordinates))
    mean_y = round(sum_y / len(coordinates))
    
    return (mean_x, mean_y)

def separate_sides(closest_points,threshold):
    side1 = []
    side2 = []
    
    for i in range(len(closest_points)):
        if i == 0:
            side1.append(closest_points[i])
        elif abs(closest_points[i][0] - average_x_coordinate(side1)) > threshold:
            side2.append(closest_points[i])
        else:
            side1.append(closest_points[i])

    return side1,side2