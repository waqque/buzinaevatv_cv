import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv

def find_clusters(sorted_values: np.array):
    differences = np.diff(sorted_values)
    threshold = np.std(differences) * 2
    cluster_boundaries = np.where(differences > threshold)
    return cluster_boundaries[0] + 1

def determine_hue(shape_region, original_image):
    center_y, center_x = shape_region.centroid
    hue_component = rgb2hsv(original_image[int(center_y), int(center_x)])[0]
    
    hue_ranges = {
        "red": 0.19202898,
        "orange": 0.30476192,
        "yellow": 0.41509435,
        "green": 0.60897434,
        "blue": 0.8333333
    }
    
    if 0.0 <= hue_component < hue_ranges['red']:
        return "red"
    elif hue_component < hue_ranges['orange']:
        return "orange"
    elif hue_component < hue_ranges['yellow']:
        return "yellow"
    elif hue_component < hue_ranges['green']:
        return "green"
    elif hue_component < hue_ranges['blue']:
        return "blue"
    else:
        return "violet"
    
input_image = plt.imread("./balls_and_rects.png")
grayscale = input_image.mean(axis=2)
binary_mask = grayscale > 0
labeled_mask = label(binary_mask)
detected_shapes = regionprops(labeled_mask)

circles_by_color = {}
squares_by_color = {}

for shape in detected_shapes:
    dominant_color = determine_hue(shape, input_image)
    if shape.eccentricity == 0:  # Circle (eccentricity ≈ 0)
        if dominant_color not in circles_by_color:
            circles_by_color[dominant_color] = 0
        circles_by_color[dominant_color] += 1
    else:  # Rectangle (eccentricity > 0)
        if dominant_color not in squares_by_color:
            squares_by_color[dominant_color] = 0
        squares_by_color[dominant_color] += 1

total_objects = sum(circles_by_color.values()) + sum(squares_by_color.values())
print("Total objects detected:", total_objects)
print("Circles:", circles_by_color)
print("Rectangles:", squares_by_color)
