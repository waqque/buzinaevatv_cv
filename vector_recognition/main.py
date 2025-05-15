import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops

def calculate_cavities(segment):
    dimensions = segment.image.shape
    padded = np.zeros((dimensions[0] + 2, dimensions[1] + 2))
    padded[1:-1, 1:-1] = segment.image
    padded = np.logical_not(padded)
    marked = label(padded)
    return np.max(marked) - 1


def get_features(segment):
    img = segment.image
    size_ratio = segment.area / img.size
    center_y, center_x = segment.centroid_local
    center_y /= img.shape[0]
    center_x /= img.shape[1]
    boundary_length = segment.perimeter / img.size
    elongation = segment.eccentricity
    vertical_lines = (np.sum(img, 0) == img.shape[0])
    vertical_lines = np.sum(vertical_lines)
    horizontal_lines = (np.sum(img, 1) == img.shape[1])
    horizontal_lines = np.sum(horizontal_lines)
    height_to_width = img.shape[0] / img.shape[1]
    cavities = calculate_cavities(segment)
    roundness = (boundary_length ** 2) / (4 * np.pi * size_ratio) if size_ratio > 0 else 0

    return np.array([
        size_ratio,
        center_y,
        center_x,
        boundary_length,
        elongation,
        vertical_lines,
        horizontal_lines,
        height_to_width,
        cavities,
        roundness
    ])


def compute_distance(vec1, vec2):
    return ((vec1 - vec2) ** 2).sum() ** 0.5


def identify_symbol(features, references):
    symbol = "_"
    min_distance = float('inf')
    for char in references:
        dist = compute_distance(features, references[char])
        if dist < min_distance:
            symbol = char
            min_distance = dist
    return symbol

symbol_image = plt.imread("alphabet-small.png")
grayscale = symbol_image.mean(axis=2)
threshold = grayscale < 1
labeled_img = label(threshold)
segments = regionprops(labeled_img)

reference_data = {
    "A": get_features(segments[2]),
    "B": get_features(segments[3]),
    "8": get_features(segments[0]),
    "0": get_features(segments[1]),
    "1": get_features(segments[4]),
    "W": get_features(segments[5]),
    "X": get_features(segments[6]),
    "*": get_features(segments[7]),
    "-": get_features(segments[9]),
    "/": get_features(segments[8])
}

test_image = plt.imread("alphabet.png")[:, :, :-1]
test_grayscale = test_image.mean(axis=2)
test_threshold = test_grayscale > 0
test_labeled = label(test_threshold)
test_segments = regionprops(test_labeled)

plt.figure(figsize=(15, 10))
for idx, segment in enumerate(test_segments):
    if idx >= 15:
        break
    features = get_features(segment)
    plt.subplot(3, 5, idx + 1)
    plt.title(identify_symbol(features, reference_data))
    plt.imshow(segment.image)
plt.tight_layout()
plt.show()