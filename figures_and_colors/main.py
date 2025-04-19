import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
img = cv2.imread("balls_and_rects.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
labeled = label(binary)
circles = 0
rect = 0
regions = regionprops(labeled, intensity_image=gray)
for region in regions:
    ecc = region.eccentricity  
    if ecc < 0.2 :
            circles += 1
    else: 
            rect += 1
print(f"Всего фигур: {len(regions)}")
print(f"Всего кругов: {circles}")
print(f"Всего прямоугольников: {rect}")
