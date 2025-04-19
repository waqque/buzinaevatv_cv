import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
# a = cv2.imread("images/img (6).jpg") 
# hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
# print(hsv[1875, 2421])
# plt.imshow(a)
# plt.show()

color_low = np.array([5, 120, 110])  
color_high = np.array([110, 250, 220]) 

count = 0  
for num in range(1, 13):
    orig = cv2.imread(f"images/img ({num}).jpg")
    hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_low, color_high)
    mask = cv2.dilate(mask, np.ones((9, 9)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9)), iterations=3)
    labeled = label(mask)
    regions = regionprops(labeled)
    
    region_penc = [region for region in regions  if (region.area > 83000 and (1 - region.eccentricity) < 0.04)]
    
    pencils = len(region_penc)
    count += pencils
    
    print(f"на изображении {num} обнаружено карандашей: {pencils}")

print(f"суммарное кол-во карандашей на всех изображениях: {count}")