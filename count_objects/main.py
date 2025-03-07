import numpy as np
import matplotlib.pyplot as plt


external = np.diag([1, 1, 1, 1]).reshape(4, 2, 2)

internal = np.logical_not(external)

cross = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])


def match(a, masks):
    for mask in masks:
        if np.all((a!=0) == (mask!=0)):
            return True
    return False


def count_objects(image):
    E = 0
    for y in range(0, image.shape[0] - 1):
        for x in range(0, image.shape[1] - 1):
            sub = image[y : y + 2, x : x + 2]
            if match(sub, external):
                E += 1
            elif match(sub, internal):
                E -= 1
            elif match(sub, cross):
                E += 2
    return E / 4
image1 = np.load(r"example1.npy")
image2 = np.load(r"example2.npy")
print("objects on the first image:", count_objects(image1)) 
c = 0
for num in range(image2.shape[2]): 
        lay = image2[:, :, num]
        c += count_objects(lay)
print("objects on the second image:", c)




