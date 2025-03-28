import numpy as np
import matplotlib.pyplot as plt

kresty = np.array([[1, 0, 1],
                  [0, 1, 0],
                  [1, 0, 1]])
plusy = np.array([[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]])
def counting(image, pattern):
    count = 0
    pattern_size = pattern.shape[0]
    for i in range(image.shape[0] - pattern_size + 1):
        for j in range(image.shape[1] - pattern_size + 1):
            eq = image[i:i+pattern_size, j:j+pattern_size]
            if np.array_equal(eq, pattern):
                count += 1
    return count
image1 = np.load(r"stars.npy")
plt.imshow(image1)
plt.show()

stars_count = counting(image1, plusy)+counting(image1, kresty)

print(f"количество звездочек: {stars_count}")
