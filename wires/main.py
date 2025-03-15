import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import(binary_closing, binary_dilation, binary_opening, binary_erosion)
data = np.load("wires6npy.txt")
labeled = label(data) #маркируем изображение
print("проводов на изображении всего:", np.max(labeled)) #cчитаем кол-во объектов. cколько проводов, не частей!!!!!
result = binary_erosion(data, np.ones(3).reshape(3, 1))#разделить на части
for count_wires in range(1, (np.max(labeled)+1)): #считаем каждый провод по отдельности
    wire = labeled == count_wires #текущий провод
    eros = binary_erosion(wire, np.ones(3).reshape(3, 1))#разделить на части
    parts = np.max(label(eros))#оставшиеся части
    if parts>1:
        print(f"провод {count_wires} порван на:" , parts, "частей")
    elif count_wires==np.max(labeled):
        print("не существует")
    else:
        print(f"с проводом {count_wires} все ок")


plt.imshow(result)
plt.show()
#каждый провод по отдельности!!