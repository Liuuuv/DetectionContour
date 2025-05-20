import matplotlib.pyplot as plt
import numpy as np

image = plt.imread("justdisappear.png")

# for i in range(len(image)):
#     for j in range(len(image[i])):
#         mean = np.mean(image[i,j])
#         image[i,j] = [mean, mean, mean]




print(image.shape)

plt.imshow(image)
plt.show()