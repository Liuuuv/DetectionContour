import matplotlib.pyplot as plt
import numpy as np

image = plt.imread("justdisappear.png")     # taille (573, 640, 3)


## black and white
# for i in range(len(image)):
#     for j in range(len(image[i])):
#         mean = np.mean(image[i,j])
#         image[i,j] = [mean, mean, mean]

# image = image.resize((128,128))
image = image[::5, ::5, :]  # Sous-échantillonnage tous les 2 pixels


print(image.shape)

plt.imshow(image)
plt.show()