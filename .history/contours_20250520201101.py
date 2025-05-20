import matplotlib.pyplot as plt
import numpy as np

image = plt.imread("justdisappear.png")     # taille (573, 640, 3)




image = image[::5, ::5, :]  # taille (115, 128, 3)



def black_and_white(img):
    img.duplicate()
    for i in range(len(img)):
        for j in range(len(img[i])):
            mean = np.mean(img[i,j])
            img[i,j] = [mean, mean, mean]
    return img

image = black_and_white(image)

print(image.shape)

plt.imshow(image)
plt.show()