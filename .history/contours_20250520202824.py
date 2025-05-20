import matplotlib.pyplot as plt
import numpy as np

image = plt.imread("justdisappear.png")     # taille (573, 640, 3)




image = image[::4, ::4, :]  # 5:taille (115, 128, 3), 4:taille (144, 160, 3)

class Filter:
    def __init__(self, center: tuple, array: np.ndarray)


def black_and_white(img):
    img.copy()
    for i in range(len(img)):
        for j in range(len(img[i])):
            mean = np.mean(img[i,j])
            img[i,j] = [mean, mean, mean]
    return img

# image = black_and_white(image)

def convolution(img, filter):
    pass





plt.imshow(image)
plt.show()