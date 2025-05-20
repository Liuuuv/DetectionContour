import matplotlib.pyplot as plt

image = plt.imread("justdisappear.png")

print(image.shape)

plt.imshow(image)
plt.show()