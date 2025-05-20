import matplotlib.pyplot as plt

image = plt.imread("justdisappear.png")
for i in range(len(image)):
    for j in range(len(image[i])):
        if i<10:
            print(image[i,j])




print(image.shape)

plt.imshow(image)
plt.show()