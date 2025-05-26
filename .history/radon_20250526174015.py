from contours import*


img = np.zeros_like(image)
for i in range(image):
    for j in range(image[0]):
        line = []
        for x in range(image.shape[1]):
            if i*x+j < image.shape[0]:
                line.append((x,int(i*x+j)))
        coef = np.sum([image[k,l] for k,l in line])
        img[i,j] += coef

plt.imshow(img)