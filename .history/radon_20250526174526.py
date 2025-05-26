from contours import*


img = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        line = []
        for x in range(image.shape[0]):
            if i*x+j < image.shape[1]:
                line.append((x,i*x+j))
        # print(line)
        coef = np.sum([image[k,l] for k,l in line])
        img[i,j] += coef
        

plt.imshow(img)
plt.show()