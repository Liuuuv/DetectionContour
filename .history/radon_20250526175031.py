from contours import*


img = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        line = []
        done = []
        for x in np.linspace(0,image.shape[0],10*image.shape[0]):
            if int(i*x+j) < image.shape[1] and not (int(x),int(i*x+j)) in line:
                line.append((int(x),int(i*x+j)))
        # print(line)
        coef = np.sum([image[k,l] for k,l in line])
        img[i,j] += coef
        

plt.imshow(img)
plt.show()