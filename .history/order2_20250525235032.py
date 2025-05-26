from contours import*



## + + GREEN
## - - RED
## + - YELLOW



def eigenvalues(img):
    
    img_xx = convolution(img, filterxx)
    img_yy = convolution(img, filteryy)
    img_xy = convolution(img, filterxy)
    
    img_ = np.zeros_like(img)
    for i in range(len(image)):
        for j in range(len(image[0])):
            
            
            hessian = np.array([
                [img_xx[i,j][0], img_xy[i,j][0]],
                [img_xy[i,j][0], img_yy[i,j][0]]
            ])
            
            eigs = np.linalg.eigvals(hessian)
            
            
            np.sort(eigs)
            if eigs[1] < 0:     # both are negative
                img_[i,j] = [- eigs[0] - eigs[1], 0, 0]
            elif eigs[0] < 0: # 0 neg, 1 pos
                img_[i,j] = [-eigs[0], eigs[1], 0]
                print(i,j)
            else:
                img_[i,j] = [0,eigs[0] + eigs[1], 0]
            
            # img_[i,j] = []
            
    img_ /= np.max(img_)
    
    return img_

image1 = eigenvalues(image)
imagemean = convolution(image, filtermean)
imagemean = convolution(imagemean, filtermean)
image2 = eigenvalues(imagemean)

plot(image)
plot(imagemean)
plot(image1)
plot(image2)
# plt.imshow(img_)

plt.show()