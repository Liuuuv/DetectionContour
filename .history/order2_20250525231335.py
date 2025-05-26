from contours import*



## + + GREEN
## - - RED
## + - YELLOW
img_ = image.copy()
for i in range(len(image)):
    for j in range(len(image[0])):
        img_xx = convolution(image, filterxx)
        img_yy = convolution(image, filteryy)
        img_xy = convolution(image, filterxy)
        
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
        else:
            img_[i,j] = [0,eigs[0] + eigs[1], 0]


plot(img_)

plt.show()