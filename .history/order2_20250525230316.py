from contours import*




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
        
        eigs = np.linalg.eig(hessian)
        print(eigs)