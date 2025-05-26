from contours import*




img_ = image.copy()
for i in range(len(image)):
    for j in range(len(image[0])):
        img_xx = convolution(image, filterxx)
        img_yy = convolution(image, filteryy)
        img_xy = convolution(image, filterxy)
        
        hessian = np.array([
            [img_xx[i,j], img_xy[i,j]],
            [img_xy[i,j], img_yy[i,j]]
        ])
        
        eigs = np.linalg.eig(hessian)
        print(eigs)