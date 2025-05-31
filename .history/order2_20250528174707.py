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
            
            
            # np.sort(eigs)
            # if eigs[1] < 0:     # both are negative
            #     img_[i,j] = [- eigs[0] - eigs[1], 0, 0]
            # elif eigs[0] < 0: # 0 neg, 1 pos
            #     img_[i,j] = [-eigs[0], 0, eigs[1]]
            # else:
            #     img_[i,j] = [0,eigs[0] + eigs[1], 0]
            
            val = np.sum(np.abs(eigs))
            img_[i,j] = [val,val,val]
            
    img_ /= np.max(img_)
    
    return img_


def det_hess(img):
    image_xy = convolution(img, filterxy)
    image_xx = convolution(img, filterxx)
    image_yy = convolution(img, filteryy)
    
    return np.abs(image_xx * image_yy - image_xy ** 2)

# image1 = eigenvalues(image)
# imagemean = convolution(image, filtermean)
# imagemean = convolution(imagemean, filtermean)
# imagemean = convolution(imagemean, filtermean)
# imagemean = convolution(imagemean, filtermean)
# imagemean = convolution(imagemean, filtermean)
# image2 = eigenvalues(imagemean)

# plot(image)
# plot(image1)
# plot(image2)
# plot(convolution(image,laplacian_filter))
# plt.imshow(img_)

# temp
from scipy.ndimage import gaussian_filter

def analyse_hessienne(img_, sigma=1.0):
    img = img_[:,:,0]
    # Lissage gaussien
    img = gaussian_filter(img, sigma)
    
    # Dérivées secondes
    Ixx = gaussian_filter(img, sigma, order=(2, 0))
    Iyy = gaussian_filter(img, sigma, order=(0, 2))
    Ixy = gaussian_filter(img, sigma, order=(1, 1))
    
    # Calcul de la trace et du déterminant
    tr = Ixx + Iyy
    det = Ixx * Iyy - Ixy**2
    
    # Classification
    contours = (det < -0.0000001)  # Contours nets
    coins = (det > 0.001) & (np.abs(tr) > 0.001)  # Jonctions
    
    return contours, coins

# Application sur une image de contours

# contours, coins = analyse_hessienne(image)
# contours = np.stack((contours,)*3, axis=-1).astype(np.float32)
# coins = np.stack((coins,)*3, axis=-1).astype(np.float32)
# plot(contours)
# plot(coins)

image1 = det_hess(image)
image1 = threshold(image1,)


plt.show()


