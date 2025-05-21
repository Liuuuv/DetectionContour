import matplotlib.pyplot as plt
import numpy as np
import time

image = plt.imread("justdisappear.png")     # taille (573, 640, 3)




image = image[::1, ::1, :]  # 5:taille (115, 128, 3), 4:taille (144, 160, 3)

class Filter:
    def __init__(self, center: np.ndarray, array: np.ndarray):
        self.center = center    # np.array([i,j])
        self.array = array
        self.shape = np.array((len(array), len(array[0])))
        
        print("shape ",self.shape)
        
        self.right_range = self.shape - self.center
        
        self.left_range = self.shape - self.right_range
        # self.right_range -= np.array([1,1])
        
        
        
        # print("ranges ", self.left_range, self.right_range)
        # print(-self.left_range[0], self.right_range[0]-1)
        # print(-self.left_range[1], self.right_range[1]-1)


# def black_and_white(img_):
#     img = img_.copy()
#     for i in range(len(img_)):
#         for j in range(len(img_[i])):
#             mean = np.mean(img_[i,j])
#             img[i,j] = [mean, mean, mean]
#     return img

def black_and_white(img_):
    # Convert to grayscale using luminance formula (better than simple mean)
    grayscale = np.dot(img_[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Stack the grayscale values into 3 channels
    return np.stack((grayscale,)*3, axis=-1).astype(img_.dtype)


def convolution(img_: np.ndarray, filter: Filter):  # 0 outside
    img = img_.copy()
    for i in range(len(img)):
        for j in range(len(img[i])):
            coef = 0
            for k in range(-filter.left_range[0], filter.right_range[0]):
                for l in range(-filter.left_range[1], filter.right_range[1]):
                    if not 0 <= i-k < img.shape[0] or not 0 <= j-l < img.shape[1]:
                        continue

                    coef += img_[i-k,j-l][0]*filter.array[filter.center[0]+k,filter.center[1]+l]
            
            # img[i,j] = 0
            img[i,j] = abs(coef)
            # if coef >= 0:
            #     img[i,j] = [0, coef, 0]
            # else:
            #     img[i,j] = [abs(coef), 0, 0]
            
    return img

from scipy.signal import convolve2d

# def convolution(img_: np.ndarray, filter: 'Filter') -> np.ndarray:
#     if img_.ndim == 3:
#         return np.stack([convolve2d(img_[...,c], filter.array, mode='same') 
#                        for c in range(img_.shape[2])], axis=-1)
#     return convolve2d(img_, filter.array, mode='same')

import numpy as np

import numpy as np

def convolution(img_: np.ndarray, filter: 'Filter') -> np.ndarray:
    """
    Convolution 2D/3D corrigée avec gestion appropriée du padding
    
    Args:
        img_: Image d'entrée (2D pour grayscale, 3D pour couleur)
        filter: Objet filtre avec:
            array: noyau de convolution 2D
            center: position du centre (tuple)
    
    Returns:
        Image convoluée (même shape que l'entrée)
    """
    # Vérification des dimensions
    if img_.ndim not in [2, 3]:
        raise ValueError("L'image doit être 2D (grayscale) ou 3D (couleur)")
    
    # Paramètres du filtre
    kernel = filter.array
    k_h, k_w = kernel.shape
    c_x, c_y = filter.center
    
    # Calcul du padding nécessaire
    pad_h = (k_h - 1) // 2
    pad_w = (k_w - 1) // 2
    
    # Application du padding selon les dimensions de l'image
    if img_.ndim == 3:
        # Image couleur (3D)
        padded = np.pad(img_, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 
                      mode='constant', constant_values=0)
    else:
        # Image grayscale (2D)
        padded = np.pad(img_, ((pad_h, pad_h), (pad_w, pad_w)), 
                      mode='constant', constant_values=0)
    
    # Initialisation de la sortie
    result = np.zeros_like(img_)
    
    # Convolution
    for i in range(img_.shape[0]):
        for j in range(img_.shape[1]):
            if img_.ndim == 3:
                # Traitement par canal pour les images couleur
                for c in range(img_.shape[2]):
                    result[i,j,c] = np.sum(padded[i:i+k_h, j:j+k_w, c] * kernel)
            else:
                # Traitement simple pour grayscale
                result[i,j] = np.sum(padded[i:i+k_h, j:j+k_w] * kernel)
    
    # Normalisation et conversion de type
    return np.clip(result, 0, 255).astype(np.uint8)


def get_magnitude(img_x, img_y):
    img = img_x.copy()
    for i in range(len(img_x)):
        for j in range(len(img_x[i])):
            magnitude = np.sqrt(img_x[i,j]**2 + img_y[i,j]**2)
            # magnitude = np.abs(img_x[i,j]) + np.abs(img_y[i,j])
            img[i,j] = magnitude
    return img


## df/dx
filterx = Filter(
    np.array([0,1]),
    np.array([
        [1,-1]
        ])
)

## df/dy
filtery = Filter(
    np.array([1,0]), 
    np.array([
        [1],
        [-1]
        ])
)

filter = Filter(
    np.array([1,1]), 
    np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]
        ])
)



image = black_and_white(image)




# image_x = convolution(image, filterx)
# image_y = convolution(image, filtery)

start_time = time.time()
image = convolution(image, filter)
end_time = time.time()
print(end_time - start_time)


# image = get_magnitude(image_x, image_y)


plt.imshow(image)
plt.show()