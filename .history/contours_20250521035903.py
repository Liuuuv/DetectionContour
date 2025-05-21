import matplotlib.pyplot as plt
import numpy as np
import time

image = plt.imread("justdisappear.png")     # taille (573, 640, 3)
# image = plt.imread("shinsei.png")


image = image[::1, ::1, :][..., :3]  # 5:taille (115, 128, 3), 4:taille (144, 160, 3)
noise = np.random.normal(0, 5, image.shape)

image += noise


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


# def convolution(img_: np.ndarray, filter: Filter):  # 0 outside
#     img = img_.copy()
#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             coef = 0
#             for k in range(-filter.left_range[0], filter.right_range[0]):
#                 for l in range(-filter.left_range[1], filter.right_range[1]):
#                     if not 0 <= i-k < img.shape[0] or not 0 <= j-l < img.shape[1]:
#                         continue

#                     coef += img_[i-k,j-l][0]*filter.array[filter.center[0]+k,filter.center[1]+l]
            
#             # img[i,j] = 0
#             img[i,j] = abs(coef)
#             # if coef >= 0:
#             #     img[i,j] = [0, coef, 0]
#             # else:
#             #     img[i,j] = [abs(coef), 0, 0]
#     return img

from scipy.signal import fftconvolve

def convolution(img_: np.ndarray, filter: 'Filter'):
    if img_.ndim != 3 or img_.shape[2] != 3:
        raise ValueError("L'image doit être au format HxWx3 (noir et blanc sur 3 canaux)")
    
    # Extraction d'un seul canal (le vert par défaut)
    img_gray = img_[:, :, 1].astype(np.float32)
    
    # Application convolution FFT
    convolved = fftconvolve(img_gray, filter.array, mode='same')
    
    # Valeur absolue et normalisation
    convolved = np.abs(convolved)
    convolved = np.clip(convolved, 0, 255).astype(img_.dtype)
    
    # Reconstruction des 3 canaux
    return np.stack((convolved,)*3, axis=-1)




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

# filter = Filter(
#     np.array([1,1]), 
#     np.array([
#         [0,1,0],
#         [1,-4,1],
#         [0,1,0]
#         ])
# )


start_time = time.time()
image = black_and_white(image)




image_x = convolution(image, filterx)
image_y = convolution(image, filtery)


# image = convolution(image, filter)



image = get_magnitude(image_x, image_y)

end_time = time.time()
print(end_time - start_time)
plt.imshow(image)
plt.show()