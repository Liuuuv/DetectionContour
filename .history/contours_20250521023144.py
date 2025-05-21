import matplotlib.pyplot as plt
import numpy as np

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


def black_and_white(img_):
    img = img_.copy()
    for i in range(len(img_)):
        for j in range(len(img_[i])):
            mean = np.mean(img_[i,j])
            img[i,j] = [mean, mean, mean]
    return img



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

def convolution(img_: np.ndarray, filter: 'Filter') -> np.ndarray:
    """
    Optimized convolution with zero-padding handling.
    
    Args:
        img_: Input image (2D or 3D array)
        filter: Filter object with array, center and range attributes
        
    Returns:
        Convolved image (same shape as input)
    """
    # Create output array
    img = np.zeros_like(img_)
    h, w = img_.shape[:2]
    
    # Pre-calculate filter bounds
    k_min, k_max = -filter.left_range[0], filter.right_range[0]
    l_min, l_max = -filter.left_range[1], filter.right_range[1]
    
    # Get filter array and center
    filter_arr = filter.array
    fc_x, fc_y = filter.center
    
    # Vectorized implementation
    for i in range(h):
        for j in range(w):
            # Calculate valid ranges
            k_start = max(k_min, -i)
            k_end = min(k_max, h - i)
            l_start = max(l_min, -j)
            l_end = min(l_max, w - j)
            
            # Extract the relevant image region
            img_section = img_[i+k_start:i+k_end+1, j+l_start:j+l_end+1]
            
            # Extract the corresponding filter section
            filter_section = filter_arr[fc_x+k_start:fc_x+k_end+1, fc_y+l_start:fc_y+l_end+1]
            
            # Compute convolution
            if img_.ndim == 3:  # Color image
                for channel in range(img_.shape[2]):
                    img[i,j,channel] = abs(np.sum(img_section[..., channel] * filter_section))
            else:  # Grayscale
                img[i,j] = abs(np.sum(img_section * filter_section))
    
    return img


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

image = convolution(image, filter)

# image = get_magnitude(image_x, image_y)




plt.imshow(image)
plt.show()