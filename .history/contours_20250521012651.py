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
        
        
        
        print("ranges ", self.left_range, self.right_range)
        print(-self.left_range[0], self.right_range[0]-1)
        print(-self.left_range[1], self.right_range[1]-1)


def black_and_white(img_):
    img = img_.copy()
    for i in range(len(img_)):
        for j in range(len(img_[i])):
            mean = np.mean(img_[i,j])
            img[i,j] = [mean, mean, mean]
    return img



def convolution(img_: np.ndarray, filter: Filter):  # 0 outside
    img = img_.copy()
    for i in range(len(img)):
        for j in range(len(img[i])):
            coef = 0
            for k in range(-filter.left_range[0], filter.right_range[0]):
                for l in range(-filter.left_range[1], filter.right_range[1]):
                    if not 0 <= i-k < img.shape[0] or not 0 <= j-l < img.shape[1]:
                        continue
                    

                    coef += img[i-k,j-l][0]*filter.array[k,l]
            
            img[i,j] = abs(coef)
            # if coef >= 0:
            #     img[i,j] = [0, 0, coef]
            # else:
            #     img[i,j] = [abs(coef), 0, 0]
            
    return img


def get_magnitude(img_x, img_y):
    img = img_x.copy()
    for i in range(len(img_x)):
        for j in range(len(img_x[i])):
            magnitude = np.sqrt(img_x[i,j]**2 + img_y[i,j]**2)
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
#         [1,1,1],
#         [1,-8,1],
#         [1,1,1]
#         ])
# )

image = black_and_white(image)
image_x = convolution(image, filterx)
image_y = convolution(image, filtery)

image = get_magnitude(image_x, image_y)




plt.imshow(image)
plt.show()