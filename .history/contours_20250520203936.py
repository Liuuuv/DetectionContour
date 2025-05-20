import matplotlib.pyplot as plt
import numpy as np

image = plt.imread("justdisappear.png")     # taille (573, 640, 3)




image = image[::4, ::4, :]  # 5:taille (115, 128, 3), 4:taille (144, 160, 3)

class Filter:
    def __init__(self, center: np.ndarray, array: np.ndarray):
        self.center = center    # np.array([i,j])
        self.array = array
        self.shape = np.array((len(array[0]), len(array)))
        
        self.right_range = self.shape - self.center
        self.left_range = self.shape - self.right_range
        
        
        self.right_range -= np.array([1,1])
        
        print(self.right_range, self.left_range)


def black_and_white(img_):
    img = img_.copy()
    for i in range(len(img_)):
        for j in range(len(img_[i])):
            mean = np.mean(img_[i,j])
            img[i,j] = [mean, mean, mean]
    return img

# image = black_and_white(image)

def convolution(img_: np.ndarray, filter: Filter):
    img = img_.copy()
    for i in range(len(img)):
        for j in range(len(img[i])):
            pass
            
            
    return img


Filter(np.array([1,1]), 
       np.array([
           [1,1,1,1],
           [1,1,1,1],
           [1,1,1,1]
           ])
)


plt.imshow(image)
plt.show()