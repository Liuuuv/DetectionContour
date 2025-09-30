import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import time
from scipy.signal import fftconvolve, convolve2d
import pywt
import pywt.data

import tools

image1 = plt.imread("justdisappear.png").astype(np.float32)
# image1 = plt.imread("shinsei.png").astype(np.float32)
# image1 = plt.imread("upiko.png").astype(np.float32)
# image1 = plt.imread("black_and_white.png").astype(np.float32)
# image1 = plt.imread("bnw_square.png").astype(np.float32)

# image2 = plt.imread("justdisappear.png").astype(np.float32)
# image2 = plt.imread("shinsei.png").astype(np.float32)
image2 = plt.imread("upiko.png").astype(np.float32)
# image2 = plt.imread("black_and_white.png").astype(np.float32)
# image2 = plt.imread("bnw_square.png").astype(np.float32)


image1 = image1[::2, ::2, :]
image2 = image2[::2, ::2, :]
image1[:,:]/255
image2[:,:]/255


def black_and_white(img_):
    # Convert to grayscale using luminance formula (better than simple mean)
    grayscale = np.dot(img_[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Stack the grayscale values into 3 channels
    return np.stack((grayscale,)*3, axis=-1).astype(img_.dtype).astype(np.float32)

def make_same_shape(img1, img2):
    shape1 = img1.shape[:2]
    shape2 = img2.shape[:2]
    
    if tools.lex_leq(shape1, shape2):
        img2 = img2[:shape1[0],:shape1[1],:]
    
    return img1, img2

def inter_img(img1, img2, t: float):
    return (1-t) * img1 + t * img2


# image1 = black_and_white(image1)
# image2 = black_and_white(image2)

image1, image2 = make_same_shape(image1, image2)

t_list = np.linspace(0,1,50)
plt.show()

for t in t_list:
    img = inter_img(image1, image2, t)
    plt.imshow(img)
    plt.pause(.1)