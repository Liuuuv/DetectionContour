import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import time
from scipy.signal import fftconvolve, convolve2d
import pywt
import pywt.data

# image1 = plt.imread("justdisappear.png").astype(np.float32)
# image1 = plt.imread("shinsei.png").astype(np.float32)
# image1 = plt.imread("upiko.png").astype(np.float32)
# image1 = plt.imread("black_and_white.png").astype(np.float32)
image1 = plt.imread("bnw_square.png").astype(np.float32)

# image2 = plt.imread("justdisappear.png").astype(np.float32)
# image2 = plt.imread("shinsei.png").astype(np.float32)
# image2 = plt.imread("upiko.png").astype(np.float32)
# image2 = plt.imread("black_and_white.png").astype(np.float32)
image2 = plt.imread("bnw_square.png").astype(np.float32)


image1 = image1[::2, ::2, :]
image2 = image2[::2, ::2, :]
image1[:,:]/255
image2[:,:]/255