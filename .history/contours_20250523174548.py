import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import time
from scipy.signal import fftconvolve, convolve2d


image = plt.imread("justdisappear.png").astype(np.float32)  # taille (573, 640, 3)
# image = plt.imread("shinsei.png").astype(np.float32)
# image = plt.imread("upiko.png").astype(np.float32)
# image = plt.imread("black_and_white.png").astype(np.float32)


image = image[::1, ::1, :]  # 5:taille (115, 128, 3), 4:taille (144, 160, 3)
image[:,:]/255




class Filter:
    def __init__(self, center: np.ndarray, array: np.ndarray):
        self.center = center    # np.array([i,j])
        self.array = array
        self.shape = np.array((len(array), len(array[0])))
        
        # print("shape ",self.shape)
        
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
    return np.stack((grayscale,)*3, axis=-1).astype(img_.dtype).astype(np.float32)


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
            # img[i,j] = abs(coef)
            img[i,j] = [coef, coef, coef]
    return img



# def convolution(img_: np.ndarray, filter: 'Filter'):
#     if img_.ndim != 3 or img_.shape[2] != 3:
#         raise ValueError("format : HxWx3")
    
#     # Extraction d'un seul canal (le vert par défaut)
#     # img_gray = img_[:, :, 1].astype(np.float32)
    
#     # Application convolution FFT
#     convolved = fftconvolve(img_[:, :, 0], filter.array, mode='same')
    
#     # Valeur absolue et normalisation
#     # convolved = np.abs(convolved)
#     # convolved = np.clip(convolved, 0, 255).astype(img_.dtype)
    
#     # Reconstruction des 3 canaux
#     img = np.stack((convolved,)*3, axis=-1)
#     return img


def convolution(img_: np.ndarray, filter: 'Filter') -> np.ndarray:
    # Vérification des dimensions
    if img_.ndim == 3:
        if not np.allclose(img_[:,:,0], img_[:,:,1]) or not np.allclose(img_[:,:,0], img_[:,:,2]):
            raise ValueError("L'image doit être en noir et blanc (canaux identiques)")
        img_2d = img_[:,:,0]  # Utilise un seul canal
    else:
        img_2d = img_

    # Conversion en float pour la précision
    img_2d = img_2d.astype(np.float32)
    
    # Application de la convolution 2D
    convolved = convolve2d(img_2d, filter.array, mode='same', boundary='fill', fillvalue=0)
    
    # Reconstruction des canaux si nécessaire
    if img_.ndim == 3:
        return np.stack((convolved,)*3, axis=-1).astype(np.float32)
    return convolved.astype(np.float32)



def get_magnitude(img_x, img_y, show_angle = False):
    magnitude = np.sqrt(img_x**2 + img_y**2)
    if not show_angle:
        return magnitude
    else:
        angles = np.arctan(img_y/img_x)[:,:,0]
        angles = np.where(img_x[:,:,0] < 0, angles + np.pi, angles)
    
        # Normalisation HSV
        h = (angles % (2*np.pi)) / (2*np.pi)    #[0,1]
        s = np.ones_like(h)     # saturation
        v = np.ones_like(h)     # value

        # hsv to rgb
        hsv = np.stack((h, s, v), axis=-1)
        
        rgb = hsv_to_rgb(hsv)
        img = rgb
        # img = (rgb).astype(np.uint8)
        return img * np.sqrt(img_x**2 + img_y**2)

def threshold(img, threshold):
    binary = np.where(img[:,:,0] > threshold, 1, 0)
    return np.stack((binary,)*3, axis=-1).astype(np.float32)

def zero_threshold(img, threshold):
    binary = np.where(np.abs(img[:,:,0]) < threshold, 1, 0)
    return np.stack((binary,)*3, axis=-1).astype(np.float32)

def combine_threshold(img1, img2):
    binary = np.where((img1[:,:,0] == img2[:,:,0]) & (img1[:,:,0] == 1), 1, 0)
    return np.stack((binary,)*3, axis=-1).astype(np.float32)

def extrem_threshold(img, threshold):
    return np.where(img > threshold, [1,1,1], np.where(img < -threshold, [-1,-1,-1], 0)).astype(np.float32)
    

def edge_detection_1(img, plot_differential = False,show_angle = False):
    image_x = convolution(img, filterx)
    image_y = convolution(img, filtery)
    if plot_differential:
        image_x_ = get_sign_color(image_x)
        image_y_ = get_sign_color(image_y)
        
        plot(image_x_)
        plot(image_y_)

    img_ = get_magnitude(image_x, image_y, show_angle)
    
    return img_

def edge_detection_2(img):
    image_x = convolution(img, filterx)
    image_y = convolution(img, filtery)
    image_xy = convolution(img, filterxy)
    image_xx = convolution(img, filterxx)
    image_yy = convolution(img, filteryy)
    magnitude = get_magnitude(image_x, image_y) + np.ones(img.shape) * 1e-8
    
    img_ = image_x ** 2 * image_xx + 2 * image_x * image_y * image_xy + image_y ** 2 * image_yy
    # img_ = np.stack((img_,)*3, axis=-1).astype(np.uint8)
    return  img_

    

def get_sign_color(img_):
    # (h, w, 3)
    mask = img_[:,:,:] >= 0

    img = np.where(
    mask[:,:,:],
        [0, 1, 0],
        [1, 0, 0]
    ) * np.abs(img_[:,:,0, np.newaxis])
    return img

def get_difference(img_):
    img = img_.copy()
    for i in range(len(img_)-1):
        for j in range(len(img_[i])-1):
            if (img_[i+1,j] * img_[i,j] == -1).all() or (img_[i+1,j+1] * img_[i,j] == -1).all() or (img_[i,j+1] * img_[i,j] == -1).all():
                img[i:i+2,j:j+2] = [1, 1, 1]
            else:
                img[i,j] = [0, 0, 0]
    return img

def det_hess(img):
    image_xy = convolution(img, filterxy)
    image_xx = convolution(img, filterxx)
    image_yy = convolution(img, filteryy)
    
    return np.abs(image_xx * image_yy - image_xy ** 2)

def plot(img):
    global plot_count
    if not multiplot:
        return
    if plot_count > rows * columns:
        print("TOO MUCH PLOTS")
        return
    fig.add_subplot(rows, columns, plot_count)
    plot_count += 1
    plt.imshow(img)
    plt.yticks([])
    

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

## d2fdxdy
filterxy = Filter(
    np.array([1,1]), 
    np.array([
        [1,-1],
        [-1,1]
        ])
)

## d2fdxdy
filterxx = Filter(
    np.array([0,1]), 
    np.array([
        [1,-2,1]
        ])
)

## d2fdxdy
filteryy = Filter(
    np.array([1,0]), 
    np.array([
        [1],
        [-2],
        [1]
        ])
)

laplacian_filter = Filter(
    np.array([1,1]), 
    np.array([
        [1,1,1],
        [1,-8,1],
        [1,1,1]
        ])
)


filtermean = Filter(
    np.array([1,1]), 
    (1/9)*np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
        ])
)

gaussian_filter_3x3 = Filter(
    np.array([1,1]), 
    (1/16)*np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
        ])
)


testfilter = Filter(
    np.array([1,1]), 
    np.array([
        [1,0,0],
        [1,0,-3],
        [1,0,0]
        ])
)



plot_count = 1
multiplot = True
multiplot = False

if multiplot:
    columns = 2
    rows = 2
    fig = plt.figure(figsize=(rows, columns))



## black and white image
# start_time = time.time()
image = black_and_white(image)
# plot(image)

# image_g = convolution(image, gaussian_filter_3x3)


# image1 = edge_detection_1(image)
# image1 = threshold(image1, .065)
# plot(image1)

# plot(image)
## noise
# noise = np.random.normal(0, .005, image.shape)[:,:,:1]
# image += noise
# image = np.clip(image, 0, 1).astype(image.dtype)

# plt.imshow(image)
# plot(image)




# image0 = convolution(image, filtermean)
# plot(image0)





# image0 = convolution(image, filtermean)

# plt.imshow(image0)
# plot(image0)




# end_time = time.time()
# print(end_time - start_time)


# image = convolution(image, gaussian_filter_3x3)
# plot(image3)

# image1 = convolution(image, testfilter)
# image1 = threshold(image1, .025)


# image1 = convolution(image, laplacian_filter)
# image1 = zero_threshold(image1, .03)

image *= 8
plt.imshow(image)

image1 = edge_detection_2(image)
# image1 = extrem_threshold(image1, .00008)
image1 = get_sign_color(image1)
image1 /= np.max(image1)

# image1 = get_difference(image1)



# image1 = threshold(image1, .01)
# plt.imshow(image1)

# plot(image1)


# plot(image)
# plot(image)

# image2 = det_hess(image)
# image2 = threshold(image2, .0005)
# image2 = image2/np.max(image2)
# image2 *= 10

# print(np.max(image1), np.where((image1 == np.max(image1))))

# image = convolution(image, gaussian_filter_3x3)
# image2 = edge_detection_2(image)
# image2 = threshold(image2, .025)
# plt.imshow(image2)
# plot(image2)



# image3 = combine_threshold(image1, image2)
# image3 = convolution(image, filterx)
# image3 = threshold(image3, 0.005)
# plot(image3)


# plt.imshow(image1)
# plot(image2)

# image4 = edge_detection_1(image)
# image4 = convolution(image, filterxx)
# image4 = threshold(image4, .008)
# plt.imshow(image4)
# plot(image4)

if multiplot:
    plt.subplots_adjust(
        left=0, right=1, bottom=0, top=1,
        wspace=0, hspace=0
    )



# plt.imshow(image1)
# plot(image1)


## plot of color gradient
gradient = np.linspace(0, 2*np.pi, 360)
h_grad = gradient / (2*np.pi)
rgb_grad = hsv_to_rgb(np.stack((h_grad, np.ones(360), np.ones(360)), -1))

height_factor = 60
rgb_grad = np.repeat(rgb_grad[np.newaxis, :, :], height_factor, axis=0)
# plot(rgb_grad)
# plt.imshow(rgb_grad)

plt.show()
