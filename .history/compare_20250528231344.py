from contours import*

image_ref = plt.imread("black_and_white_ref.png").astype(np.float32)


## pos : misses, neg
minus_image = image - image_ref
