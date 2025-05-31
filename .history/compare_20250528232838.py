from contours import*

image_ref = plt.imread("black_and_white_ref.png").astype(np.float32)

image_contour = image
## pos : misses, neg : too much, 0 : good
minus_image = image - image_ref


miss_weight = .5
too_much_weight = 1

miss_array = minus_image[minus_image > 0]
score = np.sum()