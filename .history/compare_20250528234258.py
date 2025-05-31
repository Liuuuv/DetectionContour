from contours import*

image_ref = plt.imread("black_and_white_ref.png").astype(np.float32)


image_ref = image_ref[::1, ::1, :,0]  # 5:taille (115, 128, 3), 4:taille (144, 160, 3)
image_ref[:,:]/255


image_contour = edge_detection_1(image)
image_contour = threshold(image_contour, .065)


## neg : misses, pos : too much, 0 : good
minus_image = image_contour - image_ref


miss_weight = .5
too_much_weight = 1

too_much_array = minus_image[minus_image > 0]
miss_array = minus_image[minus_image < 0]
score = np.sum(too_much_array * too_much_array + miss_array * miss_array)

plt.imshow(image_contour)
print(score)

plt.show()