from contours import*

image_ref = plt.imread("black_and_white_ref.png").astype(np.float32)


image_ref = image_ref[::1, ::1, 0:3]  # 5:taille (115, 128, 3), 4:taille (144, 160, 3)
image_ref[:,:]/255


image_contour = edge_detection_1(image)
# image_contour = threshold(image_contour, .2)

def evaluate(img):
    ## neg : misses, pos : too much, 0 : good
    minus_image = img - image_ref


    miss_weight = 1
    too_much_weight = 1

    too_much_array = np.where(minus_image > 0, 1, 0)
    miss_array = np.where(minus_image < 0, 1, 0)
    score = np.sum(too_much_array * too_much_weight + miss_array * miss_weight)

    return score

score_list = []
threshold_list = np.linspace(0.0001,.2,30)
for threshold_ in threshold_list:
    img = threshold(image_contour, threshold_)
    score = evaluate(img)
    
    score_list.append(score)

plt.plot(threshold_list, score_list, marker='+')
plt.grid()


# plot(image)
# plot(image_contour)
# plot(image_ref)
# score = evaluate(image_contour)
# print(score)

plt.show()