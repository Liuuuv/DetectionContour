from contours import*

image_ref = plt.imread("black_and_white_ref.png").astype(np.float32)
image_ref = plt.imread("justdisappear_ref.png").astype(np.float32)


image_ref = image_ref[::1, ::1, 0:3]  # 5:taille (115, 128, 3), 4:taille (144, 160, 3)
image_ref[:,:]/255


image_contour = edge_detection_1(image)
# image_contour = threshold(image_contour, .2)

def plot_evaluate(img, threshold_list):
    score_list = []
    for threshold_ in threshold_list:
        img = threshold(image_contour, threshold_)
        score = evaluate(img)
        
        score_list.append(score)

    plt.plot(threshold_list, score_list, marker='+')
    plt.grid()


def evaluate(img):
    ## neg : misses, pos : too much, 0 : good
    minus_image = img - image_ref


    miss_weight = 1.1
    too_much_weight = 1

    too_much_array = np.where(minus_image > 0, 1, 0)
    miss_array = np.where(minus_image < 0, 1, 0)
    score = np.sum(too_much_array * too_much_weight + miss_array * miss_weight)

    return score

def find_best_threshold(img, step, tolerance = .01):
    threshold_min = 0
    threshold_max = 1
    
    score_min = threshold(img, threshold_min)
    score_max = threshold(img, threshold_max)
    
    
    while threshold_max - threshold_min > tolerance:
        mid_threshold = (threshold_max + threshold_min)/2
        mid_img = threshold(img, mid_threshold)
        mid_score = evaluate(mid_img)
        
        if mid_score 


def plot_noise_score(img, st_deviation_list):
    for st_deviation in st_deviation_list:
        noise = np.random.normal(0, .05, image.shape)[:,:,:1]
        noised_image = image + noise
        noised_image = np.clip(noised_image, 0, 1).astype(image.dtype)

        threshold = find_best_threshold(noised_image)



threshold_list = np.linspace(0.0001,.1,90)
plot_evaluate(image1, threshold_list)

# plot(image)
# plot(image_contour)
# plot(image_ref)
# score = evaluate(image_contour)
# print(score)

plt.show()