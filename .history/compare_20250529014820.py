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

def find_best_threshold(img, tolerance = .0001):
    step = .001
    going_right = True
    
    last_threshold = -np.inf
    current_threshold = 0
    
    last_score = np.inf
    # current_img = threshold(img, current_threshold)
    # current_score = evaluate(current_img)
    current_score = np.inf
    
    
    while step > tolerance:
        # print(current_score, step, current_threshold) 
        current_img = threshold(img, current_threshold)
        score = evaluate(current_img)
        print(current_score, step, current_threshold, score)
        if score <= current_score:
            current_threshold += step
            current_score = score
        else:
            step /= 2
    
    return current_threshold


def plot_noise_score(img, st_deviation_list):
    for st_deviation in st_deviation_list:
        noise = np.random.normal(0, .05, image.shape)[:,:,:1]
        noised_image = image + noise
        noised_image = np.clip(noised_image, 0, 1).astype(image.dtype)

        threshold = find_best_threshold(noised_image)



threshold_list = np.linspace(0.0001,.1,90)
plot_evaluate(image1, threshold_list)

print(find_best_threshold(image1))

# plot(image)
# plot(image_contour)
# plot(image_ref)
# score = evaluate(image_contour)
# print(score)

plt.show()