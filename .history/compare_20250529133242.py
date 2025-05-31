import scipy.optimize
from contours import*
import scipy

image_ref = plt.imread("black_and_white_ref.png").astype(np.float32)
image_ref = plt.imread("justdisappear_ref.png").astype(np.float32)


image_ref = image_ref[::1, ::1, 0:3]  # 5:taille (115, 128, 3), 4:taille (144, 160, 3)
image_ref[:,:]/255


image_contour = edge_detection_2(image)
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

    too_much_penalty = np.sum(minus_image > 0) * too_much_weight
    miss_penalty = np.sum(minus_image < 0) * miss_weight
    
    return too_much_penalty + miss_penalty

def find_best_threshold(img, tolerance = .0001):
    iteration_number = 0
    step = .0002
    current_threshold = 0
    
    current_score = np.inf
    
    while step > tolerance and iteration_number < 200:
        current_img = threshold(img, current_threshold)
        score = evaluate(current_img)
        if score <= current_score:
            current_threshold += step
            current_score = score
        else:
            step /= 2
        iteration_number += 1
    
    if iteration_number == 200:
        current_score = np.inf
    
    return current_score

def find_best_threshold(img):
    best_threshold = scipy.optimize.fminbound(lambda t:evaluate(threshold(img, t)),0,1)
    best_score = evaluate(threshold(img, best_threshold))
    print("best_threshold", best_threshold)
    return best_score


def plot_noise_score(img, st_deviation_list):
    score_list = []
    for st_deviation in st_deviation_list:
        print(st_deviation)
        
        ## get a mean*
        mean_list = []
        for _ in range(1):
            noise = np.random.normal(0, st_deviation, img.shape)[:,:,:1]
            noised_image = img + noise
            noised_image = np.clip(noised_image, 0, 1).astype(img.dtype)

            score = find_best_threshold(noised_image)
            mean_list.append(score)
        score_list.append(np.mean(mean_list))
        print(score)
    
    plt.plot(st_deviation_list, score_list)
    plt.grid()
    plt.xlabel("Ecart-type du bruit")
    plt.ylabel("Score")
    # plt.yscale('log')



# threshold_list = np.linspace(0.0001,.1,90)
# plot_evaluate(image1, threshold_list)

st_deviation_list = np.linspace(0,1,20)
plot_noise_score(image_contour, st_deviation_list)

# print(find_best_threshold(image_contour))

# plot(image)
# plot(image_contour)
# plot(image_ref)
# score = evaluate(image_contour)
# print(score)

plt.show()