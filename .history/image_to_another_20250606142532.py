

# image1 = plt.imread("justdisappear.png").astype(np.float32)  # taille (573, 640, 3)
# image1 = plt.imread("shinsei.png").astype(np.float32)
# image1 = plt.imread("upiko.png").astype(np.float32)
# image1 = plt.imread("black_and_white.png").astype(np.float32)
image = plt.imread("bnw_square.png").astype(np.float32)


image = image[::2, ::2, :]  # 5:taille (115, 128, 3), 4:taille (144, 160, 3)
image[:,:]/255