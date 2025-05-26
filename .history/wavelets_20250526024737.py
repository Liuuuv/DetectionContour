from contours import*

## wavelet
# plot_wt(image,-1)



# level = 1
# x_imgs, y_imgs = edge_detection_wt_like(image, level)

# edge_image = np.zeros(image.shape)

# def weight_function(t):
#     return 1/t

# weights = [weight_function(level-i) for i in range(level)]
# weights = [weight_function(i+1) for i in range(level)]

# # weights = [0.1,0.2,0.3,0.4,0.5,1,0.2]


# for i in range(len(x_imgs)):
#     # x_img /= np.max(x_img)
#     # plot(x_img)
#     # plt.title("x" + str(i_x))
#     if i < len(weights):
#         weight = weights[i]
#     else:
#         weight = weights[-1]
#     edge_image += np.sqrt(x_imgs[i]**2 + y_imgs[i]**2) * weight
    
#     edge_image = threshold(edge_image, 0.1)
    
    
# # for i_y, y_img in enumerate(y_imgs):
# #     # y_img /= np.max(y_img)
# #     # plot(y_img)
# #     # plt.title("y" + str(i_y))
    

# plt.imshow(edge_image)





# plot(image)
# coefs = get_wt(image,5)

# ## denoise
# # sigma = np.median(np.abs(coefs[-1][0])) / 0.6745
# # N = np.max([image.shape[0], image.shape[1]])
# # T = 0.1 * sigma * np.sqrt(2 * np.log(N))
# # print('T', T)


# cA = coefs[0]
# details = coefs[1:]

# new_details = []

# for i, (cH, cV, cD) in enumerate(details):
#     if i == 0:
#         # cH = convolution(cH, mean_filter_2x2)
#         # cV = convolution(cV, mean_filter_2x2)
#         # cD = convolution(cD, mean_filter_2x2)
#         threshold_ = T
#         cH = threshold_to_zero(cH, threshold_)
#         cV = threshold_to_zero(cV, threshold_)
#         cD = threshold_to_zero(cD, threshold_)
#     new_details.append((cH, cV, cD))
    
# coefs_reconstructed = [cA] + new_details


# coefs_reconstructed = [coefs[0]] + [
#     (pywt.threshold(cH, T, mode='soft'),
#      pywt.threshold(cV, T, mode='soft'),
#      pywt.threshold(cD, T, mode='soft'))
#     for cH, cV, cD in coefs[1:]
# ]

# image1 = pywt.waverec2(coefs_reconstructed, 'sym4', mode='periodization')
# image1 = np.stack((image1,)*3, axis=-1).astype(np.float32)

# plt.imshow(image1)
# plot(image1)
print(pywt.wavelist())

studied_row = 100
max_scale = 200
# times = np.linspace(0,image.shape[1], 1000)
# sampling_period = np.diff(times).mean()
widths = np.geomspace(1, max_scale, num=100)
cwtmatr, freqs = pywt.cwt(image[studied_row,:,0], widths, 'Daubechies')

print(cwtmatr)

# # Surface = somme des valeurs absolues
# surface = np.sum(np.abs(cwtmatr))

# Affichage
plot(image[studied_row-5:studied_row+6,:])
plot(image[studied_row-25:studied_row+26,:])
# plot(np.abs(cwtmatr), cmap='jet')
# plot(np.abs(cwtmatr), extent=[0, image.shape[1], 1, max_scale+1], cmap='jet')

print(freqs)
print(cwtmatr)

plot(cwtmatr, cmap='jet')
plot(np.abs(cwtmatr), cmap='jet')

plt.colorbar()


plt.show()
