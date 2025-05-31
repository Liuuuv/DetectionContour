from contours import*

image = convolution(image, gaussian_filter_3x3)


fourier = transformee_fourier(image)


fourier = np.log(1 + np.abs(fourier))
fourier /= np.max(fourier)
# print(fourier)
# plt.imshow(fourier, cmap='magma')
# plot(image)
# plot(fourier, cmap='magma')

plt.imshow(image)
# plt.imshow(fourier, cmap='magma')

# plt.colorbar()

plt.show()