from contours import*

# image = convolution(image, gaussian_filter_3x3)

image = np.zeros_like(image)
print(image)
# np.stack((img,)*3, axis=-1).astype(np.float32)
# image[0,0] = 


fourier = transformee_fourier(image)


fourier = np.log(1 + np.abs(fourier))
fourier /= np.max(fourier)
# print(fourier)
# plt.imshow(fourier, cmap='magma')
# plot(image)
# plot(fourier, cmap='magma')

# plt.imshow(image)
plt.imshow(fourier, cmap='magma')

# plt.colorbar()

plt.show()