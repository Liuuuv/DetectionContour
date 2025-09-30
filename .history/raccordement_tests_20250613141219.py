from contours import*

studied_row = 100
max_scale = 500

plot(image[studied_row-5:studied_row+6,:])
plot(image[studied_row-25:studied_row+26,:])

intensity_list = image[studied_row, :, 0]
np.arange(image.shape[1])
plot(, intensity_list)



plt.show()
