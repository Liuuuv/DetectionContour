from contours import*

studied_row = 100
max_scale = 500

plot(image[studied_row-5:studied_row+6,:])
plot(image[studied_row-25:studied_row+26,:])

intensity_list = image[studied_row, :, 0]
column_list = np.arange(0, image.shape[1], 1)

print(column_list.shape, intensity_list.shape)

plot(column_list, intensity_list, is_lists=True)



plt.show()
