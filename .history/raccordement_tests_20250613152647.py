from contours import*
import scipy

studied_row = 100
max_column = 30
max_scale = 500

plot(image[studied_row-5:studied_row+6,:max_column])
plot(image[studied_row-25:studied_row+26,:max_column])

intensity_list = image[studied_row, :max_column, 0]
column_list = np.arange(0, max_column, 1)
# column_list = np.arange(0, image.shape[1], 1)

print(column_list.shape, intensity_list.shape)

scatter_plot(column_list, intensity_list, marker='+')


cubic_interp = scipy.interpolate.interp1d(column_list, intensity_list, kind='cubic')

plt.title("1-D Interpolation")
plt.plot(column_list, cubic_interp(intensity_list), '-', color="green")

plt.show()
