from contours import*
import scipy

studied_row = 100
max_column = 30
max_scale = 500

plot(image[studied_row-5:studied_row+6,:max_column])
plot(image[studied_row-25:studied_row+26,:max_column])

intensity_list = image[studied_row, :max_column, 0]
column_list = np.arange(0, max_column, 1)
column_list_fin = np.linspace(0, max_column, 2*max_column)
# column_list = np.arange(0, image.shape[1], 1)

print(column_list.shape, intensity_list.shape)

scatter_plot(column_list, intensity_list, marker='+')


cubic_interp = scipy.interpolate.interp1d(column_list, intensity_list, kind='cubic')

plt.title("1-D Interpolation")
plt.scatter(column_list_fin, cubic_interp(column_list_fin), '-', color="green")

plt.show()
