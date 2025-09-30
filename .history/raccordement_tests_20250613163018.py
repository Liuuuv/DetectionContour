from contours import*
import scipy

studied_row = 200
max_column = 50

plot(image[studied_row-5:studied_row+6,:max_column])
plot(image[studied_row-25:studied_row+26,:max_column])

intensity_list = image[studied_row, :max_column, 0]
column_list = np.arange(0, max_column, 1)
column_list_fin = np.linspace(0, max_column-1, 2*max_column)
# column_list = np.arange(0, image.shape[1], 1)


scatter_plot(column_list, intensity_list, marker='+')


cubic_interp = scipy.interpolate.interp1d(column_list, intensity_list, kind='cubic')

spline = scipy.interpolate.UnivariateSpline(column_list, intensity_list, s=.1)

spline2 = scipy.interpolate.UnivariateSpline(column_list, intensity_list, s=1)


plt.plot(column_list_fin, cubic_interp(column_list_fin), '-', color="green")

plt.plot(column_list_fin, spline(column_list_fin), '-', color="red")

plt.plot(column_list_fin, spline2(column_list_fin), '-', color="orange")

plot(image)

plt.show()
