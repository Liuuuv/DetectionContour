from contours import*
import scipy

studied_row = 200
max_column = 300


plot(image[studied_row-5:studied_row+6,:max_column])
plot(image[studied_row-25:studied_row+26,:max_column])

intensity_list = image[studied_row, :max_column, 0]
column_list = np.arange(0, max_column, 1)
column_list_fin = np.linspace(0, max_column-1, 2*max_column)
# column_list = np.arange(0, image.shape[1], 1)


scatter_plot(column_list, intensity_list, marker='+')


cubic_interp = scipy.interpolate.interp1d(column_list, intensity_list, kind='cubic')

spline = scipy.interpolate.UnivariateSpline(column_list, intensity_list, s=.1)

spline2 = scipy.interpolate.UnivariateSpline(column_list, intensity_list, s=0)


plt.plot(column_list_fin, cubic_interp(column_list_fin), '-', color="green")

# plt.plot(column_list_fin, spline(column_list_fin), '-', color="red")

# plt.plot(column_list_fin, spline2(column_list_fin), '-', color="orange")


interpol_intensity_list = []
alpha = 3
in_between = 1
for j in range(len(column_list)):
    if j >= 0 and j < len(column_list) - 1:
        for k in range(in_between+1):
            frac = k/in_between

            to_add = intensity_list[j] * (1 - ease_in_out(frac, alpha)) + intensity_list[j+1] * ease_in_out(frac, alpha)
            interpol_intensity_list.append(to_add)
    else:
        for _ in range(in_between+1):
            interpol_intensity_list.append(intensity_list[j])


print(intensity_list)
plt.plot(column_list_fin, interpol_intensity_list, color="red")



plot(image)

plt.show()
