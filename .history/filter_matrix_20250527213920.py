from contours import*

matrix_size = (2,2)     # rows, columns
kernel_size = (1,2)

## ab + c = [[a,b],c]
## [a,b] : a from kernel, b from matrix

## convolve
convolved_matrix = np.zeros(matrix_size)
for i in range(convolved_matrix.shape[0]):
    for j in range(convolved_matrix.shape[1]):
        coef = [[(),()]]