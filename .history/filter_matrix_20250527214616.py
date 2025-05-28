from contours import*

matrix_size = (2,2)     # rows, columns
kernel = Filter([0,0],np.array([
    [0,0]
]))

## ab + c = [[a,b],c]
## [a,b] : a from kernel, b from matrix

## convolve
convolved_matrix = np.zeros(matrix_size)
for i in range(convolved_matrix.shape[0]):
    for j in range(convolved_matrix.shape[1]):
        coef = []
        for k in range(-filter.left_range[0], filter.right_range[0]):
                for l in range(-filter.left_range[1], filter.right_range[1]):
                    if not 0 <= i-k < convolved_matrix.shape[0] or not 0 <= j-l < convolved_matrix.shape[1]:
                        continue

                    # coef += img_[i-k,j-l][0]*filter.array[filter.center[0]+k,filter.center[1]+l]
                    coef.append([(filter.center[0]+k,filter.center[1]+l),i-k,j-l])
            