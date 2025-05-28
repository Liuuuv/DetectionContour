from contours import*

matrix_size = (2,2)     # rows, columns
kernel = Filter([0,0],np.array([
    [0,0]
]))

## ab + c = [[a,b],c]
## [a,b] : a from kernel, b from matrix

## convolve
convolved_matrix = np.zeros(matrix_size, dtype=object)
for i in range(convolved_matrix.shape[0]):
    for j in range(convolved_matrix.shape[1]):
        coef = []
        for k in range(-kernel.left_range[0], kernel.right_range[0]):
                for l in range(-kernel.left_range[1], kernel.right_range[1]):
                    if not 0 <= i-k < convolved_matrix.shape[0] or not 0 <= j-l < convolved_matrix.shape[1]:
                        continue

                    # coef += img_[i-k,j-l][0]*filter.array[filter.center[0]+k,filter.center[1]+l]
                    coef.append([(kernel.center[0]+k,kernel.center[1]+l),(i-k+1,j-l+1)])
        
        convolved_matrix[i,j] = coef


i, j = 0, 0


        
def better_print(matrix):
    for i in range(matrix.shape[0]):
        string = ""
        for j in range(matrix.shape[1]):
            for index, factor in enumerate(matrix[i,j]):
                # print(factor,"jjjj")
                if index != 0:
                    string += " + "
                string += f"k_{factor[0]} * m_{factor[1]}"
            # string += str(matrix[i,j])
            string += "       "
        print(string)

better_print(convolved_matrix)