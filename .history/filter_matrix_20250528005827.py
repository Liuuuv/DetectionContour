from contours import*

matrix_size = (4,4)     # rows, columns
kernel = Filter([1,1],np.array([
    [0,0,0],
    [0,0,0],
    [0,0,0]
]))

# matrix_size = (3,3)     # rows, columns
# kernel = Filter([1,1],np.array([
#     [0,0],
#     [0,0]
# ]))

## ab + c = [[a,b],c]
## [a,b] : a from matrix, b from kernel

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
                    coef.append([(i-k+1,j-l+1),(kernel.center[0]+k,kernel.center[1]+l)])
        
        convolved_matrix[i,j] = coef


# vectorize
square_to_column_dict = {}
vectorized_matrix = []
i, j = 0, 0
for _ in range(matrix_size[0] * matrix_size[1]):
    vectorized_matrix.append(convolved_matrix[i,j])
    square_to_column_dict[(i,j)] = _
    
    j +=1
    if j >= matrix_size[1]:
        j = 0
        i += 1

convolution_matrix = np.zeros((matrix_size[0]*matrix_size[1], matrix_size[0]*matrix_size[1]), dtype=object)
for i in range(matrix_size[0] * matrix_size[1]):
    for product in vectorized_matrix[i]:
        convolution_matrix[i,square_to_column_dict[(product[0][0]-1,product[0][1]-1)]] = product[1]





        
def better_print_convoled_matrix(matrix):
    for i in range(matrix.shape[0]):
        string = ""
        for j in range(matrix.shape[1]):
            for index, factor in enumerate(matrix[i,j]):
                # print(factor,"jjjj")
                if index != 0:
                    string += " + "
                string += f"m_{factor[0]} * k_{factor[1]}"
            # string += str(matrix[i,j])
            string += "       "
        print(string)
    
    

import numpy as np

def print_perfectly_aligned(matrix):
    # Convertir tous les éléments en chaînes formatées
    str_matrix = []
    max_len = 0
    
    # Première passe pour déterminer la largeur maximale
    for row in matrix:
        str_row = []
        for elem in row:
            if isinstance(elem, tuple):
                s = f"({elem[0]:>2}, {elem[1]:<2})"
            else:
                s = f"{elem:>5}"
            str_row.append(s)
            if len(s) > max_len:
                max_len = len(s)
        str_matrix.append(str_row)
        print()
    
    # Configurer l'affichage pour éviter la troncature
    import sys
    # np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    
    # Deuxième passe pour appliquer le padding uniforme
    for i, str_row in enumerate(str_matrix):
        padded_row = [elem.ljust(max_len + 2) for elem in str_row]
        line = "[" + " ".join(padded_row) + "]"
        
        # Forcer l'écriture ligne par ligne
        sys.stdout.write(line + "\n")
        sys.stdout.flush()  # Vider le buffer immédiatement

    # Réinitialiser les paramètres d'affichage
    # np.set_printoptions(threshold=None, linewidth=None)


# better_print_convoled_matrix(convolved_matrix)
# print(vectorized_matrix)

# print(convolution_matrix)
print_perfectly_aligned(convolution_matrix)