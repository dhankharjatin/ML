import numpy as np

def softmax(matrix):
    softmax_matrix=[]
    for row in matrix:
        row=row-np.max(row)
        row=np.exp(row)
        row=row/np.sum(row)

        softmax_matrix.append(row)

    return softmax_matrix


a=np.array([[1,1.1,1.2],[1.5,1.6,1.7]])

b= softmax(a)

print(a,b)

# b=np.array([([1.44328570e-66, 1.20136826e-33, 1.00000000e+00]), ([6.52102707e-172, 2.55363018e-086, 1.00000000e+000]), ([2.94631853e-277, 5.42800012e-139, 1.00000000e+000])])

# def softmax_jacobian(b_row):
#     b = b_row.reshape(-1, 1)  # column vector
#     J = np.diagflat(b) - np.dot(b, b.T)
#     return J

# jacobians = np.array([softmax_jacobian(row) for row in b])
# print(jacobians)




