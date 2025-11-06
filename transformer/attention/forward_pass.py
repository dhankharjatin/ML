import numpy as np


def softmax(matrix):
    totals = []
    temp = []
    for i in matrix:
        total_temp = 0
        t = []
        for j in i:
            s = np.exp(j)
            t.append(s)
            total_temp += s

        temp.append(t)
        totals.append(total_temp)

    final = []

    for idx, i in enumerate(temp):
        z = []
        for j in i:
            z.append(j / totals[idx])

        final.append(z)

    print(final)

    return np.array(final)


def calculate_attention_score(Q, K, V):
    D_k = len(K)

    a = Q[0] @ K[0].T

    a = a / D_k ** (1 / 2)

    a = softmax(a)

    output_head = a @ V[0]

    print(output_head)
