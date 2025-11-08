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

    return np.array(final)


def calculate_attention_score(Q, K, V, wo):
    D_k = len(K)

    output_matrix = []
    for idx in range(len(Q)):
        score = Q[idx] @ K[idx].T

        # scaled scores
        scaled_score = score / D_k ** (1 / 2)

        scaled_score = softmax(scaled_score)

        output = scaled_score @ V[0]

        output_matrix.append(output)

    combined_matrix = np.concat(output_matrix, axis=1)

    output_MHA = combined_matrix @ wo

    print(output_MHA)
