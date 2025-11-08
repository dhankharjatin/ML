import numpy as np


def initialize_weights(input_seq):

    input_seq = np.array(input_seq)

    rows, columns = input_seq.shape

    # q = np.random.rand(columns, columns)
    # k = np.random.rand(columns, columns)
    # v = np.random.rand(columns, columns)
    # output_matrix = np.random.rand(columns, columns)

    # print(q, end="\n\n")
    # print(k, end="\n\n")
    # print(v, end="\n\n")

    q = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    k = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6],
        ]
    )
    v = np.array(
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4, 0.4],
        ]
    )

    wo = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # print(q, end="\n\n")
    # print(k, end="\n\n")
    # print(v, end="\n\n")

    # linear projection

    Q = input_seq @ q
    K = input_seq @ k
    V = input_seq @ v

    # print(Q, end="\n\n")
    # print(K, end="\n\n")
    # print(V, end="\n\n")

    return Q, K, V, wo


def split_into_heads(num_heads, Q, K, V):

    rows, columns = Q.shape

    if columns % num_heads != 0:
        print("invalid number of heads, Defaulting to num_head = 1")
        num_heads = 1

    Q = np.array_split(Q, num_heads, axis=1)
    K = np.array_split(K, num_heads, axis=1)
    V = np.array_split(V, num_heads, axis=1)

    # for i in Q:
    #     print(i, end="\n\n")
    # for i in K:
    #     print(i, end="\n\n")
    # for i in V:
    #     print(i, end="\n\n")

    return Q, K, V
