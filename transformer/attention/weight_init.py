import numpy as np


def initialize_weights(input_seq, num_heads):
    r, c = inp.shape

    q = np.random.rand(c, c)
    k = np.random.rand(c, c)
    v = np.random.rand(c, c)

    print(q, end="\n\n")
    print(k, end="\n\n")
    print(v, end="\n\n")

    print("\n=========\n")

    Q = input_seq @ q
    K = input_seq @ k
    V = input_seq @ v

    print(Q, end="\n\n")
    print(K, end="\n\n")
    print(V, end="\n\n")

    print("\n=======\n")


inp = np.random.rand(3, 4)

initialize_weights(inp)
