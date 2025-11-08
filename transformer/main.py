import numpy as np
from attention.weight_init import initialize_weights, split_into_heads
from attention.forward_pass import calculate_attention_score

input_seq = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

q, k, v, wo = initialize_weights(input_seq)

Q, K, V = split_into_heads(2, q, k, v)

calculate_attention_score(Q, K, V, wo)

f = [[11, 11, 11, 11], [11, 11, 11, 11], [11, 11, 11, 11]]

print(np.array(input_seq) + np.array(f))
