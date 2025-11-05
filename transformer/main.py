import numpy as np
from attention.weight_init import initialize_weights,split_into_heads

input_seq=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]

Q,K,V=initialize_weights(input_seq)

split_into_heads(2,Q,K,V)