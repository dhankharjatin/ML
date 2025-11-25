from attention.attention_block import AttentionBlock
from LayerNorm.ln import Norm
from fnn.fnn_block import FNN
import numpy as np

# input_seq = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
input_seq = [[1, 2], [5, 6], [9, 10]]


class Transformer:
    def __init__(self, num_layers):
        self.layers = []

        for _ in range(num_layers):

            # layer norm ------------------------
            l_n = Norm(np.array(input_seq))
            l_n.forward()
            l_n.weight_init()
            l_n.scale()

            print("=========layer norm=========")
            for i in l_n.normalized_matrix:
                print(i)
            for i in l_n.scaled_matrix:
                print(i)

            # MHA ------------------------
            a_b = AttentionBlock(input_seq=l_n.scaled_matrix,num_heads=2)
            a_b.weight_init()
            a_b.forward_pass()

            print("=========MHA=========")
            for i in a_b.output_MHA:
                print(i)

            
            # residual  ------------------------
            residual_connection= np.array(input_seq) + a_b.output_MHA
            
            print("=========residual connection=========")
            for i in residual_connection:
                print(i)

            # FNN  ------------------------
            fnn = FNN(residual_connection,hidden_layers=3,hidden_size=2)
            fnn.weight_init()
            fnn.forward()

t = Transformer(1)
