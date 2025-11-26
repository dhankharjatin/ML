from attention.attention_block import AttentionBlock
from LayerNorm.ln import Norm
from fnn.fnn_block import FNN
import numpy as np
from Utils.display_weights import show

# input_seq = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# input_seq = [[1, 2], [5, 6], [9, 10]]
input_seq = [[1.11, 1.45, 0.99], [0.99, 1.32, 1.44], [0.88, 0.94, 1.22]]


class TransformerBlock:
    def __init__(self, input_seq, num_layers, num_heads):

        self.layers = []
        input_seq=np.array(input_seq)

        for _ in range(num_layers):

            ln_1 = Norm(input_seq)
            ln_1.weight_init()

            attention_block = AttentionBlock(
                num_heads=num_heads,
                input_seq=input_seq,
            )
            attention_block.weight_init()

            ln_2 = Norm(input_seq)
            ln_2.weight_init()

            fnn=FNN(input_seq,hidden_size=3,hidden_layers=2)
            fnn.weight_init()

            self.layers.append([ln_1,attention_block,ln_2,fnn])

        show(self.layers)
            # print("\n\n************ layer **************\n")

            # # layer norm ------------------------
            # l_n = Norm(input_to_next_layer)
            # l_n.forward()
            # l_n.weight_init()
            # l_n.scale()

            # print("=========layer norm=========")
            # for i in l_n.normalized_matrix:
            #     print(i)
            # for i in l_n.scaled_matrix:
            #     print(i)

            # # MHA ------------------------
            # a_b = AttentionBlock(input_seq=l_n.scaled_matrix, num_heads=2)
            # a_b.weight_init()
            # a_b.forward_pass()

            # print("=========MHA=========")
            # for i in a_b.output_MHA:
            #     print(i)

            # # residual  ------------------------
            # residual_connection_1 = np.array(input_to_next_layer) + a_b.output_MHA

            # print("=========residual connection=========")
            # for i in residual_connection_1:
            #     print(i)

            # # layer norm ------------------------
            # l_n_2 = Norm(residual_connection_1)
            # l_n_2.forward()
            # l_n_2.weight_init()
            # l_n_2.scale()

            # print("=========layer norm=========")
            # for i in l_n_2.normalized_matrix:
            #     print(i)
            # for i in l_n_2.scaled_matrix:
            #     print(i)

            # # FNN  ------------------------
            # fnn = FNN(l_n_2.scaled_matrix, hidden_layers=3, hidden_size=2)
            # fnn.weight_init()
            # fnn.forward()

            # # residual  ------------------------
            # residual_connection_2 = residual_connection_1 + fnn.output_fnn

            # print("=========residual connection=========")
            # for i in residual_connection_2:
            #     print(i)

            # input_to_next_layer = residual_connection_2


t = TransformerBlock(input_seq=input_seq,num_heads=2,num_layers=2)
