from attention.attention_block import AttentionBlock
from LayerNorm.ln import Norm
from fnn.fnn_block import FNN
import numpy as np
from Utils.display_weights import show

# input_seq = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# input_seq = [[1, 2], [5, 6], [9, 10]]
input_seq = [[1.11, 1.45, 0.99], [0.99, 1.32, 1.44], [0.88, 0.94, 1.22]]


class TransformerBlock:
    def __init__(self, input_seq, num_transformer_layers, num_attention_heads,fnn_hidden_size,fnn_hidden_layers):

        self.layers = []
        self.input_seq=np.array(input_seq)

        for _ in range(num_transformer_layers):

            ln_1 = Norm(self.input_seq)
            ln_1.weight_init()

            attention_block = AttentionBlock(
                num_heads=num_attention_heads,
                input_seq=self.input_seq,
            )
            attention_block.weight_init()

            ln_2 = Norm(self.input_seq)
            ln_2.weight_init()

            fnn=FNN(self.input_seq,hidden_size=fnn_hidden_size,hidden_layers=fnn_hidden_layers)
            fnn.weight_init()

            self.layers.append([ln_1,attention_block,ln_2,fnn])

        # show(self.layers)

    def forward_pass(self):

        self.output_from_last_layer=self.input_seq
        self.forward_pass_values=[]


        for layer in self.layers:
            ln_1 : Norm=layer[0]
            attention : AttentionBlock=layer[1]
            ln_2 : Norm=layer[2]
            fnn : FNN=layer[3]


            ln_1.forward(self.output_from_last_layer)
            ln_1.scale()

            attention.forward_pass(input_from_last_layer=ln_1.scaled_matrix)
            
            residual_connection_1=self.output_from_last_layer+attention.output_MHA

            ln_2.forward(residual_connection_1)
            ln_2.scale()

            fnn.forward(input_from_last_layer=ln_2.scaled_matrix)

            residual_connection_2=residual_connection_1+fnn.f_pass_values[-1]

            self.output_from_last_layer=residual_connection_2

            self.forward_pass_values.append(
                [
                    ln_1.normalized_matrix,
                    ln_1.scaled_matrix,
                    attention.output_MHA,
                    residual_connection_1,
                    ln_2.normalized_matrix,
                    ln_2.scaled_matrix,
                    fnn.f_pass_values[-1],
                    residual_connection_2,
                ]
            )


# t = TransformerBlock(input_seq=input_seq,num_attention_heads=2,num_transformer_layers=2,fnn_hidden_size=3,fnn_hidden_layers=2)
# t.forward_pass()

# for i in t.forward_pass_values:

#     print("\n\n------------ ------------------\n")
#     for j in i:
#         print(j,end="\n\n")

# ==========================================================================================

# f=FNN(input_seq=np.array(input_seq),hidden_size=1,hidden_layers=2)
# f.weight_init()
# for i in f.weights:
#     print(i,end="\n\n")
# f.forward(input_from_last_layer=np.array(input_seq))
# f.backpropagation(np.eye(3,2))

l=Norm(input_seq=np.array(input_seq))
l.weight_init()
l.forward(input_from_last_layer=np.array(input_seq))

l.create_jacobian()
l.backpropagation(np.eye(3,3),l.jacobian_matrix)