from attention.attention_block import AttentionBlock
from LayerNorm.ln import Norm
from fnn.fnn_block import FNN
import numpy as np
from Utils.display_weights import show
# input_seq = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# input_seq = [[1, 1], [1, 1]]
# input_seq = [[1.11, 1.45, 0.99], [0.99, 1.32, 1.44], [0.88, 0.94, 1.22]]

# input_seq = [[1.11, 1.45, 0.99,1.00], [0.99, 1.32, 1.44,1.1], [0.88, 0.94, 1.22,.77]]
# output_seq=[[1,2],[2,3],[3,4]]

input_seq = [[0,0], [1,1], [1,0],[0,1]]
output_seq=[[1],[1],[0],[0]]

class TransformerBlock:
    def __init__(self, input_seq, output_seq,num_transformer_layers,fnn_hidden_size,fnn_hidden_layers):

        self.layers = []
        self.input_seq=np.array(input_seq)
        self.output_seq=np.array(output_seq)

        self.t_output_weight=np.random.rand(self.input_seq.shape[1],self.output_seq.shape[1])

        for _ in range(num_transformer_layers):

            ln_1 = Norm(self.input_seq)
            ln_1.weight_init()

            fnn=FNN(self.input_seq,hidden_size=fnn_hidden_size,hidden_layers=fnn_hidden_layers)
            fnn.weight_init()

            self.layers.append([ln_1,fnn])

        # show(self.layers)

    def forward_pass(self):

        self.output_from_last_layer=self.input_seq
        self.forward_pass_values=[]


        # print("=================================")
        abc=0
        for layer in self.layers:
            ln_1 : Norm=layer[0]
            fnn : FNN=layer[1]

            # print("-------- pass",abc)
            abc+=1
            # print("input recived",self.output_from_last_layer,end="\n\n")

            ln_1.forward(self.output_from_last_layer)
            ln_1.scale()
            # print("scaled _matrix",ln_1.scaled_matrix,end="\n\n")

            fnn.forward(input_from_last_layer=ln_1.scaled_matrix)
            # print("fnn output",fnn.f_pass_values[-1][0],end="\n\n")

            residual_connection_2=self.output_from_last_layer+fnn.f_pass_values[-1][0]
            # print("residual connection",residual_connection_2,end="\n\n")

            self.output_from_last_layer=residual_connection_2

            self.forward_pass_values.append(
                [
                    ln_1.normalized_matrix,
                    ln_1.scaled_matrix,
                    fnn.f_pass_values[-1],
                ]
            )
        self.t_output = self.output_from_last_layer @ self.t_output_weight
        self.loss = self.t_output - self.output_seq

        self.error = 0.5 * np.sum(self.loss ** 2)

    def backpropagation(self,lr):
        delta_t_out_weight= self.output_from_last_layer.T @ self.loss
        self.t_output_weight -= delta_t_out_weight * lr
        
        gradient=self.loss @ self.t_output_weight.T

        for layer in self.layers[::-1]:
            fnn:FNN = layer[-1]
            ln_1:Norm = layer[-2]

            res_gradient=gradient.copy()
            fnn.backpropagation(gradient_from_last_layer=gradient)
            fnn.update_weights(lr=lr)

            ln_1.create_jacobian()
            ln_1.backpropagation(gradient_from_last_layer=fnn.gradient_to_next_layer,jacobian_matrix=ln_1.jacobian_matrix)
            ln_1.update_weights(lr=lr)

            gradient=ln_1.gradient_to_next_layer+res_gradient
            # gradient=ln_1.gradient_to_next_layer

    

t = TransformerBlock(input_seq=input_seq,output_seq=output_seq,num_transformer_layers=2,fnn_hidden_size=6,fnn_hidden_layers=1)

EPOCHS=5
for _ in range(EPOCHS):
    t.forward_pass()
    print(f"PREDICTION -> {np.array(t.t_output).T} LOSS -> {np.array(t.loss).T} ERROR -> {t.error}")
    t.backpropagation(lr=0.01)


# l=Norm(input_seq=np.array(input_seq),verbose=True)
# l.forward(input_from_last_layer=np.array(input_seq))
